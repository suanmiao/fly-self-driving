#!/usr/bin/env python3
"""Record one closed-loop episode as per-tick state, for an external replay renderer.

Writes <out>.json (scene + per-tick poses + events) and <out>.npz (the same tick arrays plus
the per-decision neural activity of every neuron with a measured soma position). Nothing here
depends on the training renderer except that the policy is driven by the same 64x32 frames.

    python record_episode.py --checkpoint runs/<run>/checkpoint.pt --seed 9000 --steps 500 --out episodes/ep-9000
"""
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import pyarrow.feather as feather
sys.path.insert(0, str(Path(os.environ.get('FLYHARD_ROOT', Path(__file__).resolve().parents[3] / 'flyhard')) / 'src'))  # sibling flyhard checkout, or set FLYHARD_ROOT
import street_env as E
from train_street import StreetPolicy, DensePolicy


def world(road, x, lateral):
    """Road coordinates (x along, lateral left of the centreline) to world x, y and yaw."""
    x = np.asarray(x, dtype=float); lateral = np.asarray(lateral, dtype=float)
    return x, road.centre(x) + lateral, road.heading(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--graph', default=os.environ.get('FLYHARD_ROOT', '../../../flyhard') + '/data/graph-traced-v1')
    p.add_argument('--seed', type=int, default=9000)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False); cfg = ck['config']
    nodes = feather.read_table(Path(args.graph)/'nodes.feather')
    classes = np.asarray(nodes['superclass'].fill_null('').to_pylist())
    soma = np.array([v if v is not None else [np.nan]*3 for v in nodes['somaLocation'].to_pylist()])
    valid = np.flatnonzero(np.isfinite(soma).all(axis=1))
    is_conn = cfg.get('model', 'connectome') == 'connectome'
    if is_conn:
        graph = dict(np.load(Path(args.graph)/'graph.npz'))
        if cfg['graph_mode'] == 'shuffled':
            graph['col'] = np.random.default_rng(20260910).permutation(len(graph['crow'])-1)[graph['col']]
        policy = StreetPolicy(graph, np.flatnonzero(classes == 'ol_sensory'), np.flatnonzero(classes == 'vnc_motor'), cfg['seed'])
    else:
        policy = DensePolicy(cfg['hidden'] if cfg['model'] == 'mlp' else 0)
    policy.load_state_dict(ck['model']); policy.to(args.device).eval()

    road = E.Road(args.seed); car = E.Car(road, np.random.default_rng(args.seed+7919))
    ticks, activity, motor, failure = [], [], [], None
    carried = torch.zeros(policy.core.n, 1, device=args.device) if is_conn else None
    for _ in range(args.steps):
        image = E.render(car)
        with torch.no_grad():
            xi = torch.as_tensor(image[None], device=args.device)
            if is_conn:
                carried = policy.step_state(carried, xi); command = float(policy.readout(carried)[0])
                activity.append(carried[valid, 0].cpu().numpy().astype(np.float16))
                motor.append(carried[policy.motor_ids, 0].cpu().numpy())
            else:
                command = float(policy(xi)[0])
        tx, ty, tyaw = world(road, *road.traffic_positions(car.t).T) if len(road.traffic) else (np.zeros(0),)*3
        ox, oy, oyaw = world(road, *road.oncoming_positions(car.t).T) if len(road.oncoming) else (np.zeros(0),)*3
        ticks.append({'t': round(car.t, 3), 'x': car.x, 'y': car.y, 'yaw': car.theta, 'steer': car.steer, 'command': command,
                      'lateral': car.lateral, 'lane_offset': car.lateral - E.LANE_CENTRE, 'overtaking_expert_flag': bool(E.overtaking(car)),
                      'traffic': [[float(a), float(b), float(c)] for a, b, c in zip(tx, ty, tyaw)],
                      'oncoming': [[float(a), float(b), float(c + np.pi)] for a, b, c in zip(ox, oy, oyaw)]})
        car.step(command); failure = car.crashed()
        if failure:
            break
    px, py, pyaw = world(road, road.parked[:, 0], road.parked[:, 1]) if len(road.parked) else (np.zeros(0),)*3
    xs = np.arange(0.0, road.length, 1.0)
    scene = {'seed': args.seed, 'task_version': E.TASK_VERSION, 'checkpoint': args.checkpoint, 'dt': E.DT, 'speed_mps': E.SPEED,
             'lane_width': E.LANE_W, 'road_half_width': E.ROAD_HALF, 'right_lane_centre_lateral': E.LANE_CENTRE,
             'vehicle_half_size': [E.CAR_HALF_L, E.CAR_HALF_W], 'car_box': 'length 4.2 m, width 1.8 m, body height 1.5 m, cabin to 2.1 m',
             'coordinates': 'world x along the road, y positive to the LEFT of travel, z up; yaw in radians, 0 = +x',
             'centreline': [[float(x), float(y)] for x, y in zip(xs, road.centre(xs))],
             'buildings': [{'x0': r[0], 'x1': r[1], 'lateral0': r[2], 'lateral1': r[3], 'z0': r[4], 'z1': r[5], 'albedo': r[6],
                            'kind': 'lamp_post' if r[3]-r[2] < 0.5 else 'building',
                            'note': 'lateral relative to centreline at the box x; add centreline y for world y'} for r in road.static.tolist()],
             'parked_cars': [[float(a), float(b), float(c)] for a, b, c in zip(px, py, pyaw)],
             'traffic_speed_mps': E.TRAFFIC_SPEED, 'oncoming_speed_mps': E.ONCOMING_SPEED,
             'result': {'steps': len(ticks), 'failure': failure, 'distance_m': car.x},
             'neural': {'neurons_with_soma': int(len(valid)), 'note': 'activity[t, i] is the tanh rate state of neuron node_index[i] after decision t; '
                        'brain panel convention: colour each neuron on its own |max| over the episode, orange positive, blue negative'}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'scene': scene, 'ticks': ticks}))
    arrays = {'t': np.array([k['t'] for k in ticks]), 'pose': np.array([[k['x'], k['y'], k['yaw'], k['steer'], k['command']] for k in ticks]),
              'soma_xyz': soma[valid].astype(np.float32), 'node_index': valid, 'body_ids': np.asarray(nodes['bodyId'].to_pylist())[valid] if 'bodyId' in nodes.column_names else valid,
              'superclass': classes[valid]}
    if is_conn:
        arrays['activity'] = np.stack(activity); arrays['motor'] = np.stack(motor).astype(np.float32)
    np.savez_compressed(out.with_suffix('.npz'), **arrays)
    print(json.dumps({'seed': args.seed, 'steps': len(ticks), 'failure': failure, 'distance_m': round(car.x, 1),
                      'json': str(out.with_suffix('.json')), 'npz': str(out.with_suffix('.npz'))}))


if __name__ == '__main__':
    main()
