#!/usr/bin/env python3
"""Closed-loop evaluation with all roads stepped in lockstep, one batched
forward pass per control tick. Same environment, same per-road logic as
`drive_env.rollout`; only the batching differs, so results match the serial
evaluator up to float summation order and run ~20x faster on the GPU."""
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import pyarrow.feather as feather
sys.path.insert(0, str(Path(os.environ.get('FLYHARD_ROOT', Path(__file__).resolve().parents[3] / 'flyhard')) / 'src'))  # sibling flyhard checkout, or set FLYHARD_ROOT
import drive_env as E
from train_vision import VisionPolicy


def load_policy(checkpoint_path, graph_path, device):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    graph = dict(np.load(Path(graph_path)/'graph.npz'))
    if ck['config']['graph_mode'] == 'shuffled':
        graph['col'] = np.random.default_rng(20260910).permutation(len(graph['crow'])-1)[graph['col']]
    nodes = feather.read_table(Path(graph_path)/'nodes.feather')
    classes = np.asarray(nodes['superclass'].fill_null('').to_pylist())
    policy = VisionPolicy(graph, np.flatnonzero(classes == 'ol_sensory'),
                          np.flatnonzero(classes == 'vnc_motor'), ck['config']['seed'])
    policy.load_state_dict(ck['model']); policy.to(device).eval()
    return policy, ck['config']


def evaluate_batched(policy, device, seeds, steps, obstacles=True):
    roads = [E.Road(s, obstacles=obstacles) for s in seeds]
    cars = [E.Car(r, np.random.default_rng(s+7919)) for r, s in zip(roads, seeds)]
    lateral = [[] for _ in seeds]; failure = [None]*len(seeds); active = list(range(len(seeds)))
    for _ in range(steps):
        if not active:
            break
        images = np.stack([E.render(cars[i]) for i in active])
        with torch.no_grad():
            commands = policy(torch.as_tensor(images, device=device)).cpu().numpy()
        still = []
        for i, cmd in zip(active, commands):
            cars[i].step(float(cmd)); lateral[i].append(abs(cars[i].lateral))
            failure[i] = cars[i].crashed()
            if not failure[i]:
                still.append(i)
        active = still
    results = [{'seed': s, 'steps': len(lateral[i]), 'failure': failure[i],
                'mean_abs_lateral': float(np.mean(lateral[i])), 'max_abs_lateral': float(np.max(lateral[i])),
                'distance_m': float(cars[i].x)} for i, s in enumerate(seeds)]
    return {'episodes': len(results), 'completed': sum(r['failure'] is None for r in results),
            'off_road': sum(r['failure'] == 'off_road' for r in results),
            'obstacle_hits': sum(r['failure'] == 'obstacle' for r in results),
            'mean_abs_lateral_m': float(np.mean([r['mean_abs_lateral'] for r in results])),
            'mean_distance_m': float(np.mean([r['distance_m'] for r in results])),
            'per_episode': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--graph', default=os.environ.get('FLYHARD_ROOT', '../../../flyhard') + '/data/graph-traced-v1')
    p.add_argument('--roads', type=int, default=20)
    p.add_argument('--first-road', type=int, default=9000)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    policy, config = load_policy(args.checkpoint, args.graph, device)
    seeds = list(range(args.first_road, args.first_road+args.roads))
    with_obs = evaluate_batched(policy, device, seeds, args.steps, obstacles=True)
    no_obs = evaluate_batched(policy, device, seeds, args.steps, obstacles=False)
    summary = {'checkpoint': args.checkpoint, 'graph_mode': config['graph_mode'],
               'task_version_at_eval': getattr(E, 'TASK_VERSION', 1),
               'task_version_at_train': config.get('task_version', 1),
               'with_obstacles': {k: v for k, v in with_obs.items() if k != 'per_episode'},
               'no_obstacles': {k: v for k, v in no_obs.items() if k != 'per_episode'},
               'per_episode_with_obstacles': with_obs['per_episode']}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_episode_with_obstacles'}, indent=2))
    print('failures:', [(r['seed'], r['failure'], round(r['distance_m'])) for r in with_obs['per_episode'] if r['failure']])


if __name__ == '__main__':
    main()
