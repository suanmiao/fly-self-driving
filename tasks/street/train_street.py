#!/usr/bin/env python3
"""The street-driving task: stateful connectome policy plus dense baselines.

Connectome recipe as before - measured adjacency frozen, per-edge gains and
per-neuron leaks train, frozen random interfaces - with the state carried
across the 50 ms decisions (truncated BPTT over windows). Pixels of the 64x32
street render enter the optic-lobe sensory neurons; steering is read from the
VNC motor neurons. --model linear|mlp train small dense baselines on the same
demonstrations (memoryless, as before); --graph-mode shuffled is the control.
"""
import os
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import pyarrow.feather as feather
sys.path.insert(0, str(Path(os.environ.get('FLYHARD_ROOT', Path(__file__).resolve().parents[3] / 'flyhard')) / 'src'))  # sibling flyhard checkout, or set FLYHARD_ROOT
from flyhard.connectome import SparseConnectome
import street_env as E

PIXELS = E.IMAGE_W*E.IMAGE_H


class StreetPolicy(nn.Module):
    def __init__(self, graph, sensory_ids, motor_ids, seed=123):
        super().__init__()
        self.core = SparseConnectome(graph['crow'], graph['col'], graph['counts'])
        rng = np.random.default_rng(seed)
        self.register_buffer('sensory_ids', torch.as_tensor(sensory_ids, dtype=torch.long))
        self.register_buffer('motor_ids', torch.as_tensor(motor_ids, dtype=torch.long))
        self.register_buffer('feature_ids', torch.as_tensor(rng.integers(0, PIXELS, len(sensory_ids))))
        self.register_buffer('input_signs', torch.as_tensor(rng.choice([-1., 1.], len(sensory_ids)), dtype=torch.float32))
        self.register_buffer('decoder', torch.as_tensor(rng.normal(size=(1, len(motor_ids)))/np.sqrt(len(motor_ids)), dtype=torch.float32))

    def drive_for(self, images):
        features = ((images.flatten(1) - 0.5)*2.0).clamp(-2, 2)
        drive = torch.zeros(self.core.n, len(images), device=images.device)
        drive[self.sensory_ids] = features[:, self.feature_ids].T*self.input_signs[:, None]
        return drive

    def step_state(self, state, images, updates=4):
        return self.core(state, steps=updates, drive=self.drive_for(images))

    def readout(self, state):
        return E.MAX_STEER*torch.tanh(state[self.motor_ids].T @ self.decoder.T)[:, 0]

    def run_window(self, images_seq, state):
        outs = []
        for k in range(images_seq.shape[0]):
            state = self.step_state(state, images_seq[k]); outs.append(self.readout(state))
        return torch.stack(outs), state

    def calibrate(self, images):
        with torch.no_grad():
            state = self.step_state(torch.zeros(self.core.n, len(images), device=images.device), images)
            raw = state[self.motor_ids].T @ self.decoder.T
            self.decoder.mul_(0.5/raw.std().clamp_min(1e-5))


class DensePolicy(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(PIXELS, hidden), nn.Tanh(), nn.Linear(hidden, 1)) if hidden else nn.Linear(PIXELS, 1)

    def forward(self, images):
        return E.MAX_STEER*torch.tanh(self.net((images.flatten(1) - 0.5)*2.0))[:, 0]


def demonstrations(seeds, noise, steps, after=40):
    """Expert roll-outs with steering noise. `phase` marks the overtake: the frames where the
    expert holds the left lane plus the `after` frames that follow (the pull-in, driven blind)."""
    rng = np.random.default_rng(4242)
    images, actions, starts, phase = [], [], [], []
    for seed in seeds:
        road = E.Road(seed); car = E.Car(road, np.random.default_rng(seed+7919)); starts.append(len(images)); left = -10**9
        for k in range(steps):
            a = E.expert(car); images.append(E.render(car)); actions.append(a)
            if E.overtaking(car):
                left = k
            phase.append(k - left <= after)
            car.step(a + rng.normal(0, noise))
            if car.crashed():
                break
    starts.append(len(images))
    return np.array(images, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(starts), np.array(phase)


def collect_on_policy(policy, device, seeds, steps, rng_offset, after=40):
    """DAgger: drive the training roads with the current policy (state carried) and label every
    frame the policy actually reaches with the expert's action. Returns the same arrays as
    demonstrations(), so the closed-loop states the clone drifts into join the training set."""
    policy.eval()
    roads = [E.Road(s) for s in seeds]
    cars = [E.Car(r, np.random.default_rng(s+7919+rng_offset)) for r, s in zip(roads, seeds)]
    frames = [[] for _ in seeds]; labels = [[] for _ in seeds]; flags = [[] for _ in seeds]
    state = torch.zeros(policy.core.n, len(seeds), device=device); active = list(range(len(seeds)))
    for _ in range(steps):
        if not active:
            break
        imgs = [E.render(cars[i]) for i in active]
        with torch.no_grad():
            sub = policy.step_state(state[:, active], torch.as_tensor(np.stack(imgs), device=device)); state[:, active] = sub
            commands = policy.readout(sub).cpu().numpy()
        still = []
        for i, img, cmd in zip(active, imgs, commands):
            frames[i].append(img); labels[i].append(E.expert(cars[i])); flags[i].append(E.overtaking(cars[i]))
            cars[i].step(float(cmd))
            if not cars[i].crashed():
                still.append(i)
        active = still
    policy.train()
    images, actions, starts, phase = [], [], [], []
    for f, l, g in zip(frames, labels, flags):
        starts.append(len(images)); images += f; actions += l
        left = -10**9
        for k, on in enumerate(g):
            if on:
                left = k
            phase.append(k - left <= after)
    starts.append(len(images))
    return np.array(images, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(starts), np.array(phase)


def window_sampling(starts, phase, window, weight):
    """Window start indices that stay inside one episode, and their sampling weights."""
    valid_starts = np.concatenate([np.arange(starts[e], starts[e+1]-window+1) for e in range(len(starts)-1) if starts[e+1]-starts[e] >= window])
    touches = np.array([phase[s0:s0+window].any() for s0 in valid_starts])
    window_p = np.where(touches, weight, 1.0); window_p /= window_p.sum()
    return valid_starts, window_p, touches


def evaluate(policy, device, seeds, steps, obstacles=True, stateful=True):
    policy.eval()
    roads = [E.Road(s, obstacles=obstacles) for s in seeds]
    cars = [E.Car(r, np.random.default_rng(s+7919)) for r, s in zip(roads, seeds)]
    lateral = [[] for _ in seeds]; failure = [None]*len(seeds); active = list(range(len(seeds)))
    state = torch.zeros(policy.core.n, len(seeds), device=device) if stateful else None
    for _ in range(steps):
        if not active:
            break
        images = torch.as_tensor(np.stack([E.render(cars[i]) for i in active]), device=device)
        with torch.no_grad():
            if stateful:
                sub = policy.step_state(state[:, active], images); state[:, active] = sub
                commands = policy.readout(sub).cpu().numpy()
            else:
                commands = policy(images).cpu().numpy()
        still = []
        for i, cmd in zip(active, commands):
            cars[i].step(float(cmd)); lateral[i].append(abs(cars[i].lateral - E.LANE_CENTRE))
            failure[i] = cars[i].crashed()
            if not failure[i]:
                still.append(i)
        active = still
    policy.train()
    results = [{'seed': s, 'steps': len(lateral[i]), 'failure': failure[i], 'mean_abs_lateral': float(np.mean(lateral[i])),
                'max_abs_lateral': float(np.max(lateral[i])), 'distance_m': float(cars[i].x)} for i, s in enumerate(seeds)]
    return {'episodes': len(results), 'completed': sum(r['failure'] is None for r in results),
            **{kind: sum(r['failure'] == kind for r in results) for kind in E.FAILURES},
            'mean_abs_lateral_m': float(np.mean([r['mean_abs_lateral'] for r in results])),
            'mean_distance_m': float(np.mean([r['distance_m'] for r in results])), 'per_episode': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', default=os.environ.get('FLYHARD_ROOT', '../../../flyhard') + '/data/graph-traced-v1')
    p.add_argument('--out', required=True)
    p.add_argument('--model', choices=['connectome', 'linear', 'mlp'], default='connectome')
    p.add_argument('--hidden', type=int, default=40)
    p.add_argument('--graph-mode', choices=['measured', 'shuffled'], default='measured')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.04)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--demo-roads', type=int, default=60)
    p.add_argument('--demo-steps', type=int, default=300)
    p.add_argument('--eval-roads', type=int, default=20)
    p.add_argument('--eval-steps', type=int, default=500)
    p.add_argument('--noise', type=float, default=0.06)
    p.add_argument('--device', default=None)
    p.add_argument('--overtake-weight', type=float, default=4.0, help='sampling weight of windows touching an overtake')
    p.add_argument('--init', default=None, help='warm-start from this checkpoint (same model/graph mode)')
    p.add_argument('--dagger-rounds', type=int, default=0, help='DAgger rounds after the initial training')
    p.add_argument('--dagger-steps', type=int, default=600, help='training steps per DAgger round')
    p.add_argument('--dagger-roads', type=int, default=30, help='training roads driven per DAgger round')
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); torch.manual_seed(args.seed); torch.set_num_threads(4)
    device = args.device or ('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    stateful = args.model == 'connectome'

    train_roads = list(range(1000, 1000+args.demo_roads)); eval_roads = list(range(9000, 9000+args.eval_roads))
    images, actions, starts, phase = demonstrations(train_roads, args.noise, args.demo_steps)
    print(json.dumps({'stage': 'demonstrations', 'pairs': len(images), 'overtake_frames': int(phase.sum())}), flush=True)
    x = torch.as_tensor(images, device=device); y = torch.as_tensor(actions, device=device)
    config = {**vars(args), 'device': device, 'task_version': E.TASK_VERSION, 'stateful': stateful,
              'image': [E.IMAGE_H, E.IMAGE_W], 'train_roads': [train_roads[0], train_roads[-1]], 'eval_roads': [eval_roads[0], eval_roads[-1]],
              'teacher': 'Lane-keeping pure pursuit with in-lane swerves around parked cars and left-lane overtakes of slow cars; noise-injected demonstrations.',
              'claim': 'Closed-loop driving in a ray-cast street scene with parked, oncoming and slow same-direction cars; not CARLA.'}
    if stateful:
        graph = dict(np.load(Path(args.graph)/'graph.npz')); manifest = json.loads((Path(args.graph)/'manifest.json').read_text())
        if args.graph_mode == 'shuffled':
            graph['col'] = np.random.default_rng(20260910).permutation(len(graph['crow'])-1)[graph['col']]
        nodes = feather.read_table(Path(args.graph)/'nodes.feather'); classes = np.asarray(nodes['superclass'].fill_null('').to_pylist())
        policy = StreetPolicy(graph, np.flatnonzero(classes == 'ol_sensory'), np.flatnonzero(classes == 'vnc_motor'), args.seed).to(device)
        policy.calibrate(x[::max(len(x)//64, 1)])
        config.update({'graph_sha256': manifest['graph_sha256'], 'neural_steps_per_decision': 4, 'state_reset': 'episode start only',
                       'training': f'truncated BPTT over windows of {args.window} decisions, batch {args.batch} windows',
                       'interfaces': 'Frozen random signed pixel->ol_sensory map and frozen random vnc_motor->steering readout; only edge gains and leaks train.'})
        valid_starts, window_p, touches = window_sampling(starts, phase, args.window, args.overtake_weight)
        config['overtake_windows'] = int(touches.sum()); config['overtake_sampling_share'] = float(window_p[touches].sum())
    else:
        policy = DensePolicy(args.hidden if args.model == 'mlp' else 0).to(device)
    if args.init:
        policy.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=False)['model']); config['init'] = args.init
    config['trainable_parameters'] = sum(t.numel() for t in policy.parameters())
    (out/'config.json').write_text(json.dumps(config, indent=2))

    baseline = evaluate(policy, device, eval_roads[:10], args.eval_steps, stateful=stateful)
    print(json.dumps({'stage': 'untrained', **{k: v for k, v in baseline.items() if k != 'per_episode'}}), flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    history = []; train_start = time.perf_counter(); rng = np.random.default_rng(args.seed)

    def train_steps(n, tag):
        nonlocal x, y, valid_starts, window_p
        for _ in range(n):
            step = len(history)
            opt.zero_grad(set_to_none=True)
            if stateful:
                s0 = rng.choice(valid_starts, args.batch, p=window_p)
                idx = torch.as_tensor(s0[None, :] + np.arange(args.window)[:, None], device=device)
                state = torch.zeros(policy.core.n, args.batch, device=device)
                with torch.no_grad():
                    for k in range(min(2, args.window)):
                        state = policy.step_state(state, x[idx[k]])
                pred, _ = policy.run_window(x[idx], state)
                loss = ((pred - y[idx])/E.MAX_STEER).square().mean()
            else:
                idx = torch.randint(0, len(x), (16,), device=device)
                loss = ((policy(x[idx]) - y[idx])/E.MAX_STEER).square().mean()
            assert torch.isfinite(loss); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            history.append({'step': step+1, 'loss': float(loss.detach()), 'elapsed_seconds': time.perf_counter()-train_start, 'stage': tag})
            if step % 25 == 0:
                print(json.dumps(history[-1]), flush=True); (out/'history.json').write_text(json.dumps(history, indent=2))

    train_steps(args.steps, 'clone')
    dagger_log = []
    for r in range(args.dagger_rounds if stateful else 0):
        roads = [int(v) for v in rng.choice(train_roads, args.dagger_roads, replace=False)]
        xi, yi, si, pi = collect_on_policy(policy, device, roads, args.demo_steps, rng_offset=1000*(r+1))
        offset = len(images); images = np.concatenate([images, xi]); actions = np.concatenate([actions, yi])
        starts = np.concatenate([starts[:-1], si + offset]); phase = np.concatenate([phase, pi])
        x = torch.as_tensor(images, device=device); y = torch.as_tensor(actions, device=device)
        valid_starts, window_p, touches = window_sampling(starts, phase, args.window, args.overtake_weight)
        dagger_log.append({'round': r+1, 'new_pairs': int(len(xi)), 'total_pairs': int(len(images)), 'roads_finished': int(sum(np.diff(si) >= args.demo_steps))})
        print(json.dumps({'stage': 'dagger', **dagger_log[-1]}), flush=True)
        train_steps(args.dagger_steps, f'dagger{r+1}')
    config['dagger'] = dagger_log
    training_seconds = time.perf_counter()-train_start
    torch.save({'model': policy.state_dict(), 'config': config, 'steps': len(history)}, out/'checkpoint.pt')

    trained = evaluate(policy, device, eval_roads, args.eval_steps, stateful=stateful)
    no_obstacles = evaluate(policy, device, eval_roads, args.eval_steps, obstacles=False, stateful=stateful)
    expert_ref = [E.rollout(s, lambda img, car: E.expert(car), steps=args.eval_steps) for s in eval_roads]
    metrics = {**config, 'untrained': {k: v for k, v in baseline.items() if k != 'per_episode'},
               'trained': {k: v for k, v in trained.items() if k != 'per_episode'},
               'trained_no_obstacles': {k: v for k, v in no_obstacles.items() if k != 'per_episode'},
               'expert_reference': {'completed': sum(r['failure'] is None for r in expert_ref), 'episodes': len(expert_ref)},
               'first_loss': history[0]['loss'] if history else None, 'last_loss': history[-1]['loss'] if history else None,
               'training_seconds': training_seconds, 'wall_seconds': time.perf_counter()-start}
    (out/'history.json').write_text(json.dumps(history, indent=2))
    (out/'trained-episodes.json').write_text(json.dumps(trained['per_episode'], indent=2))
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ['model', 'graph_mode', 'trainable_parameters', 'untrained', 'trained', 'trained_no_obstacles', 'expert_reference', 'first_loss', 'last_loss', 'training_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
