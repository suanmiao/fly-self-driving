#!/usr/bin/env python3
"""Can the measured fly connectome learn pixels-in, steering-out?

Same discipline as the flyhard wheel pilot: the measured adjacency is frozen,
only the per-edge gains and per-neuron leaks train, and both the sensory and
motor interfaces are frozen random projections so nothing bypasses the graph.
What changes is the input - a rendered camera image onto the optic-lobe
sensory population - and the task, which is closed-loop lane keeping with
obstacles rather than holding a requested wheel angle.

--graph-mode shuffled is the control: the column indices are relabelled by one
random permutation, which preserves every neuron's in-degree exactly and the
out-degree sequence as a whole, and destroys only which neuron talks to which.
"""
import os
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import pyarrow.feather as feather
import sys
sys.path.insert(0, str(Path(os.environ.get('FLYHARD_ROOT', Path(__file__).resolve().parents[3] / 'flyhard')) / 'src'))  # sibling flyhard checkout, or set FLYHARD_ROOT
from flyhard.connectome import SparseConnectome
import drive_env as E


class VisionPolicy(nn.Module):
    def __init__(self, graph, sensory_ids, motor_ids, seed=123):
        super().__init__()
        self.core = SparseConnectome(graph['crow'], graph['col'], graph['counts'])
        rng = np.random.default_rng(seed)
        pixels = E.IMAGE_W*E.IMAGE_H
        self.register_buffer('sensory_ids', torch.as_tensor(sensory_ids, dtype=torch.long))
        self.register_buffer('motor_ids', torch.as_tensor(motor_ids, dtype=torch.long))
        self.register_buffer('feature_ids', torch.as_tensor(rng.integers(0, pixels, len(sensory_ids))))
        self.register_buffer('input_signs', torch.as_tensor(rng.choice([-1., 1.], len(sensory_ids)), dtype=torch.float32))
        self.register_buffer('decoder', torch.as_tensor(rng.normal(size=(1, len(motor_ids)))/np.sqrt(len(motor_ids)), dtype=torch.float32))

    def neural_state(self, images):
        features = ((images.flatten(1) - 0.5)*2.0).clamp(-2, 2)
        drive = torch.zeros(self.core.n, len(images), device=images.device)
        drive[self.sensory_ids] = features[:, self.feature_ids].T*self.input_signs[:, None]
        return self.core(torch.zeros_like(drive), steps=4, drive=drive)

    def forward(self, images):
        state = self.neural_state(images)
        return E.MAX_STEER*torch.tanh(state[self.motor_ids].T @ self.decoder.T)[:, 0]

    def calibrate(self, images):
        with torch.no_grad():
            raw = self.neural_state(images)[self.motor_ids].T @ self.decoder.T
            self.decoder.mul_(0.5/raw.std().clamp_min(1e-5))


def demonstrations(seeds, noise, steps):
    """Noise-injected expert rollouts: the car is pushed off-centre, the label
    stays the expert's action at the state actually visited."""
    rng = np.random.default_rng(4242)
    images, actions = [], []
    for seed in seeds:
        road = E.Road(seed)
        car = E.Car(road, np.random.default_rng(seed+7919))
        for _ in range(steps):
            action = E.expert(car)
            images.append(E.render(car)); actions.append(action)
            car.step(action + rng.normal(0, noise))
            if car.crashed():
                break
    return np.array(images, dtype=np.float32), np.array(actions, dtype=np.float32)


def evaluate(policy, device, seeds, steps, obstacles=True):
    policy.eval()
    def act(image, car):
        with torch.no_grad():
            return float(policy(torch.as_tensor(image[None], device=device))[0].cpu())
    results = [E.rollout(s, act, steps=steps, obstacles=obstacles) for s in seeds]
    policy.train()
    completed = sum(r['failure'] is None for r in results)
    return {'episodes': len(results), 'completed': completed,
            'off_road': sum(r['failure'] == 'off_road' for r in results),
            'obstacle_hits': sum(r['failure'] == 'obstacle' for r in results),
            'mean_abs_lateral_m': float(np.mean([r['mean_abs_lateral'] for r in results])),
            'mean_distance_m': float(np.mean([r['distance_m'] for r in results])),
            'per_episode': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph', default=os.environ.get('FLYHARD_ROOT', '../../../flyhard') + '/data/graph-traced-v1')
    p.add_argument('--out', required=True)
    p.add_argument('--graph-mode', choices=['measured', 'shuffled'], default='measured')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.04)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--demo-roads', type=int, default=60)
    p.add_argument('--demo-steps', type=int, default=260)
    p.add_argument('--eval-roads', type=int, default=20)
    p.add_argument('--eval-steps', type=int, default=260)
    p.add_argument('--noise', type=float, default=0.06)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    graph = dict(np.load(Path(args.graph)/'graph.npz'))
    manifest = json.loads((Path(args.graph)/'manifest.json').read_text())
    if args.graph_mode == 'shuffled':
        permutation = np.random.default_rng(20260910).permutation(len(graph['crow'])-1)
        graph['col'] = permutation[graph['col']]
    nodes = feather.read_table(Path(args.graph)/'nodes.feather')
    classes = np.asarray(nodes['superclass'].fill_null('').to_pylist())
    sensory = np.flatnonzero(classes == 'ol_sensory')
    motor = np.flatnonzero(classes == 'vnc_motor')

    train_roads = list(range(1000, 1000+args.demo_roads))
    eval_roads = list(range(9000, 9000+args.eval_roads))
    config = {**vars(args), 'device': device, 'graph_sha256': manifest['graph_sha256'], 'task_version': getattr(E, 'TASK_VERSION', 1),
              'sensory_population': 'ol_sensory', 'sensory_neurons': len(sensory),
              'motor_population': 'vnc_motor', 'motor_neurons': len(motor),
              'image': [E.IMAGE_H, E.IMAGE_W], 'neural_steps_per_decision': 4,
              'interfaces': 'Frozen random signed pixel->ol_sensory map and frozen random vnc_motor->steering readout; only core edge gains and leaks train.',
              'teacher': 'Obstacle-aware pure pursuit, noise-injected demonstrations. Never used at evaluation.',
              'train_roads': [train_roads[0], train_roads[-1]], 'eval_roads': [eval_roads[0], eval_roads[-1]],
              'claim': 'Closed-loop visual lane keeping with static obstacles in a numpy driving task; not CARLA and not the published experiment.'}
    (out/'config.json').write_text(json.dumps(config, indent=2))

    images, actions = demonstrations(train_roads, args.noise, args.demo_steps)
    print(json.dumps({'stage': 'demonstrations', 'pairs': len(images)}), flush=True)
    x = torch.as_tensor(images, device=device); y = torch.as_tensor(actions, device=device)
    policy = VisionPolicy(graph, sensory, motor, args.seed).to(device)
    policy.calibrate(x[::max(len(x)//64, 1)])

    baseline = evaluate(policy, device, eval_roads[:10], args.eval_steps)
    print(json.dumps({'stage': 'untrained', **{k: v for k, v in baseline.items() if k != 'per_episode'}}), flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    history = []; train_start = time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(0, len(x), (args.batch,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = ((policy(x[idx]) - y[idx])/E.MAX_STEER).square().mean()
        assert torch.isfinite(loss); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        history.append({'step': step+1, 'loss': float(loss.detach()),
                        'elapsed_seconds': time.perf_counter()-train_start})
        if step % 25 == 0:
            print(json.dumps(history[-1]), flush=True)
            (out/'history.json').write_text(json.dumps(history, indent=2))
    torch.save({'model': policy.state_dict(), 'config': config, 'steps': len(history)}, out/'checkpoint.pt')

    trained = evaluate(policy, device, eval_roads, args.eval_steps)
    no_obstacles = evaluate(policy, device, eval_roads, args.eval_steps, obstacles=False)
    expert_ref = [E.rollout(s, lambda img, car: E.expert(car), steps=args.eval_steps) for s in eval_roads]
    metrics = {**config, 'untrained': {k: v for k, v in baseline.items() if k != 'per_episode'},
               'trained': {k: v for k, v in trained.items() if k != 'per_episode'},
               'trained_no_obstacles': {k: v for k, v in no_obstacles.items() if k != 'per_episode'},
               'expert_reference': {'completed': sum(r['failure'] is None for r in expert_ref),
                                    'episodes': len(expert_ref),
                                    'mean_abs_lateral_m': float(np.mean([r['mean_abs_lateral'] for r in expert_ref]))},
               'trainable_parameters': sum(t.numel() for t in policy.parameters()),
               'first_loss': history[0]['loss'], 'last_loss': history[-1]['loss'],
               'training_seconds': time.perf_counter()-train_start,
               'wall_seconds': time.perf_counter()-start}
    (out/'history.json').write_text(json.dumps(history, indent=2))
    (out/'trained-episodes.json').write_text(json.dumps(trained['per_episode'], indent=2))
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ('per_episode',)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
