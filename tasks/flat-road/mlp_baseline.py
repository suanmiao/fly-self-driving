#!/usr/bin/env python3
"""MLP baselines on Jack Opus's visual driving task, same protocol as train_vision.py.

Reuses his `demonstrations` and `evaluate` verbatim (imported), so the roads,
noise injection, labels, and closed-loop scoring are identical by construction.
Only the policy differs: a feed-forward net from the same normalised pixels to
the same MAX_STEER*tanh(.) output head, in three sizes:

  linear   : pixels -> 1                 (~1.2k params)
  small    : pixels -> 40 -> 1           (~46k params)
  matched  : pixels -> h -> h -> 1, h chosen so params == connectome's 25,728,319

Training: Adam, batch 16, 1,200 updates, MSE on steer/MAX_STEER, grad clip 1.0,
exactly as his loop. `--lr` is swept because 0.04 is a connectome-gain learning
rate and would not be a fair setting for dense weights; both are reported.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import drive_env as E
from train_vision import demonstrations, evaluate

CONNECTOME_TRAINABLE = 25_728_319
PIXELS = E.IMAGE_W * E.IMAGE_H


class MLPPolicy(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        mods, d = [], PIXELS
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, images):
        features = ((images.flatten(1) - 0.5) * 2.0).clamp(-2, 2)   # same normalisation as VisionPolicy
        return E.MAX_STEER * torch.tanh(self.net(features))[:, 0]


def matched_hidden(target=CONNECTOME_TRAINABLE):
    # params(h) = PIXELS*h + h + h*h + h + h + 1
    b, c = PIXELS + 3, 1 - target
    return int(round((-b + (b * b - 4 * c) ** 0.5) / 2))


TIERS = {'linear': (0, 0), 'small': (40, 1), 'matched': (matched_hidden(), 2)}


def run(tier, lr, seed, args, x, y, device, eval_roads):
    torch.manual_seed(seed)
    policy = MLPPolicy(*TIERS[tier]).to(device)
    n_params = sum(t.numel() for t in policy.parameters())
    untrained = evaluate(policy, device, eval_roads[:10], args.eval_steps)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history, t0 = [], time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(0, len(x), (args.batch,), device=device)
        opt.zero_grad(set_to_none=True)
        loss = ((policy(x[idx]) - y[idx]) / E.MAX_STEER).square().mean()
        assert torch.isfinite(loss), f'{tier} lr={lr} diverged at step {step}'
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        history.append(float(loss.detach()))
    train_s = time.perf_counter() - t0
    trained = evaluate(policy, device, eval_roads, args.eval_steps)
    no_obs = evaluate(policy, device, eval_roads, args.eval_steps, obstacles=False)
    strip = lambda r: {k: v for k, v in r.items() if k != 'per_episode'}
    return {'tier': tier, 'lr': lr, 'seed': seed, 'trainable_parameters': n_params,
            'first_loss': history[0], 'last_loss': float(np.mean(history[-50:])),
            'training_seconds': train_s, 'untrained': strip(untrained),
            'trained': strip(trained), 'trained_no_obstacles': strip(no_obs),
            'trained_per_episode': trained['per_episode']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='runs/mlp')
    p.add_argument('--tiers', default='linear,small,matched')
    p.add_argument('--lrs', default='0.04,0.001')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--demo-roads', type=int, default=60)
    p.add_argument('--demo-steps', type=int, default=300)
    p.add_argument('--eval-roads', type=int, default=20)
    p.add_argument('--eval-steps', type=int, default=500)
    p.add_argument('--noise', type=float, default=0.06)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)

    train_roads = list(range(1000, 1000 + args.demo_roads))
    eval_roads = list(range(9000, 9000 + args.eval_roads))
    t0 = time.perf_counter()
    images, actions = demonstrations(train_roads, args.noise, args.demo_steps)
    print(json.dumps({'stage': 'demonstrations', 'pairs': len(images),
                      'seconds': time.perf_counter() - t0}), flush=True)
    x = torch.as_tensor(images, device=device); y = torch.as_tensor(actions, device=device)

    # References: the expert and a straight-ahead policy, same roads and step budget.
    expert_ref = [E.rollout(s, lambda img, car: E.expert(car), steps=args.eval_steps) for s in eval_roads]
    straight = [E.rollout(s, lambda img, car: 0.0, steps=args.eval_steps) for s in eval_roads]
    ref = lambda rs: {'completed': sum(r['failure'] is None for r in rs), 'episodes': len(rs),
                      'mean_distance_m': float(np.mean([r['distance_m'] for r in rs])),
                      'mean_abs_lateral_m': float(np.mean([r['mean_abs_lateral'] for r in rs]))}
    results = {'config': {**vars(args), 'device': device, 'task_version': getattr(E, 'TASK_VERSION', 1), 'tiers': {k: {'hidden': v[0], 'layers': v[1]} for k, v in TIERS.items()},
                          'train_roads': [train_roads[0], train_roads[-1]], 'eval_roads': [eval_roads[0], eval_roads[-1]],
                          'protocol': 'demonstrations() and evaluate() imported from train_vision.py unchanged'},
               'expert_reference': ref(expert_ref), 'straight_reference': ref(straight), 'runs': []}
    print(json.dumps({'expert': results['expert_reference'], 'straight': results['straight_reference']}), flush=True)

    for tier in args.tiers.split(','):
        for lr in [float(v) for v in args.lrs.split(',')]:
            for seed in [int(v) for v in args.seeds.split(',')]:
                try:
                    r = run(tier, lr, seed, args, x, y, device, eval_roads)
                except AssertionError as err:
                    r = {'tier': tier, 'lr': lr, 'seed': seed, 'diverged': str(err)}
                results['runs'].append(r)
                print(json.dumps({k: v for k, v in r.items() if k != 'trained_per_episode'}), flush=True)
                (out / 'results.json').write_text(json.dumps(results, indent=2))
    results['wall_seconds'] = time.perf_counter() - t0
    (out / 'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
