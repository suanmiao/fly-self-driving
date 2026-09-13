#!/usr/bin/env python3
"""Independent held-out evaluation of a trained wheel checkpoint, with full angle traces.

Runs the same 100 held-out targets (seed 61733) and pass gate as train_wheel.py,
but also records the wheel angle at every 50 ms decision so that tracking
quality (achieved vs requested angle, settle time, motion range) can be judged
rather than only the pass count. Optionally includes the untrained
(core-reset) control and a constant-angle strawman for calibration.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
import torch
import flyhard.connectome_mps  # noqa: F401
from flyhard.cockpit import WheelRig
from flyhard.motor_policy import WheelPolicy


def observation(rig, target):
    return np.r_[target, rig.angle, rig.data.qpos[rig.active_qpos]].astype(np.float32)


def run_trials(policy, rig, targets, device):
    rows = []
    for target in targets:
        rig.reset(); angles = []; failure = None
        for _ in range(60):
            obs = torch.tensor(observation(rig, target)[None], device=device)
            with torch.no_grad():
                action = policy(obs)[0].cpu().numpy()
            try:
                for _ in range(10):
                    rig.step(action)
            except AssertionError:
                failure = 'physical numerical instability'; break
            angles.append(rig.angle)
        angles = np.asarray(angles)
        err = float(np.max(np.abs(angles[-20:] - target))) if len(angles) == 60 else None
        settle = next((i for i in range(60) if np.all(np.abs(angles[i:] - target) <= 0.13)), None) if len(angles) == 60 else None
        rows.append({'target': float(target), 'passed': failure is None and err <= 0.13,
                     'max_last_second_error_rad': err, 'final_angle': float(angles[-1]) if len(angles) else None,
                     'mean_last_second_angle': float(angles[-20:].mean()) if len(angles) == 60 else None,
                     'settle_decision': settle, 'failure': failure, 'angles': angles.tolist()})
    return rows


def summarize(rows, tag):
    t = np.array([r['target'] for r in rows]); a = np.array([r['mean_last_second_angle'] for r in rows], dtype=float)
    e = np.array([r['max_last_second_error_rad'] for r in rows], dtype=float)
    ok = np.isfinite(a)
    slope, intercept = np.polyfit(t[ok], a[ok], 1) if ok.sum() > 2 else (np.nan, np.nan)
    r = np.corrcoef(t[ok], a[ok])[0, 1] if ok.sum() > 2 else np.nan
    settle = [r_['settle_decision'] for r_ in rows if r_['settle_decision'] is not None]
    return {'tag': tag, 'passed': int(sum(r_['passed'] for r_ in rows)), 'trials': len(rows),
            'error_deg_mean': float(np.degrees(np.nanmean(e))), 'error_deg_median': float(np.degrees(np.nanmedian(e))),
            'error_deg_max': float(np.degrees(np.nanmax(e))),
            'achieved_vs_requested_slope': float(slope), 'achieved_vs_requested_intercept_deg': float(np.degrees(intercept)),
            'achieved_vs_requested_r': float(r),
            'settle_time_s_median': float(np.median(settle) * 0.05) if settle else None,
            'settle_time_s_max': float(np.max(settle) * 0.05) if settle else None,
            'wheel_angle_deg_min': float(np.degrees(min(min(r_['angles']) for r_ in rows))),
            'wheel_angle_deg_max': float(np.degrees(max(max(r_['angles']) for r_ in rows)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--graph', default='data/graph-traced-v1')
    p.add_argument('--out', required=True); p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--control', action='store_true', help='also evaluate the untrained core (edge_gain=leak=0) with the same interfaces')
    args = p.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); start = time.perf_counter()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False); state = ck['model']
    graph = np.load(Path(args.graph) / 'graph.npz')
    policy = WheelPolicy(graph, state['sensory_ids'].numpy(), state['motor_ids'].numpy(),
                         state['neutral'].numpy(), state['action_scale'].numpy(), ck['config']['seed'])
    policy.load_state_dict(state); policy.to(args.device).eval()
    manifest = json.loads((Path(args.graph) / 'manifest.json').read_text())
    rig = WheelRig()
    targets = np.random.default_rng(61733).uniform(0.18, 0.42, 100) * np.tile([1, -1], 50)
    report = {'checkpoint_sha256': hashlib.file_digest(open(args.checkpoint, 'rb'), 'sha256').hexdigest(),
              'graph_sha256': manifest['graph_sha256'], 'device': args.device, 'trainable_parameters': sum(t.numel() for t in policy.parameters()),
              'pass_gate': '|angle - target| <= 0.13 rad for every 50 ms sample of the last second (3 s trial)',
              'target_range_deg': [float(np.degrees(0.18)), float(np.degrees(0.42))]}
    trained = run_trials(policy, rig, targets, args.device)
    report['trained'] = summarize(trained, 'trained')
    json.dump(trained, open(out / 'trained-trials.json', 'w'))
    print(json.dumps(report['trained']), flush=True)
    if args.control:
        with torch.no_grad():
            policy.core.edge_gain.zero_(); policy.core.leak.zero_()
        control = run_trials(policy, rig, targets, args.device)
        report['untrained_control'] = summarize(control, 'untrained_control')
        json.dump(control, open(out / 'control-trials.json', 'w'))
        print(json.dumps(report['untrained_control']), flush=True)
    # Strawman: what the pass gate alone can and cannot distinguish.
    report['constant_sign_policy_would_pass'] = int(np.sum(np.abs(np.abs(targets) - 0.30) <= 0.13))
    report['wall_seconds'] = time.perf_counter() - start
    (out / 'report.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
