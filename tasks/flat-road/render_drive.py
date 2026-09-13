#!/usr/bin/env python3
"""Render a closed-loop drive from a trained vision checkpoint as a video.

Three panels from one recorded clock: the road from above with the car's
trajectory and the obstacles, the 48x24 camera image the connectome actually
receives, and the model's signed rate states on measured soma locations.
"""
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
import pyarrow.feather as feather
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(os.environ.get('FLYHARD_ROOT', Path(__file__).resolve().parents[3] / 'flyhard')) / 'src'))  # sibling flyhard checkout, or set FLYHARD_ROOT
import drive_env as E
from train_vision import VisionPolicy

W, H = 1440, 720
DARK, MUTED, PALE, GREEN, RED, ROAD, VERGE = ((15, 20, 22), (154, 168, 165), (237, 241, 228),
                                             (173, 220, 119), (235, 96, 80), (70, 76, 72), (35, 44, 36))
FONT_PATH = next((p for p in ['/System/Library/Fonts/Supplemental/Arial.ttf',
                              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()), None)
font = lambda s: ImageFont.truetype(FONT_PATH, s) if FONT_PATH else ImageFont.load_default(size=s)


def load(checkpoint_path, graph_path, device):
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    graph = dict(np.load(Path(graph_path)/'graph.npz'))
    if ck['config']['graph_mode'] == 'shuffled':
        graph['col'] = np.random.default_rng(20260910).permutation(len(graph['crow'])-1)[graph['col']]
    state = ck['model']
    nodes = feather.read_table(Path(graph_path)/'nodes.feather')
    classes = np.asarray(nodes['superclass'].fill_null('').to_pylist())
    policy = VisionPolicy(graph, np.flatnonzero(classes == 'ol_sensory'),
                          np.flatnonzero(classes == 'vnc_motor'), ck['config']['seed'])
    policy.load_state_dict(state); policy.to(device).eval()
    soma = np.array([v if v is not None else [np.nan]*3 for v in nodes['somaLocation'].to_pylist()])
    valid = np.flatnonzero(np.isfinite(soma).all(axis=1))
    subset = np.sort(np.random.default_rng(12).choice(valid, min(12000, len(valid)), replace=False))
    return policy, ck['config'], subset, soma[subset]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--graph', default=os.environ.get('FLYHARD_ROOT', '../../../flyhard') + '/data/graph-traced-v1')
    p.add_argument('--seed', type=int, default=9000)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--out', required=True)
    p.add_argument('--fps', type=int, default=20)
    args = p.parse_args()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    policy, config, soma_idx, soma_xyz = load(args.checkpoint, args.graph, device)

    road = E.Road(args.seed); car = E.Car(road, np.random.default_rng(args.seed+7919))
    frames, states, path, images, commands, failure = [], [], [], [], [], None
    for _ in range(args.steps):
        image = E.render(car)
        with torch.no_grad():
            neural = policy.neural_state(torch.as_tensor(image[None], device=device))
            command = float(E.MAX_STEER*torch.tanh(neural[policy.motor_ids].T @ policy.decoder.T)[0, 0])
        states.append(neural[soma_idx, 0].cpu().numpy()); images.append(image); commands.append(command)
        path.append((car.x, car.y, car.theta, car.lateral))
        car.step(command)
        failure = car.crashed()
        if failure:
            path.append((car.x, car.y, car.theta, car.lateral)); break
    states = np.array(states)
    nonzero = np.abs(states[np.abs(states) > 1e-8])
    scale = float(np.quantile(nonzero, 0.95)) if len(nonzero) else 1.0
    normalized = np.clip(np.arcsinh(states/max(scale*0.05, 1e-8))/np.arcsinh(20), -1, 1)

    # Soma projection, brain left and cord right, fitted to the right-hand panel.
    pos = soma_xyz[:, [0, 2]].astype(float); pos -= pos.min(axis=0)
    pos *= min(385/np.ptp(pos[:, 0]), 435/np.ptp(pos[:, 1]))
    pos += np.array([1000 + (385-np.ptp(pos[:, 0]))/2, 165]); pos = pos.astype(int)

    xs = np.linspace(0, road.length, 2400)
    def world_to_panel(x, y, cx):
        # Camera follows the car; 1 m = 6 px; road runs left to right.
        return 40 + 6.0*(x - cx + 45), 300 - 6.0*(y - road.centre(cx))

    writer = imageio.get_writer(args.out, fps=args.fps, codec='libx264', quality=8, macro_block_size=1)
    for i, (x, y, theta, lateral) in enumerate(path[:len(states)]):
        canvas = Image.new('RGB', (W, H), DARK); d = ImageDraw.Draw(canvas)
        d.text((27, 22), 'FLYHARD', fill=GREEN, font=font(34))
        d.text((225, 30), 'Pixels in, steering out', fill=PALE, font=font(22))
        d.text((27, 70), f"165,122 neurons  /  25.56 million measured connections  /  {config['graph_mode']} graph  /  one Apple M4",
               fill=MUTED, font=font(17))
        # Top-down road panel.
        d.rectangle([27, 110, 935, 470], fill=(22, 28, 26))
        cx = x
        left = [world_to_panel(xx, road.centre(xx) + E.LANE_HALF_WIDTH/np.cos(road.heading(xx)), cx) for xx in xs if abs(xx-cx) < 80]
        right = [world_to_panel(xx, road.centre(xx) - E.LANE_HALF_WIDTH/np.cos(road.heading(xx)), cx) for xx in xs if abs(xx-cx) < 80]
        if len(left) > 2:
            d.polygon(left + right[::-1], fill=ROAD)
            d.line(left, fill=VERGE, width=2); d.line(right, fill=VERGE, width=2)
        for ox, oy in road.obstacles:
            if abs(ox-cx) < 80:
                px, py = world_to_panel(ox, oy, cx); r = 6.0*E.OBSTACLE_RADIUS
                d.ellipse([px-r, py-r, px+r, py+r], fill=RED)
        trail = [world_to_panel(px_, py_, cx) for px_, py_, _, _ in path[max(0, i-120):i+1]]
        if len(trail) > 1:
            d.line(trail, fill=GREEN, width=3)
        px, py = world_to_panel(x, y, cx)
        nose = (px + 14*np.cos(theta), py - 14*np.sin(theta))
        d.ellipse([px-6, py-6, px+6, py+6], fill=PALE); d.line([(px, py), nose], fill=PALE, width=3)
        d.text((40, 120), f'HELD-OUT ROAD {args.seed}     t = {i*E.DT:.2f} s     {x:.0f} m', fill=PALE, font=font(19))
        # Camera panel: what the brain sees.
        cam = Image.fromarray((images[i]*255).astype(np.uint8)).resize((432, 216), Image.NEAREST)
        canvas.paste(cam, (27, 490)); d.rectangle([27, 490, 459, 706], outline=VERGE, width=2)
        d.text((40, 496), 'CAMERA  48 x 24  ->  4,114 optic-lobe sensory neurons', fill=PALE, font=font(15))
        # Readouts.
        d.text((490, 500), f'Steering command  {np.degrees(commands[i]):+6.1f} deg', fill=GREEN, font=font(22))
        d.text((490, 540), f'Lateral offset    {lateral:+6.2f} m   (lane edge at 3.50)', fill=PALE, font=font(19))
        d.text((490, 572), f'Speed             {E.SPEED:.0f} m/s, fixed', fill=MUTED, font=font(17))
        bar_x, bar_y = 490, 620
        d.rectangle([bar_x, bar_y, bar_x+420, bar_y+12], outline=MUTED)
        d.rectangle([bar_x+210, bar_y-4, bar_x+211, bar_y+16], fill=MUTED)
        k = bar_x + 210 + 210*(commands[i]/E.MAX_STEER)
        d.ellipse([k-8, bar_y-2, k+8, bar_y+14], fill=GREEN)
        d.text((490, 645), 'full left                                              full right', fill=MUTED, font=font(13))
        if failure and i == len(states)-1:
            d.text((490, 675), {'off_road': 'LEFT THE ROAD', 'obstacle': 'HIT AN OBSTACLE'}[failure], fill=RED, font=font(24))
        # Neural panel.
        d.text((1000, 120), 'THE CONNECTOME MODEL', fill=GREEN, font=font(19))
        d.text((1000, 148), 'Measured soma locations + recorded rate states', fill=MUTED, font=font(14))
        for (sx, sy), v in zip(pos, normalized[i]):
            base = np.array([54, 66, 65])
            colour = base + (np.array([235, 150, 60]) - base)*v if v > 0 else base + (np.array([80, 160, 235]) - base)*(-v)
            d.point((sx, sy), fill=tuple(int(c) for c in colour))
        d.text((1000, 620), '12,000 fixed sampled somata; signed asinh colour', fill=MUTED, font=font(14))
        d.text((1000, 645), f'Neural decision at {i*E.DT:.2f} s   (4 graph updates per decision)', fill=PALE, font=font(16))
        d.text((27, 707 - 6), '', fill=MUTED)
        frames.append(np.asarray(canvas)); writer.append_data(np.asarray(canvas))
    writer.close()
    Image.fromarray(frames[min(len(frames)-1, 200)]).save(Path(args.out).with_suffix('.png'))
    print(json.dumps({'seed': args.seed, 'steps': len(states), 'failure': failure,
                      'distance_m': float(path[-1][0]), 'mean_abs_lateral_m': float(np.mean([abs(p_[3]) for p_ in path])),
                      'frames': len(frames), 'out': args.out}, indent=2))


if __name__ == '__main__':
    main()
