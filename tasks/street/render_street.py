#!/usr/bin/env python3
"""Cinematic render of a street drive by a trained checkpoint.

Left: a chase camera behind the driver's car at 640x320, coloured by material,
with the 64x32 driver's-eye grayscale the brain actually sees shown inset. Right: all 140,024 neurons
with measured soma positions, each coloured on its own activity range over the
episode (flyhard's convention), so the central brain and cord light up as the
signal moves through them rather than being drowned out by the eyes. Every
panel shares one recorded clock."""
import os
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
import pyarrow.feather as feather
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, str(Path(os.environ.get('FLYHARD_ROOT', Path(__file__).resolve().parents[3] / 'flyhard')) / 'src'))  # sibling flyhard checkout, or set FLYHARD_ROOT
import street_env as E
from train_street import StreetPolicy, DensePolicy

W, H = 1600, 800
DARK, MUTED, PALE, GREEN, RED = (12, 15, 18), (150, 162, 160), (237, 241, 228), (173, 220, 119), (235, 96, 80)
FONT_PATH = next((p for p in ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()), None)
font = lambda s: ImageFont.truetype(FONT_PATH, s) if FONT_PATH else ImageFont.load_default(size=s)


# Material tints for the chase-camera view; shading and haze come from the ray-cast itself.
TINT = np.array([[0, 0, 0], [0.38, 0.46, 0.30], [0.30, 0.31, 0.34], [0.93, 0.91, 0.80], [0.64, 0.62, 0.58], [0.74, 0.72, 0.66],
                 [1.0, 0.94, 0.84], [0.5, 0.52, 0.54], [0.82, 0.83, 0.86], [0.78, 0.22, 0.18], [0.22, 0.48, 0.84], [0.68, 0.86, 0.47],
                 [0.60, 0.82, 0.40], [0.40, 0.60, 0.32], [0.88, 0.16, 0.12], [0.90, 0.94, 1.0], [0.10, 0.11, 0.12]])   # fly body, head, eyes, wings, shadow
SKY_TOP, SKY_HORIZON = np.array([0.44, 0.61, 0.88]), np.array([0.80, 0.86, 0.94])


def colourise(view):
    """Colour a detailed render: tint by material, modulate by surface light, blend into haze."""
    mat, light, haze = view['material'], view['light'], view['haze']
    h, w = mat.shape
    v = (np.arange(h)/h)[:, None, None]
    sky = SKY_TOP*(1-v) + SKY_HORIZON*v
    rgb = TINT[mat]*light[..., None]
    rgb = rgb*haze[..., None] + SKY_HORIZON*(1-haze[..., None])
    return np.clip(np.where((mat == E.SKY)[..., None], sky*np.ones((h, w, 3)), rgb), 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--graph', default=os.environ.get('FLYHARD_ROOT', '../../../flyhard') + '/data/graph-traced-v1')
    p.add_argument('--seed', type=int, default=9000)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--big', type=int, default=640)
    args = p.parse_args()
    device = args.device
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
    policy.load_state_dict(ck['model']); policy.to(device).eval()

    road = E.Road(args.seed); car = E.Car(road, np.random.default_rng(args.seed+7919))
    bw, bh = args.big, args.big//2
    snaps, states, failure = [], [], None
    carried = torch.zeros(policy.core.n, 1, device=device) if is_conn else None
    for _ in range(args.steps):
        small = E.render(car); big = E.render_chase(car, bw, bh)
        with torch.no_grad():
            xi = torch.as_tensor(small[None], device=device)
            if is_conn:
                carried = policy.step_state(carried, xi); command = float(policy.readout(carried)[0])
                states.append(carried[valid, 0].cpu().numpy())
            else:
                command = float(policy(xi)[0])
        snaps.append({'t': car.t, 'small': small, 'big': big, 'command': command, 'lateral': car.lateral - E.LANE_CENTRE,
                      'x': car.x, 'steer': car.steer})
        car.step(command); failure = car.crashed()
        if failure:
            break
    # Per-neuron normalisation over the episode: each neuron's colour spans its own range.
    if is_conn:
        S_ = np.array(states)                                    # [T, N]
        amp = np.maximum(np.abs(S_).max(axis=0), 1e-9)      # floor only guards division; every neuron shows its own range
        norm = np.clip(S_/amp, -1, 1)
        pos = soma[valid][:, [0, 2]].astype(float); pos -= pos.min(axis=0)
        panel_w, panel_h = 560, 640
        pos *= min((panel_w-20)/np.ptp(pos[:, 0]), (panel_h-20)/np.ptp(pos[:, 1]))
        pos += [1010 + (panel_w-np.ptp(pos[:, 0]))/2, 120 + (panel_h-np.ptp(pos[:, 1]))/2]
        pos = pos.astype(int)
        order = np.argsort(np.abs(norm).mean(axis=0))          # draw the quiet ones first, bright on top

    label = (f"{cfg['graph_mode']} graph, state carried" if is_conn else f"{cfg['model']} baseline")
    writer = imageio.get_writer(args.out, fps=20, codec='libx264', quality=8, macro_block_size=1)
    for i, snap in enumerate(snaps):
        canvas = Image.new('RGB', (W, H), DARK); d = ImageDraw.Draw(canvas)
        d.text((27, 20), 'FLYHARD', fill=GREEN, font=font(34))
        d.text((225, 28), 'A fly connectome driving a street', fill=PALE, font=font(22))
        d.text((27, 68), f"165,122 neurons  /  25.56 million measured connections  /  {label}  /  one Apple M4", fill=MUTED, font=font(16))
        # Street.
        big = Image.fromarray((colourise(snap['big'])*255).astype(np.uint8)).resize((960, 480), Image.BILINEAR)
        canvas.paste(big, (27, 110))
        small = Image.fromarray((np.clip(snap['small'], 0, 1)*255).astype(np.uint8)).resize((192, 96), Image.NEAREST)
        canvas.paste(small, (27+960-192-12, 110+12)); d.rectangle([27+960-192-12, 122, 27+960-12, 218], outline=PALE, width=1)
        d.text((27+960-192-12, 220), "driver's eye, what the brain sees  64 x 32", fill=PALE, font=font(12))
        d.text((40, 118), f'HELD-OUT STREET {args.seed}     t = {snap["t"]:.2f} s     {snap["x"]:.0f} m', fill=PALE, font=font(18))
        # Readouts.
        d.text((27, 606), f"Steering  {np.degrees(snap['command']):+6.1f} deg", fill=GREEN, font=font(22))
        d.text((300, 606), f"Lane offset  {snap['lateral']:+5.2f} m", fill=PALE, font=font(18))
        d.text((560, 606), f"Speed  {E.SPEED:.0f} m/s, fixed", fill=MUTED, font=font(16))
        bar_x, bar_y = 27, 650
        d.rectangle([bar_x, bar_y, bar_x+400, bar_y+10], outline=MUTED); d.rectangle([bar_x+200, bar_y-4, bar_x+201, bar_y+14], fill=MUTED)
        k = bar_x + 200 + 200*(snap['command']/E.MAX_STEER); d.ellipse([k-7, bar_y-2, k+7, bar_y+12], fill=GREEN)
        d.text((27, 700), 'Hold the right lane, swerve past parked cars, overtake the slow car through the oncoming lane and pull back in.', fill=MUTED, font=font(14))
        d.text((27, 720), 'Pixels enter 4,114 photoreceptor neurons; steering is read from 708 motor neurons; the state persists between decisions.', fill=MUTED, font=font(14))
        if failure and i == len(snaps)-1:
            d.text((560, 640), {'off_road': 'LEFT THE ROAD', 'parked_car': 'HIT A PARKED CAR', 'oncoming_car': 'HIT ONCOMING TRAFFIC',
                                'traffic_car': 'HIT THE SLOW CAR'}[failure], fill=RED, font=font(24))
        # Brain.
        if is_conn:
            d.text((1010, 82), 'THE CONNECTOME, ALL 140,024 NEURONS WITH MEASURED POSITIONS', fill=GREEN, font=font(15))
            v = norm[i]
            m = np.abs(v)
            colour = np.where((v > 0)[:, None], np.stack([255*m, 190*m, 50*m], 1), np.stack([60*m, 150*m, 255*m], 1))
            colour[m < 0.04] = 0
            layer = np.zeros((H, W, 3), dtype=np.float32)
            np.maximum.at(layer, (pos[:, 1], pos[:, 0]), colour)          # brightest neuron per pixel, no additive white-out
            glow = np.asarray(Image.fromarray(np.clip(layer, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.8))).astype(np.float32)
            base = np.zeros((H, W, 3), dtype=np.float32); base[pos[::2, 1], pos[::2, 0]] = (34, 42, 42)
            merged = Image.fromarray(np.clip(np.maximum(base, glow*0.9 + layer*0.75), 0, 255).astype(np.uint8))
            canvas.paste(merged.crop((1000, 110, 1590, 770)), (1000, 110))
            d.text((1010, 772), 'Each neuron coloured on its own range over this drive: orange positive, blue negative, grey quiet.', fill=MUTED, font=font(13))
        writer.append_data(np.asarray(canvas))
        if i == min(len(snaps)-1, 220):
            canvas.save(Path(args.out).with_suffix('.png'))
    writer.close()
    print(json.dumps({'seed': args.seed, 'steps': len(snaps), 'failure': failure, 'distance_m': round(car.x, 1), 'out': args.out}))


if __name__ == '__main__':
    main()
