"""Render the connectome activity panel per tick from the recorded episode (.npz) and composite it
with the Blender street frames and the real 64x32 driver's-eye view into the final video.

  python brain_panel.py --npz task/street/episodes/ep-v5-9000.npz --views ep9000/views.npy \
      --frames ep9000/frames --out ep9000/final.mp4 [--fps 20] [--title ...]

Layout (1920x1080): street render 1280x720 top-left, driver's-eye inset in its corner, telemetry strip
below it, brain panel 640 wide on the right (dorsal view: soma x vs z, every neuron a point coloured by
its own normalised rate: orange positive, blue negative, grey quiet).
"""
import argparse, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080
PANEL_X, PANEL_W = 1300, 600
BG = (10, 11, 14)


def font(size):
    for p in ('/System/Library/Fonts/Helvetica.ttc', '/System/Library/Fonts/SFNS.ttf', '/Library/Fonts/Arial.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--json', default=None, help='episode json (for telemetry: lateral, steer, failure)')
    ap.add_argument('--views', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=int, default=20)
    ap.add_argument('--first-tick', type=int, default=0)
    ap.add_argument('--title', default='FLYHARD  ·  a fly connectome driving a street')
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    soma, motor, pose = z['soma_xyz'], z['motor'], z['pose']
    if 'activity' in z.files:
        act = z['activity']
    else:  # int8 per-neuron quantised: activity = activity_q * activity_scale
        act = z['activity_q'].astype(np.float32) * np.asarray(z['activity_scale'], np.float32).reshape(1, -1)
    ticks = None
    if args.json:
        import json
        ticks = json.load(open(args.json))['ticks']
    views = np.load(args.views)
    # dorsal projection: x across, z along the body axis (MaleCNS voxel coords)
    pos = soma[:, [0, 2]].astype(np.float64)
    pos -= pos.min(axis=0)
    span = pos.max(axis=0)
    ph = 760
    sc = min((PANEL_W - 40) / span[0], (ph - 40) / span[1])
    pos = (pos * sc).astype(np.int32)
    px = pos[:, 0] + PANEL_X + int((PANEL_W - span[0] * sc) / 2)
    py = pos[:, 1] + 150 + int((ph - span[1] * sc) / 2)
    # per-neuron normalisation over the episode, as in the training renderer
    amax = np.abs(act.astype(np.float32)).max(axis=0)
    amax[amax < 1e-6] = 1.0
    order = np.argsort(np.abs(act[0].astype(np.float32)))  # draw strongest last; recomputed per frame below

    frame_files = sorted(f for f in os.listdir(args.frames) if f.startswith('frame_') and f.endswith('.png'))
    n = len(frame_files)
    tmp = os.path.join(os.path.dirname(args.out) or '.', 'composite')
    os.makedirs(tmp, exist_ok=True)
    f_title, f_sub, f_num, f_small = font(40), font(22), font(30), font(18)
    lut_pos = np.array([255, 140, 40], np.float32)
    lut_neg = np.array([70, 140, 255], np.float32)
    quiet = np.array([48, 50, 56], np.float32)
    for i, fn in enumerate(frame_files):
        tk = args.first_tick + i
        canvas = Image.new('RGB', (W, H), BG)
        street = Image.open(os.path.join(args.frames, fn)).convert('RGB').resize((1260, 709))
        canvas.paste(street, (20, 90))
        # driver's-eye inset (the actual 64x32 input)
        v = views[min(tk, len(views) - 1)]
        inset = Image.fromarray((np.clip(v, 0, 1) * 255).astype(np.uint8), 'L').resize((256, 128), Image.NEAREST).convert('RGB')
        canvas.paste(inset, (20 + 1260 - 256 - 12, 90 + 12))
        d = ImageDraw.Draw(canvas)
        d.rectangle((20 + 1260 - 256 - 13, 90 + 11, 20 + 1260 - 12, 90 + 12 + 128), outline=(200, 200, 200))
        d.text((20 + 1260 - 256 - 12, 90 + 12 + 132), 'what the brain sees  64 × 32', fill=(190, 190, 190), font=f_small)
        # brain panel
        a = act[min(tk, len(act) - 1)].astype(np.float32) / amax
        mag = np.abs(a) ** 0.6                      # lift mid-range activity so the brain reads as alive
        col = np.where(a[:, None] >= 0, lut_pos, lut_neg) * mag[:, None] + quiet * (1 - mag)[:, None]
        idx = np.argsort(mag)
        panel = np.zeros((H, W, 3), np.float32)
        for dy in (0, 1):
            for dx in (0, 1):
                panel[np.clip(py[idx] + dy, 0, H - 1), np.clip(px[idx] + dx, 0, W - 1)] = col[idx]
        glow = Image.fromarray(np.clip(panel, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
        arr = np.array(canvas).astype(np.float32)
        arr = np.clip(arr + 0.9 * np.array(glow).astype(np.float32) + panel, 0, 255)
        canvas = Image.fromarray(arr.astype(np.uint8))
        d = ImageDraw.Draw(canvas)
        # text
        d.text((20, 22), args.title, fill=(150, 220, 90), font=f_title)
        d.text((PANEL_X, 100), 'THE CONNECTOME · 140,024 neurons, measured positions', fill=(150, 220, 90), font=f_sub)
        d.text((PANEL_X, 930), 'orange: positive rate · blue: negative · grey: quiet\neach neuron scaled on its own range over this drive', fill=(150, 150, 150), font=f_small)
        row = pose[min(tk, len(pose) - 1)]
        x, y, yaw, steer = row[:4]
        d.text((20, 815), f'steering  {math_deg(steer):+.1f}°', fill=(150, 220, 90), font=f_num)
        if len(row) >= 7:   # v6: x, y, yaw, steer, steer_command, speed, target_speed
            d.text((330, 815), f'speed  {row[5] * 3.6:4.0f} km/h', fill=(150, 220, 90), font=f_num)
            d.text((640, 815), f't = {tk * 0.05:5.2f} s   {x:5.0f} m', fill=(200, 200, 200), font=f_num)
        else:
            d.text((330, 815), f't = {tk * 0.05:5.2f} s     {x:5.0f} m', fill=(200, 200, 200), font=f_num)
        if ticks is not None:
            t = ticks[min(tk, len(ticks) - 1)]
            d.text((640 if len(row) < 7 else 960, 815), f'lane offset {t["lane_offset"]:+.2f} m', fill=(200, 200, 200), font=f_num)
        d.text((20, 870), 'Pixels enter 4,114 photoreceptor neurons; steering is read from 708 motor neurons; the neural state persists between decisions.', fill=(150, 150, 150), font=f_small)
        d.text((20, 895), 'Task: hold the right lane, pass parked cars, overtake the slow car through oncoming traffic, pull back in.', fill=(150, 150, 150), font=f_small)
        canvas.save(os.path.join(tmp, f'c_{i:04d}.png'))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps), '-i', os.path.join(tmp, 'c_%04d.png'),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', args.out], check=True)
    print('wrote', args.out, n, 'frames')


def math_deg(r):
    return r * 180 / 3.141592653589793


if __name__ == '__main__':
    main()
