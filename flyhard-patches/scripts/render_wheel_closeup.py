#!/usr/bin/env python3
"""Re-render a recorded trace with a camera looking straight down the wheel axle, at 1x speed.

Same recorded qpos as the upstream render; only the camera, playback speed and
overlay differ. Purpose: make the (small, fast) wheel rotation visible.
"""
import argparse, json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
from flyhard.cockpit import WheelRig


def main():
    p = argparse.ArgumentParser(); p.add_argument('--trace', required=True); p.add_argument('--out', required=True)
    p.add_argument('--speed', type=float, default=1.0); p.add_argument('--fps', type=int, default=30)
    a = p.parse_args(); trace = np.load(a.trace)
    rig = WheelRig(); data = mj.MjData(rig.model)
    W, H = 720, 720
    renderer = mj.Renderer(rig.model, height=H, width=W)
    cam = mj.MjvCamera(); cam.lookat[:] = rig.center; cam.distance = 3.2; cam.azimuth = 0; cam.elevation = 0  # along +x axle
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    font = lambda s: ImageFont.truetype(font_path, s) if Path(font_path).exists() else ImageFont.load_default(size=s)
    # 200 Hz body samples; pick every k-th sample so that playback is `speed` x real time at `fps`.
    stride = max(1, int(round(200 * a.speed / a.fps)))
    replay_err = 0.0
    with imageio.get_writer(a.out, fps=a.fps, quality=8, macro_block_size=1) as video:
        for i in range(0, len(trace['time']), stride):
            data.qpos[:] = trace['qpos'][i]; data.time = float(trace['time'][i]); mj.mj_forward(rig.model, data)
            replay_err = max(replay_err, float(np.abs(data.qpos - trace['qpos'][i]).max()))
            renderer.update_scene(data, camera=cam)
            frame = Image.fromarray(renderer.render()); d = ImageDraw.Draw(frame)
            t = float(trace['time'][i]); ep = int(trace['episode'][i]); tgt = np.degrees(float(trace['target'][i])); ang = np.degrees(float(trace['angle'][i]))
            d.rectangle((0, 0, W, 78), fill=(15, 20, 22))
            d.text((16, 10), f'Held-out trial {ep+1}/2   t = {t:.2f} s   playback {a.speed:g}x   camera on wheel axle', fill=(237, 241, 228), font=font(20))
            d.text((16, 42), f'Requested {tgt:+.1f} deg    Physical wheel {ang:+.1f} deg', fill=(173, 220, 119), font=font(22))
            # dial: needle for wheel angle and tick for target
            cx, cy, R = W - 90, H - 90, 60
            d.ellipse((cx-R, cy-R, cx+R, cy+R), outline=(154, 168, 165), width=2)
            for val, col, w in [(tgt, (154, 168, 165), 3), (ang, (173, 220, 119), 5)]:
                th = np.radians(90 - val)
                d.line((cx, cy, cx + R*np.cos(th), cy - R*np.sin(th)), fill=col, width=w)
            video.append_data(np.asarray(frame))
    renderer.close()
    print(json.dumps({'frames': len(range(0, len(trace['time']), stride)), 'replay_max_qpos_error': replay_err, 'speed': a.speed}))


if __name__ == '__main__':
    main()
