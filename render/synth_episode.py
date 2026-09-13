"""Synthetic episode log in the replay schema, for testing the renderer before the real state log arrives.
Geometry mirrors drive_env: sinusoidal centreline, 3.5 m half-width, parked cars, a slow car ahead, oncoming cars."""
import json, math, sys
import numpy as np

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
out = sys.argv[2] if len(sys.argv) > 2 else 'synth_episode.json'
rng = np.random.default_rng(seed)
amps = rng.uniform(1.5, 4.5, 3) * rng.choice([-1.0, 1.0], 3)
freqs = rng.uniform(0.018, 0.048, 3)
phases = rng.uniform(0, 2 * np.pi, 3)
centre = lambda x: float(sum(a * np.sin(f * x + p) for a, f, p in zip(amps, freqs, phases)))
heading = lambda x: float(np.arctan(sum(a * f * np.cos(f * x + p) for a, f, p in zip(amps, freqs, phases))))

DT, SPEED, N = 0.05, 8.0, 500
LANE = 1.75   # right-lane centre offset
xs = np.arange(-30.0, 300.0, 1.0)
road = {'x': xs.tolist(), 'y': [centre(x) for x in xs]}

def lateral(x, off):
    h = heading(x)
    return x - math.sin(h) * off, centre(x) + math.cos(h) * off, h

parked = []
for px in np.arange(40, 220, 48.0):
    px += rng.uniform(-5, 5)
    x, y, h = lateral(px, -(3.5 - 0.9))
    parked.append({'x': x, 'y': y, 'heading': h, 'kind': str(rng.choice(['sedan', 'taxi', 'hatchback', 'van']))})

fly, slow, oncoming = [], [], []
x = 5.0
slow_x = 30.0
onc_x = [120.0, 190.0, 260.0]
for k in range(N):
    off = -LANE
    # overtake the slow car between 60 and 100 m
    if 45 < x < 75:
        off = -LANE + (x - 45) / 30 * 2 * LANE
    elif 75 <= x < 100:
        off = LANE
    elif 100 <= x < 125:
        off = LANE - (x - 100) / 25 * 2 * LANE
    steer = 0.25 * (off - (-LANE)) / LANE if 45 < x < 75 else (-0.25 * (off - LANE) / LANE if 100 <= x < 125 else 0.0)
    fx, fy, fh = lateral(x, off)
    fly.append([fx, fy, fh, steer])
    sx, sy, sh = lateral(slow_x, -LANE)
    slow.append([sx, sy, sh])
    row = []
    for o in onc_x:
        ox, oy, oh = lateral(o, LANE)
        row.append([ox, oy, oh + math.pi])
    oncoming.append(row)
    x += SPEED * DT
    slow_x += 4.5 * DT
    onc_x = [o - 7.0 * DT for o in onc_x]

traffic = [{'kind': 'suv', 'poses': slow}] + [{'kind': str(rng.choice(['sedan', 'police', 'sports'])), 'poses': [r[i] for r in oncoming]} for i in range(3)]
json.dump({'dt': DT, 'lane_half_width': 3.5, 'road': road, 'parked': parked, 'traffic': traffic,
           'fly': {'poses': fly}, 'collision': {'tick': None, 'with': None}}, open(out, 'w'))
print('wrote', out, 'ticks', N)
