"""Convert Jack Opus's recorded street episode (task/street/episodes/ep-*.json) into the replay schema
read by build_scene.py. World frame is shared: x along the road, y to the left, yaw radians.

Also regenerates the 64x32 driver's-eye frames the brain saw (from seed + recorded pose, using the
task's own renderer) so the inset in the final video is the real input, not an approximation.

Usage: python adapt_episode.py task/street/episodes/ep-v5-9000.json out_dir/
"""
import json, os, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
d = json.load(open(src))
s, ticks = d['scene'], d['ticks']

kinds = ['sedan', 'taxi', 'hatchback', 'van', 'suv', 'sports', 'police', 'delivery']
rng = np.random.default_rng(s['seed'])
log = {
    'dt': s['dt'], 'lane_half_width': s['road_half_width'], 'seed': s['seed'], 'task_version': s['task_version'],
    'road': {'x': [p[0] for p in s['centreline']], 'y': [p[1] for p in s['centreline']]},
    'buildings': s.get('buildings', []),
    'parked': [{'x': p[0], 'y': p[1], 'heading': p[2], 'kind': str(rng.choice(kinds))} for p in s['parked_cars']],
    'fly': {'poses': [[t['x'], t['y'], t['yaw'], t['steer']] for t in ticks]},
    'collision': {'tick': None if s['result']['failure'] in (None, 'None') else len(ticks) - 1,
                  'with': None if s['result']['failure'] in (None, 'None') else s['result']['failure']},
}
n_tr = max(len(t['traffic']) for t in ticks)
n_on = max(len(t['oncoming']) for t in ticks)
traffic = []
for j in range(n_tr):
    traffic.append({'kind': 'suv', 'role': 'slow', 'poses': [t['traffic'][j] if j < len(t['traffic']) else t['traffic'][-1] for t in ticks]})
for j in range(n_on):
    traffic.append({'kind': str(rng.choice(['sedan', 'police', 'sports', 'taxi'])), 'role': 'oncoming',
                    'poses': [t['oncoming'][j] if j < len(t['oncoming']) else [1e6, 0, 0] for t in ticks]})
log['traffic'] = traffic
# v6 additions: speed control and crossers (pedestrians / dogs) per tick
if s.get('speed_control'):
    log['speed'] = [[t.get('speed'), t.get('target_speed')] for t in ticks]
    log['crosser_kinds'] = s.get('crosser_kinds', {})
    log['crossers'] = [t.get('crossers', []) for t in ticks]   # each: [x, y, kind, side]
json.dump(log, open(os.path.join(out, 'episode.json'), 'w'))
print('wrote', os.path.join(out, 'episode.json'), 'ticks', len(ticks), 'traffic', len(traffic), 'parked', len(log['parked']), 'buildings', len(log['buildings']))

# driver's-eye frames: recorded `views` (v6 npz, uint8 T x 32 x 64) when present, else regenerated
npz_path = src[:-5] + '.npz'
if os.path.exists(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    if 'views' in z.files:
        views = z['views'].astype(np.float32) / 255.0
        np.save(os.path.join(out, 'views.npy'), views)
        print('views (recorded)', views.shape)
        sys.exit(0)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(src)), '..'))
os.environ.setdefault('FLY_TASK_VERSION', str(s['task_version']))
import street_env as E  # noqa: E402
road = E.Road(s['seed'])
car = E.Car(road, np.random.default_rng(s['seed'] + 7919))
views = []
for t in ticks:
    car.x, car.y, car.theta, car.t = t['x'], t['y'], t['yaw'], t['t']
    views.append(E.render(car))
views = np.stack(views)
np.save(os.path.join(out, 'views.npy'), views)
print('views', views.shape, views.dtype, float(views.min()), float(views.max()))
