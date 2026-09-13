# Street task (task v5 code; v6 adds speed control and crossers)

A MaleCNS connectome (165,122 traced neurons, 25.56 M edges; flyhard recipe) drives a
ray-cast 3D street from pixels alone: 64x32 grayscale driver's-eye frames enter the 4,114
optic-lobe sensory neurons, steering is read from the 708 VNC motor neurons, and the neural
state is carried between the 50 ms decisions (4 graph updates per decision).

## Files

| file | role |
|---|---|
| `street_env.py` | scene + traffic + vehicle dynamics + expert + first-person renderer (`render`) and chase renderer (`render_chase`). `FLY_TASK_VERSION` env var selects the task version (3 = parked + oncoming only, 4 = slow car at 22 m grey, 5 = current: bright slow car, pull-out at 16 m). |
| `train_street.py` | behaviour cloning from the expert (`demonstrations`), stateful truncated-BPTT trainer, closed-loop `evaluate`, DAgger rounds (`collect_on_policy`), dense baselines (`--model linear|mlp`). |
| `render_street.py` | the current cinematic video (chase camera + driver's-eye inset + brain panel). |
| `record_episode.py` | replays a checkpoint on a street and writes the per-tick state log described below. |
| `connectome.py` | patched `flyhard/src/flyhard/connectome.py`: MPS gather kernel + cached transposed CSR for the CUDA backward (2.5x faster, same gradients). Drop it over the flyhard checkout. |
| `episodes/ep-v5-9000.{json,npz}` | one recorded episode of the current best checkpoint on held-out street 9000 (completed, 500 ticks). |
| `runs/*/metrics.json` | results of the runs listed below. |

Graph data is not in this repository: build `flyhard/data/graph-traced-v1` (graph sha256 `eff4093b…`) with the flyhard checkout's acquisition scripts. The scripts look for the flyhard checkout as a sibling of this repository, or wherever `FLYHARD_ROOT` points.

## Run commands

```
# connectome, task v5, the current best recipe (about 97 min on an M4, 35 min on an H100 alone)
python train_street.py --model connectome --graph-mode measured --demo-steps 320 --overtake-weight 3 \
    --window 8 --batch 4 --steps 1200 --lr 0.04 --out runs/street-v5-measured-stateful
# DAgger continuation from a checkpoint (on-policy relabelling, 3 rounds x 600 steps)
python train_street.py --model connectome --graph-mode measured --demo-steps 320 --overtake-weight 3 \
    --init runs/street-v5-measured-stateful/checkpoint.pt --steps 0 --dagger-rounds 3 --dagger-steps 600 --out runs/dagger-w8-r3
# dense baselines (memoryless)
python train_street.py --model linear --lr 0.001 --demo-steps 320 --device cpu --out runs/street-v5-linear
python train_street.py --model mlp --hidden 40 --lr 0.001 --demo-steps 320 --device cpu --out runs/street-v5-mlp40
# video and state log
python render_street.py --device cpu --checkpoint runs/<run>/checkpoint.pt --seed 9000 --steps 500 --out out.mp4
python record_episode.py --checkpoint runs/<run>/checkpoint.pt --seed 9000 --steps 500 --out episodes/ep-9000
```

Held-out streets are seeds 9000-9019; training streets 1000-1059. Evaluation is 500 decisions (25 s, 200 m) per street; a street counts as completed if no collision or off-road event occurs. Failure types: `off_road`, `parked_car`, `oncoming_car`, `traffic_car` (the slow car).

## Current numbers (task v5, 20 held-out streets with traffic / without traffic)

| model | with traffic | failure split | no traffic |
|---|---|---|---|
| expert (pure pursuit, sees state) | 20/20 | – | 20/20 |
| connectome, measured graph, state carried, w8 (`street-v5-measured-stateful`; reproduced on the H100 to the episode) | **12/20** | 8 oncoming | 20/20 |
| connectome, task v4 (grey slow car, 22 m pull-out) | 2/20 | 14 slow car, 3 oncoming, 1 off-road | 19/20 |
| linear pixels->steer (v4 scene) | 1/20 | 13 slow car, 3 oncoming, 2 parked, 1 off-road | 11/20 |
| MLP-40 (v4 scene) | 1/20 | 12 oncoming, 7 slow car | 20/20 |

The remaining connectome failures are drift: steering commands are 1-2 deg where the expert gives 8-18 deg, so it wanders toward the centre line and, after an overtake, into oncoming traffic. DAgger fixed it; longer BPTT windows alone (24 and 40 decisions) did not. See ../../docs/results.md.

## Episode state log (`episodes/ep-v5-9000`)

`ep-v5-9000.json`: `scene` (seed, geometry, `centreline` as [x, y] every metre, `buildings` as boxes with lateral extents relative to the centreline, `parked_cars` as [x, y, yaw] world, speeds, result) and `ticks`, one entry per 50 ms decision:
`t, x, y, yaw, steer, command, lateral, lane_offset, overtaking_expert_flag, traffic [[x, y, yaw], ...], oncoming [[x, y, yaw], ...]` — all world coordinates (x along the road, y positive to the LEFT of travel, z up; yaw radians). The vehicle box is 4.2 x 1.8 m; the collision test is the same box for every car. Oncoming yaw already includes the pi turn. Collision events: `scene.result.failure` is the event that ended the episode at the last tick (null if completed).

`ep-v5-9000.npz`: `t`, `pose [T, 5] = x, y, yaw, steer, command`, `soma_xyz [N, 3]` (measured soma positions of the N = 140,024 neurons that have one, MaleCNS voxel coordinates), `node_index [N]` (index into the 165,122-neuron graph), `superclass [N]`, `activity [T, N] float16` (the rate state after each decision; this is what the brain panel shows, each neuron normalised by its own |max| over the episode), `motor [T, 708]` (the VNC motor neurons the steering is read from).
