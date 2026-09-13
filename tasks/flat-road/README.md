# Flat road task

The first driving task: a curving flat road with static obstacles, a 48 × 24 ray-cast camera, a kinematic bicycle car at 8 m/s, steering only. Deliberately not CARLA: everything is numpy so it runs on a Mac, and it asks the one question the original demo could not: given pixels, can the connectome keep a car on a road and off the obstacles.

| File | What it is |
| --- | --- |
| `drive_env.py` | Road, car, ray-cast renderer, obstacle-aware pure-pursuit expert, closed-loop `rollout`. `FLY_TASK_VERSION=1` reproduces the first renderer (obstacles as ground discs a 24-row image cannot resolve beyond ~8 m); the default v2 raycasts upright cylinders. |
| `train_vision.py` | Connectome trainer: noise-injected demonstrations, frozen random pixel → `ol_sensory` map and `vnc_motor` → steering readout, `--graph-mode shuffled` for the rewired control. As shipped it resets the state every decision, the flyhard way; the state-carried variant used for the street task lives in `../street`. |
| `train_vision_mps.py` | Same trainer with the Apple-GPU drop-in installed first. |
| `eval_batched.py` | Steps all 20 held-out roads in lockstep with one batched forward pass per tick; matches the serial evaluator and is ~20× faster. |
| `mlp_baseline.py` | The linear / small-MLP / parameter-matched-MLP baselines on the identical protocol (`demonstrations()` and `evaluate()` imported unchanged from the trainer). |
| `render_drive.py` | The simple top-down / driver's-eye video for this task. |

Protocol: training roads 1000–1059, held-out roads 9000–9019, 300 noise-injected (σ = 0.06) demo steps per road, batch 16, lr 0.04, 1,200 updates, 500-tick evaluation. Results are in `../../results/flat-road` and `../../docs/results.md`.
