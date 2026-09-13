# Results

All numbers are on held-out streets or roads generated from seeds the training never used. "20 · 19 · 19" means the same checkpoint scored on three independent sets of twenty. Metrics files are in `results/`.

## Street task with traffic (task v5)

Parked cars, oncoming traffic, a slow car to overtake. 20 streets per set.

| Model | With traffic | How it failed | No traffic |
| --- | --- | --- | --- |
| Expert (pure pursuit, reads the true state) | 20 / 20 | – | 20 / 20 |
| Steer straight | 0 / 20 | off road at 35 m | 0 / 20 |
| Linear map, pixels to steering (1,153 parameters) | 12 / 20 | oncoming, slow car | 20 / 20 |
| MLP, one hidden layer of 40 (46k parameters) | 18 / 20 | oncoming | 20 / 20 |
| Fly connectome, behaviour cloning only, window 8 | 12 / 20 | 8 oncoming | 20 / 20 |
| Fly connectome, window 24 or 40, cloning only | 8–11 / 20 | oncoming | 20 / 20 |
| Fly connectome + DAgger ×3, window 8 | 19 · 19 / 20 | 1 off road each | 20 / 20 |
| **Fly connectome + DAgger ×3, window 16** | **20 · 19 · 19 / 20** | 2 off road, no car collisions | 20 · 20 · 19 |
| Same recipe, randomly rewired graph | 16 / 20 | 4 off road | 18 / 20 |

Lane offset of the best model is within 4 cm of the expert's on every set (the expert itself sits about 1.1 m off lane centre on average because the overtake spends five seconds in the other lane). Two extra DAgger rounds did not improve it (20 · 19 · 19 again).

## Speed control, pedestrians and dogs (task v6)

Two outputs, steering and target speed. Crossers wait at a kerb and step off when the car is 26–38 m away; the expert brakes for them.

| Model | With traffic and crossers | How it failed | No traffic |
| --- | --- | --- | --- |
| Expert | 20 / 20 | – | 20 / 20 |
| Fly connectome, two outputs, DAgger ×3 | 18 · 19 · 16 / 20 | 1 off road, 2 parked, 2 oncoming, 2 crossers | 20 · 20 · 19 |
| Same, lower learning rate | 18 / 20 | 1 oncoming, 1 crosser | – |
| Same recipe, second training seed | 13 / 20 | 6 in the wait-for-a-gap overtake | – |
| Same recipe, randomly rewired graph | 5 / 20 | largely fails to learn | 5 / 20 |

**Read this one carefully.** Across all 20 recorded episodes of the best model the car's speed never dropped below 7.1 m/s. In every one of the 17 crossings it completed, the pedestrian or dog had already reached the far kerb before the car arrived, because the step-off distance gave them time to cross a road the car needed the same time to reach. The one time braking was genuinely needed (street 9016, a pedestrian ten metres ahead in the lane) it slowed by 0.5 m/s and hit them. The speed output learned to cruise because the demonstrations almost never contained a stop. A version with closer step-offs, where stopping is required, is the next experiment.

## The first driving task (flat road, task v1/v2)

A curving flat road with static obstacles, 48 × 24 camera, steering only, 20 held-out roads (`tasks/flat-road`).

| Model | v1 renderer (disc obstacles) | v2 renderer (upright obstacles) |
| --- | --- | --- |
| Linear map, 1,153 parameters, 3 seeds | 5 · 5 · 9 / 20 | 11 · 19 · 4 / 20 |
| MLP-40, 46k parameters, 3 seeds | 16 · 19 · 20 / 20 | 19 · 20 · 20 / 20 |
| MLP, 25.7 M parameters (matched), lr 3e-5 | 14 · 18 · 20 / 20 | – |
| Fly connectome, **state reset every decision** (flyhard recipe) | 3 / 20 (lr 0.01), 7 / 20 (lr 0.04), 6 / 20 (lr 0.1) | – |
| Fly connectome, **state carried** | 17 / 20 (19 / 20 without obstacles) | – |
| Same recipe, randomly rewired graph, state carried | did as well or better | – |

Two findings from this task shaped everything after it. Carrying the neural state between decisions is what makes the fly wiring usable at all. And on this task the specific wiring did not matter: a randomly rewired graph matched or beat it, and a 46k-parameter MLP beat both. The street task with traffic is the first where the measured wiring pulled ahead of the rewired control; with a second output to learn the rewired graph largely failed.

## The original steering-wheel pilot

Before any of this we reproduced flyhard's own pilot on a Mac mini: the connectome holding a requested steering-wheel angle through a simulated foreleg. 0 of 100 held-out targets before training, 100 of 100 after, same 25,728,319 trainable parameters, graph hash identical to the published one, worst hold error 4.41° against the published 4.85°. That demo is instructed steering, not driving: the policy has no camera input and the turn requests are scripted, as its author states. The scripts are in `flyhard-patches/scripts`.

## Caveats

- One training seed per condition, twenty streets per set. Gaps, not verdicts.
- Nothing here is a claim about a fly's mind or about biological learning. What is measured is the topology; what is learned is how loudly each synapse speaks.
- The renderer is presentation only. The brain saw the 64 × 32 frames in the recordings, never the Blender scene.
