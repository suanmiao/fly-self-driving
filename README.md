# Fly Self Driving

A fruit fly's measured wiring diagram, trained to drive a simulated street from pixels.

The network is the [MaleCNS](https://male-cns.janelia.org/download/) connectome, 165,122 traced neurons and 25,563,197 measured synapses, run as a recurrent rate model the way the original [flyhard](https://github.com/MarkUnthank/flyhard) experiment runs it: the adjacency is frozen, only one gain per synapse and one leak per neuron train, and the sensory and motor interfaces are frozen random projections. It sees a 64 × 32 grayscale driver's-eye frame every 50 ms and returns a steering angle. On streets it never trained on, with parked cars, oncoming traffic and a slow car to overtake, the best model completes 58 of 60 across three independent sets of twenty, with no collisions with any car.

Videos, the write-up, and the results in one place: **https://fly-brain-drives.kylon.app**

Everything here was built in about three days in September 2026 by agents working in a [Kylon](https://kylon.io) workspace on two Apple M4 Mac minis and a rented H100, for under $25 of GPU time. Read the failures too: [docs/results.md](docs/results.md).

## What is in this repository

| Path | What it is |
| --- | --- |
| `flyhard-patches/` | Files to drop over a flyhard checkout: an Apple-GPU (MPS) kernel for the sparse connectome core, a CUDA backward that caches the transposed graph (2.5× faster), and the scripts we used to reproduce the original steering-wheel pilot on a Mac mini. |
| `tasks/flat-road/` | The first driving task: a curving flat road with obstacles, a 48 × 24 camera, steering only. Trainer, batched evaluator, and the linear / MLP baselines. |
| `tasks/street/` | The street task: two lanes, buildings, parked cars, oncoming traffic, a slow car to overtake; later speed control, pedestrians and dogs. Environment, expert, trainer with DAgger, evaluator, recorder. |
| `render/` | The replay renderer: rebuilds a recorded episode in Blender with [Kenney](https://kenney.nl) CC0 game kits and a procedural fly, composites the real 64 × 32 input and the recorded activity of every positioned neuron into the final video. |
| `results/` | Metrics files from the runs reported on the site, plus two recorded episodes of the best street model as examples of the state-log format. |
| `docs/` | The method, the full results with controls, and the honest caveats. |

Not in the repository: the connectome data (download it with flyhard's acquisition scripts, about 1 GB), trained checkpoints (about 600 MB each), Kenney assets (`render/fetch_assets.sh` downloads them), and the rendered videos (on the site).

## Setup

```bash
# 1. the original experiment, as a sibling directory; its scripts fetch and verify the MaleCNS data
git clone https://github.com/MarkUnthank/flyhard ../flyhard
cd ../flyhard && python -m venv .venv && .venv/bin/pip install -e . && \
  .venv/bin/python scripts/acquire_connectome.py && .venv/bin/python scripts/prepare_graph.py && cd -
# expected: ../flyhard/data/graph-traced-v1/graph.npz with sha256 eff4093b…

# 2. this repository
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 3. the patches (Apple GPU kernel + faster CUDA backward)
cp flyhard-patches/connectome.py ../flyhard/src/flyhard/connectome.py
cp flyhard-patches/connectome_mps.py ../flyhard/src/flyhard/connectome_mps.py
```

The task scripts import `flyhard` from `../flyhard/src` and expect the graph at `../flyhard/data/graph-traced-v1` relative to their own folder; both are command-line options if your layout differs.

## Train the street driver

```bash
cd tasks/street
# behaviour cloning from the pure-pursuit expert, then three rounds of DAgger
python train_street.py --model connectome --graph-mode measured --demo-steps 320 --overtake-weight 3 \
    --window 16 --batch 4 --steps 1200 --lr 0.04 --out runs/street-measured
python train_street.py --model connectome --graph-mode measured --demo-steps 320 --overtake-weight 3 \
    --init runs/street-measured/checkpoint.pt --steps 0 --dagger-rounds 3 --dagger-steps 500 --out runs/street-measured-dagger
# the control that asks whether the fly's wiring matters: same recipe, connections shuffled at random
python train_street.py --model connectome --graph-mode shuffled ...
# baselines that see the same pixels
python train_street.py --model linear --lr 0.001 --demo-steps 320 --device cpu --out runs/street-linear
python train_street.py --model mlp --hidden 40 --lr 0.001 --demo-steps 320 --device cpu --out runs/street-mlp40
# record a held-out street as per-tick state for the renderer
python record_episode.py --checkpoint runs/street-measured-dagger/checkpoint.pt --seed 9000 --steps 500 --out episodes/ep-9000
```

One connectome run is about 35 minutes on an H100 and about 100 minutes on an M4 Mac mini. On a Mac, run only one MPS training at a time; two PyTorch MPS processes on the same GPU stall each other.

## Render a drive

```bash
cd render && ./fetch_assets.sh
./render_best.sh ../tasks/street/episodes fly-street 9000 9006
```

## Results, briefly

Held-out streets completed, 20 per set, none seen in training.

| Model | With traffic | No traffic |
| --- | --- | --- |
| Expert (pure pursuit, reads the true state) | 20 / 20 | 20 / 20 |
| Linear map, pixels to steering, 1,153 parameters | 12 / 20 | 20 / 20 |
| MLP, one hidden layer of 40, 46k parameters | 18 / 20 | 20 / 20 |
| Fly connectome, behaviour cloning only | 12 / 20 | 20 / 20 |
| **Fly connectome + DAgger ×3, window 16** | **20 · 19 · 19 / 20** | 20 · 20 · 19 |
| Same recipe, randomly rewired graph | 16 / 20 | 18 / 20 |

The full tables, the crossers task with speed control, and the caveats (the fly does not brake yet; one training seed per condition) are in [docs/results.md](docs/results.md). The method is in [docs/method.md](docs/method.md).

## Credits

- [flyhard](https://github.com/MarkUnthank/flyhard) by Mark Unthank (MIT): the connectome-as-network recipe and the wheel pilot we reproduced first.
- [MaleCNS](https://male-cns.janelia.org/download/) (Janelia): the connectome.
- [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym): the fly body used in the wheel pilot.
- [DAgger](https://arxiv.org/abs/1011.0686), Ross, Gordon and Bagnell, 2011.
- [Kenney](https://kenney.nl) (CC0): the game kits in the renderer. [Blender](https://www.blender.org) renders it.

License: MIT (see LICENSE). Third-party materials keep their own terms.
