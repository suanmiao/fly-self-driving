# flyhard patches

Files to drop over a checkout of [flyhard](https://github.com/MarkUnthank/flyhard) (MIT, Mark Unthank). They change nothing about the model, its parameters or its topology; they make the same mathematics run on other hardware and add the scripts we used to reproduce the pilot.

| File | Copy to | What it does |
| --- | --- | --- |
| `connectome_mps.py` | `src/flyhard/connectome_mps.py` | Apple-GPU (MPS) path for the sparse core. PyTorch has no sparse-CSR kernels on MPS, so this installs a drop-in `_EdgeSparseMM` built from dense gather + chunked `index_add_`. Validated against the CPU sparse path to ~1e-6 forward and ~1e-8 on edge gradients. Import it once before building the model; it is a no-op on CPU/CUDA. |
| `connectome.py` | `src/flyhard/connectome.py` | The upstream core plus a cached transposed CSR for the CUDA backward. Upstream re-transposed the 25.5 M-edge graph on every step; caching it is 2.5× faster per step with gradients equal to 1e-4. Also dispatches to the MPS path when the model lives on `mps`. |
| `scripts/train_wheel_mps.py` | `scripts/` | Upstream wheel-task training with only the device line changed. |
| `scripts/verify_checkpoint.py` | `scripts/` | Replays a checkpoint on all 100 held-out wheel targets and records the held angles. |
| `scripts/render_wheel_closeup.py` | `scripts/` | Renders a 1× axle-view video of the learned wheel turn. |

Run with `PYTHONPATH=src MUJOCO_GL=cgl` on macOS. Pinned dependencies that worked: Python 3.12, torch 2.14, mujoco 3.9.0, flygym at commit 38c8ec6.
