# Replay renderer

Rebuilds a recorded episode as a cinematic video without touching the brain's input: the fly and every car and crosser follow the logged positions, the inset is the exact 64 × 32 frame the brain consumed, and the panel on the right is the recorded activity of all 140,024 neurons with measured soma positions.

```bash
./fetch_assets.sh                                   # Kenney CC0 kits into assets/
./render_best.sh ../tasks/street/episodes drive 9000 9006
```

| File | What it does |
| --- | --- |
| `adapt_episode.py` | Converts a `record_episode.py` log (`ep-<seed>.json` + `.npz`) into the replay schema and extracts the recorded driver's-eye frames. |
| `build_scene.py` | Runs inside Blender (`blender -b -noaudio --python build_scene.py -- --log episode.json --out frames`). Builds the road ribbon with lane markings, curbs and sidewalks, places Kenney buildings on the recorded footprints, lamp posts, trees, parked and moving cars, pedestrians (Kenney mini-characters) and a procedural dog, animates the fly on the logged poses with a chase camera, and renders with EEVEE. About 1.5–3 s per 720p frame on an M4. |
| `fly_model.py` | The procedural fly: thorax, banded abdomen, red faceted compound eyes, six legs, halteres, wings keyed per frame. |
| `brain_panel.py` | Composites the frames with the inset, telemetry strip and the glowing connectome panel into a 1920 × 1080 mp4 with ffmpeg. |
| `synth_episode.py` | Writes a synthetic episode in the replay schema, for testing the renderer without a trained model. |

Blender 4.2+ (5.2 tested), ffmpeg, Pillow and numpy. Only one GPU job at a time on a Mac if a PyTorch MPS training is running on the same machine.
