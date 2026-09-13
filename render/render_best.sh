#!/usr/bin/env bash
# Render one or more recorded episodes end to end: Blender replay frames -> composite with the
# brain panel -> one mp4 per episode -> concatenated video.
#
#   ./render_best.sh <episodes_dir> <out_name> <seed> [<seed> ...]
#
# <episodes_dir> holds ep-<seed>.json and ep-<seed>.npz as written by tasks/street/record_episode.py.
# Requires: blender and ffmpeg on PATH, python with the requirements installed, render/assets/ fetched.
set -euo pipefail
cd "$(dirname "$0")"
EPS=${1:?episodes dir}; OUT=${2:?output name}; shift 2
PY=${PY:-python3}
parts=()
for s in "$@"; do
  mkdir -p "out/$s"
  echo "=== $(date '+%H:%M:%S') adapt $s"
  $PY adapt_episode.py "$EPS/ep-$s.json" "out/$s"
  echo "=== $(date '+%H:%M:%S') render $s"
  blender -b -noaudio --python build_scene.py -- --log "out/$s/episode.json" --out "out/$s/frames" --res 1280x720 > "out/$s/render.log" 2>&1
  echo "=== $(date '+%H:%M:%S') composite $s ($(ls out/$s/frames | grep -c png) frames)"
  $PY brain_panel.py --npz "$EPS/ep-$s.npz" --json "$EPS/ep-$s.json" --views "out/$s/views.npy" --frames "out/$s/frames" \
      --out "out/$s/street-$s.mp4" --title "FLY SELF DRIVING  ·  a fly connectome driving a street  ·  held-out street $s" > "out/$s/composite.log" 2>&1
  parts+=("out/$s/street-$s.mp4")
done
printf "file '%s'\n" "${parts[@]}" > out/concat.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i out/concat.txt -c copy "out/$OUT.mp4"
echo "=== $(date '+%H:%M:%S') FINAL out/$OUT.mp4"
