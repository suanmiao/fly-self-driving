#!/usr/bin/env bash
# Download the Kenney CC0 kits the replay renderer uses into render/assets/.
# Kenney (https://kenney.nl) releases these under CC0 1.0; no attribution is required, but it is appreciated.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p assets && cd assets
for slug in car-kit city-kit-roads city-kit-commercial city-kit-suburban nature-kit mini-characters; do
  if [ -d "$slug" ]; then echo "have $slug"; continue; fi
  url=$(curl -sL "https://kenney.nl/assets/$slug" | grep -o 'https://kenney.nl/media/pages/assets/[^"]*\.zip' | head -1)
  if [ -z "$url" ]; then echo "could not find a download link for $slug; open https://kenney.nl/assets/$slug and unzip it here" >&2; continue; fi
  echo "fetching $slug"
  curl -sL -o "$slug.zip" "$url" && mkdir -p "$slug" && unzip -oq "$slug.zip" -d "$slug" && rm "$slug.zip"
done
echo "done: $(ls -d */ | tr '\n' ' ')"
