#!/bin/bash
# Cache Isaac Lab's unchanged ground asset so simulation startup needs no S3 download.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Environments/Grid/default_ground_plane_checker_v1
for file in default_ground_plane.usda Materials/Textures/default_ground_plane_albedo.png Materials/Textures/default_ground_plane_roughness.png; do
  target="data/ground/$file"
  mkdir -p "$(dirname "$target")"
  curl --fail --show-error --silent --location --retry 2 --connect-timeout 10 --max-time 60 \
    "$BASE/$file" -o "$target.part"
  mv "$target.part" "$target"
done
