#!/usr/bin/env bash
# Fallback for a CUDA runtime image without Git. Keep the upstream package pins.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
sha=$(cat source/OPENPI_COMMIT)
src=$root/source/openpi-$sha
uv=$root/tools/uv/bin/uv
mkdir -p logs source/lerobot
(
  cd "$src"
  "$uv" export --frozen --no-dev --no-emit-workspace --no-emit-package lerobot \
    --no-hashes --output-file "$root/requirements.locked.txt"
)
curl -fL --retry 2 --max-time 120 \
  https://codeload.github.com/huggingface/lerobot/tar.gz/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 \
  -o source/lerobot.tar.gz
tar -xzf source/lerobot.tar.gz --strip-components=1 -C source/lerobot
/public/software/apps/apptainer/1.3.4/bin/apptainer exec --cleanenv --bind /public:/public \
  containers/cuda12-ubuntu22.sif env PI05_ROOT="$root" UV_CACHE_DIR="$root/cache/uv" \
  bash "$root/scripts/install_inside.sh"
