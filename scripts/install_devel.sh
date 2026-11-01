#!/usr/bin/env bash
# Requires source/lerobot, requirements.locked.txt and .venv from install_locked.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
a=/public/software/apps/apptainer/1.3.4/bin/apptainer
export APPTAINER_CACHEDIR=$root/cache/apptainer
image=containers/cuda12-devel-ubuntu22.sif
if ! test -s "$image"; then
  "$a" pull "$image" docker://nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
fi
"$a" exec --cleanenv --bind /public:/public "$image" env PI05_ROOT="$root" \
  UV_CACHE_DIR="$root/cache/uv" bash "$root/scripts/install_inside.sh"
