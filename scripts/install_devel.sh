#!/usr/bin/env bash
# Requires source/lerobot, requirements.locked.txt and .venv from install_locked.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
a=/public/software/apps/apptainer/1.3.4/bin/apptainer
export APPTAINER_CACHEDIR=$root/cache/apptainer
mkdir -p containers cache/apptainer logs
image=containers/cuda12-devel-ubuntu22.sif
if ! test -s "$image"; then
  "$a" pull "$image" docker://nvcr.io/nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
fi
if ! test -d containers/pi05-devel/.singularity.d; then
  "$a" build --sandbox containers/pi05-devel "$image"
  mkdir -p containers/pi05-devel/public
fi
bash scripts/run_container.sh bash "$root/scripts/install_inside.sh"
