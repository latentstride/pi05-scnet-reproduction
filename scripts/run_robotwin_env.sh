#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
image="$root/research/robotwin/robotwin-cu122"
a=${APPTAINER_BIN:-/public/software/apps/apptainer/1.3.4/bin/apptainer}
options=(--writable --cleanenv --bind /public:/public --pwd /opt/robotwin/src/RoboTwin)
if [[ ${ROBOTWIN_GPU:-0} == 1 ]]; then options+=(--nv); fi
exec "$a" exec "${options[@]}" "$image" env \
 PI05_ROOT="$root" XDG_CACHE_HOME=/opt/robotwin/cache/xdg \
 TORCH_EXTENSIONS_DIR=/opt/robotwin/cache/torch_extensions CUDA_CACHE_PATH=/opt/robotwin/cache/cuda \
 TMPDIR=/opt/robotwin/tmp \
 VK_ICD_FILENAMES=/opt/robotwin/sysroot/etc/vulkan/icd.d/nvidia_icd.json \
 PATH=/opt/robotwin/venv/bin:/opt/robotwin/sysroot/usr/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
 LD_LIBRARY_PATH=/opt/robotwin/sysroot/usr/lib/x86_64-linux-gnu:/opt/robotwin/sysroot/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/.singularity.d/libs \
 PYTHONPYCACHEPREFIX=/opt/robotwin/cache/pycache "$@"
