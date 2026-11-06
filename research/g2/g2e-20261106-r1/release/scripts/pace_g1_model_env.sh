#!/usr/bin/env bash
# Use the existing standalone OpenPI environment, separate from RoboTwin.
set -euo pipefail
mode=${1:?Expected cpu or GPU UUID}
shift
root="${PI05_ROOT:?}"
a=${APPTAINER_BIN:-/public/software/apps/apptainer/1.3.4/bin/apptainer}
# Reuse the existing extracted Ubuntu image to avoid expanding the SIF on each launch.
image="$root/containers/openpi-ubuntu"
python="$root/tools/pace-model-venv/bin/python"
test -r "$image" && test -x "$python"
printf '[%s] model_container_launch mode=%s\n' "$(date -u +%FT%TZ)" "$mode"
options=(--writable --cleanenv --bind /public:/public --pwd "$root")
settings=(JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=)
if [[ "$mode" != cpu ]]; then
 [[ "$mode" == GPU-* ]]
 options+=(--nv)
 settings=(JAX_PLATFORMS=cuda "CUDA_VISIBLE_DEVICES=$mode" JAX_LOG_COMPILES=1)
fi
exec "$a" exec "${options[@]}" "$image" env \
 "PATH=$root/tools/pace-git/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
 "LD_LIBRARY_PATH=$root/sysroot/root/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
 "OPENPI_DATA_HOME=$root/cache/openpi-pace" "XDG_CACHE_HOME=$root/cache/openpi-pace/xdg" \
 "HF_HOME=$root/cache/openpi-pace/hf" "CUDA_CACHE_PATH=$root/cache/openpi-pace/cuda" \
 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
 XLA_PYTHON_CLIENT_PREALLOCATE=false "${settings[@]}" "$python" "$@"
