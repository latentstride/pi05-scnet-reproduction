#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
a=${APPTAINER_BIN:-/public/software/apps/apptainer/1.3.4/bin/apptainer}
options=(--writable --cleanenv --bind /public:/public)
if [[ ${PI05_GPU:-0} == 1 ]]; then options+=(--nv); fi
exec "$a" exec "${options[@]}" containers/pi05-devel env \
  PI05_ROOT="$root" OPENPI_DATA_HOME="$root/cache/openpi" \
  UV_CACHE_DIR="$root/cache/uv" PYTHONPYCACHEPREFIX="/tmp/pi05-python-cache-$USER" \
  LIBERO_CONFIG_PATH="$root/config/libero" \
  MUJOCO_GL="${PI05_RENDER_BACKEND:-egl}" PYOPENGL_PLATFORM="${PI05_RENDER_BACKEND:-egl}" \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 bash -c '
export LD_LIBRARY_PATH="$PI05_ROOT/sysroot/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBGL_DRIVERS_PATH="$PI05_ROOT/sysroot/root/usr/lib/x86_64-linux-gnu/dri"
exec "$@"
' bash "$@"
