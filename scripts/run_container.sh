#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
a=${APPTAINER_BIN:-apptainer}
options=(--writable --cleanenv)
if [[ -n ${PI05_HOST_BIND:-} ]]; then options+=(--bind "$PI05_HOST_BIND"); fi
if [[ ${PI05_GPU:-0} == 1 ]]; then
  options+=(--nv)
  smi=$(command -v nvidia-smi)
  test -x "$smi"
  test -e containers/pi05-devel/usr/bin/nvidia-smi || touch containers/pi05-devel/usr/bin/nvidia-smi
  options+=(--bind "$smi:/usr/bin/nvidia-smi:ro")
fi
mkdir -p cache/xdg cache/matplotlib cache/numba cache/huggingface tools/python
exec "$a" exec "${options[@]}" containers/pi05-devel env \
  PI05_ROOT="$root" OPENPI_DATA_HOME="$root/cache/openpi" \
  UV_CACHE_DIR="$root/cache/uv" UV_PYTHON_INSTALL_DIR="$root/tools/python" \
  XDG_CACHE_HOME="$root/cache/xdg" MPLCONFIGDIR="$root/cache/matplotlib" \
  NUMBA_CACHE_DIR="$root/cache/numba" HF_HOME="$root/cache/huggingface" \
  PYTHONPYCACHEPREFIX="/tmp/pi05-python-cache-${USER:-$(id -un)}" \
  LIBERO_CONFIG_PATH="$root/config/libero" \
  MUJOCO_GL="${PI05_RENDER_BACKEND:-egl}" PYOPENGL_PLATFORM="${PI05_RENDER_BACKEND:-egl}" \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 bash -c '
export LD_LIBRARY_PATH="$PI05_ROOT/sysroot/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBGL_DRIVERS_PATH="$PI05_ROOT/sysroot/root/usr/lib/x86_64-linux-gnu/dri"
export PATH="$PI05_ROOT/tools/build/cmake/data/bin:$PATH"
s=$PI05_ROOT/source/openpi-$(cat "$PI05_ROOT/source/OPENPI_COMMIT")
export PYTHONPATH="$s/third_party/libero:$s/packages/openpi-client/src"
exec "$@"
' bash "$@"
