#!/usr/bin/env bash
# Shared CPU and allocated-GPU entry point. No downloads or scheduler operations.
set -euo pipefail
mode=${1:?Expected cpu or gpu}
root=${2:?Expected the absolute SCNet project root}
shift 2
[[ "$root" == /* && "$root" == "$(cd "$root" && pwd -P)" && "$root" == */code_cpf/pi05 ]]
code=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
[[ "$code" == "$root/"* ]]
src="$root/source/Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae"
a=${APPTAINER_BIN:-/public/software/apps/apptainer/1.3.4/bin/apptainer}
image="$root/containers/openpi-ubuntu"
python="$root/tools/pace-model-venv/bin/python"
[[ -x "$a" && -x "$python" && -d "$image" && -r "$src/src/openpi/models/model.py" ]]
[[ -r "$root/cache/pace-handover/handover_block/params/manifest.ocdbt" ]]
cd "$root"
args=(--writable --cleanenv --bind /public:/public --pwd "$root")
case "$mode" in
    cpu) platform=cpu; visible= ;;
    gpu)
        [[ -n "${SLURM_JOB_ID:-}" && "${CUDA_VISIBLE_DEVICES:-}" =~ ^([0-9]+|GPU-[0-9A-Fa-f-]+)$ ]]
        platform=cuda; visible=$CUDA_VISIBLE_DEVICES; args+=(--nv)
        ;;
    *) exit 2 ;;
esac
exec "$a" exec "${args[@]}" "$image" env \
    "PATH=$root/tools/pace-git/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    "LD_LIBRARY_PATH=$root/sysroot/root/usr/lib/x86_64-linux-gnu:/.singularity.d/libs" \
    "PYTHONPATH=$code:$src/src:$src/pace/src:$src/packages/openpi-client/src" \
    "OPENPI_DATA_HOME=$root/cache/openpi-pace" "XDG_CACHE_HOME=$root/cache/step5-arx/xdg" \
    "HF_HOME=$root/cache/step5-arx/hf" "CUDA_CACHE_PATH=$root/cache/step5-arx/cuda" \
    "JAX_COMPILATION_CACHE_DIR=$root/cache/step5-arx/jax" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "JAX_PLATFORMS=$platform" "CUDA_VISIBLE_DEVICES=$visible" "$python" "$@"
