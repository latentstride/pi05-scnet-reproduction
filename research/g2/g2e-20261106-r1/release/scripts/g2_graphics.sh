#!/usr/bin/env bash
#SBATCH --job-name=cpf-rw-graphics
#SBATCH --partition=ksagnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/robotwin-graphics-%j.out
set -euo pipefail
ulimit -c 0
root="$HOME/code_cpf/pi05"
image="$root/research/robotwin/robotwin-cu122"
a=${APPTAINER_BIN:-/public/software/apps/apptainer/1.3.4/bin/apptainer}
test -f "$image/opt/robotwin/logs/BUILD_COMPLETE"

# Run the image's userspace tool as a native host process. The explicit
# interpreter and library paths keep it independent of the host's glibc
# version and of vulkaninfo availability on the compute node.
host_vulkaninfo() {
 local sysroot="$image/opt/robotwin/sysroot"
 local arch=x86_64-linux-gnu
 local libs="$sysroot/usr/lib/$arch:$sysroot/lib/$arch:$image/usr/lib/$arch:$image/lib/$arch"
 if [[ -n ${host_icd_library:-} ]]; then
  libs="$libs:$(dirname "$host_icd_library")"
 fi
 timeout --kill-after=5 45 env -u DISPLAY -u WAYLAND_DISPLAY -u LD_PRELOAD -u LD_LIBRARY_PATH \
  "$image/lib/$arch/ld-linux-x86-64.so.2" --library-path "$libs" \
  "$sysroot/usr/bin/vulkaninfo" "$@"
}
if [[ ${1:-} == --host-tool-check ]]; then
 # This vulkaninfo release prints usage and exits with status 1 for --help.
 help_status=0
 help_output=$(host_vulkaninfo --help 2>&1) || help_status=$?
 printf '%s\n' "$help_output"
 [[ "$help_status" -le 1 && "$help_output" == *"USAGE:"* && "$help_output" == *"--summary"* ]]
 exit
fi
mode=${1:-}
[[ "$mode" == --g2 ]]
release=${2:?Expected fixed release directory}
g2_out=${3:?Expected graphics output path}
if [[ "$mode" == --g1 ]]; then
 release=${2:?Expected fixed release directory}
 preflight=${3:?Expected free preflight result directory}
 [[ "$release" == /* && "$preflight" == /* ]]
fi

cd "${SLURM_SUBMIT_DIR:?Submit from personal project root}"
test "$PWD" = "$root"
: "${SLURM_JOB_ID:?Must be an allocated Slurm GPU job}"
out="$root/research/robotwin/graphics-${SLURM_JOB_ID}"
if [[ "$mode" == --g1 ]]; then out="$root/results/pace-g1-${SLURM_JOB_ID}"; fi
out="$g2_out"
mkdir -p "$out"
failed_stage=allocation
trap 'code=$?; if ((code)); then printf "%s exit=%s\n" "$failed_stage" "$code" > "$out/FAILED"; fi' EXIT
stage() { failed_stage=$1; printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1"; }
run_check() {
 local name=$1 status=0
 shift
 stage "$name"
 "$@" || status=$?
 printf '%s\t%s\n' "$name" "$status" >> "$out/check_status.tsv"
 if ((status)); then printf '%s failed (exit %s)\n' "$name" "$status" >&2; fi
 return "$status"
}
: > "$out/check_status.tsv"

options=(--writable --nv --cleanenv --bind /public:/public --pwd "$root")
gpu_env=(PYTHONUNBUFFERED=1)
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then gpu_env+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"); fi
if [[ ${CUDA_DEVICE_ORDER+x} ]]; then gpu_env+=("CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"); fi
graphics_env=(DISPLAY=)
driver_libs=/opt/robotwin/sysroot/usr/lib/x86_64-linux-gnu:/opt/robotwin/sysroot/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/.singularity.d/libs
make_sim_command() {
 sim_cmd=("$a" exec "${options[@]}" "$image" env \
  PATH=/opt/robotwin/venv/bin:/opt/robotwin/sysroot/usr/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH="$driver_libs" PI05_ROOT="$root" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  XDG_CACHE_HOME=/opt/robotwin/cache/xdg TORCH_EXTENSIONS_DIR=/opt/robotwin/cache/torch_extensions \
  CUDA_CACHE_PATH=/opt/robotwin/cache/cuda TMPDIR=/opt/robotwin/tmp \
  PYTHONPYCACHEPREFIX=/opt/robotwin/cache/pycache \
  "${gpu_env[@]}" "${graphics_env[@]}")
}
inside() {
 local seconds=$1
 shift
 make_sim_command
 timeout --kill-after=5 "$seconds" "${sim_cmd[@]}" "$@"
}
stage nvidia_visibility
{
 printf 'SLURM_STEP_GPUS=%s\n' "${SLURM_STEP_GPUS:-}"
 printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-}"
 printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
} > "$out/gpu_visibility.txt"
# Query PCI and UUID from the same CUDA device. NVML is diagnostic only.
cuda_identity() {
 "$@" - <<'PY'
import ctypes as c
import uuid

cuda = c.CDLL("libcuda.so.1")


def check(operation, *args):
    code = getattr(cuda, operation)(*args)
    if code:
        raise RuntimeError(f"{operation} failed with CUDA status {code}")


check("cuInit", 0)
count = c.c_int()
check("cuDeviceGetCount", c.byref(count))
if count.value != 1:
    raise RuntimeError(f"Expected one visible CUDA GPU, found {count.value}")
device = c.c_int()
check("cuDeviceGet", c.byref(device), 0)
pci = c.create_string_buffer(32)
check("cuDeviceGetPCIBusId", pci, len(pci), device)
identifier = (c.c_ubyte * 16)()
check("cuDeviceGetUuid", c.byref(identifier), device)
print(f"GPU-{uuid.UUID(bytes=bytes(identifier))} {pci.value.decode('ascii')}")
PY
}
timeout --kill-after=5 10 nvidia-smi -L > "$out/nvidia-smi-list.txt" 2>&1 || true
if identity=$(cuda_identity timeout --kill-after=5 15 python3 2>> "$out/gpu_visibility.txt"); then
 source=host
elif identity=$(cuda_identity inside 20 python 2>> "$out/gpu_visibility.txt"); then
 source=container
else
 cat "$out/gpu_visibility.txt" "$out/nvidia-smi-list.txt" >&2
 exit 1
fi
read -r gpu_uuid gpu_pci <<< "$identity"
[[ "$gpu_uuid" =~ ^GPU-[[:xdigit:]-]{36}$ ]]
[[ "$gpu_pci" =~ ^[[:xdigit:]]{4,8}:[[:xdigit:]]{2}:[[:xdigit:]]{2}\.[0-7]$ ]]
for assigned in "${SLURM_STEP_GPUS:-}" "${SLURM_JOB_GPUS:-}" "${CUDA_VISIBLE_DEVICES:-}"; do
 if [[ "$assigned" == GPU-* && "${assigned,,}" != "${gpu_uuid,,}" ]]; then
  echo "CUDA device UUID disagrees with the scheduler-provided UUID" >&2
  exit 1
 fi
done
printf 'CUDA_SOURCE=%s\nCUDA_UUID=%s\nCUDA_PCI=%s\n' "$source" "$gpu_uuid" "$gpu_pci" >> "$out/gpu_visibility.txt"
printf '{"gpu_uuid":"%s","pci_bus_id":"%s","identity_source":"%s"}\n' \
 "$gpu_uuid" "$gpu_pci" "$source" > "$out/allocation.json"
gpu_env=(PYTHONUNBUFFERED=1 "CUDA_VISIBLE_DEVICES=$gpu_uuid" CUDA_DEVICE_ORDER=PCI_BUS_ID "ROBOTWIN_EXPECTED_PCI=$gpu_pci")

# CUDA planning does not require a Vulkan ICD. Capture its outcome first.
if [[ -z "$mode" ]]; then
 run_check curobo_plan inside 240 python scripts/robotwin_gpu_smoke.py --stage plan --output "$out" \
  > "$out/curobo.log" 2>&1 || true
fi

discover_graphics() {
# Only bind installed host NVIDIA libraries, read-only; never install a driver.
local probe="$release/scripts/robotwin_graphics_driver.py"
if [[ "$mode" == --g1 ]]; then probe="$release/robotwin_graphics_driver.py"; fi
timeout --kill-after=5 15 python3 "$probe" "$out" "$image"
}
check_vulkan_identity() {
python3 - "$out" <<'PY'
from pathlib import Path
import sys,re,json
p=Path(sys.argv[1]); allocation=json.loads((p/'allocation.json').read_text())
expected=allocation['gpu_uuid'][4:].lower().replace('-','')
seen=[]
for name in ['host','container']:
 text=(p/(name+'_vulkan.txt')).read_text(); uuids=set()
 for line in text.splitlines():
  if re.search(r'\bdeviceUUID\s*=',line):
   value=re.sub('0x','',line.split('=',1)[1],flags=re.I)
   value=re.sub('[^0-9a-f]','',value.lower())
   if len(value)==32: uuids.add(value)
 assert expected in uuids, name+': Vulkan does not enumerate the allocated NVIDIA UUID'
 seen.append(expected)
(p/'vulkan_identity.json').write_text(json.dumps({'pass':True,'allocated_uuid':allocation['gpu_uuid'],
 'host_and_container_same_allocated_gpu':seen[0]==seen[1]},indent=2))
PY
}
if run_check graphics_driver_discovery discover_graphics > "$out/graphics_driver_discovery.log" 2>&1; then
 host_icd_library=$(cat "$out/host_icd_library.txt")
 VK_DRIVER_FILES="$out/host_nvidia_icd.json" VK_ICD_FILENAMES="$out/host_nvidia_icd.json" \
  run_check host_vulkan host_vulkaninfo > "$out/host_vulkan.txt" 2>&1 || true
 driver_inside=$(cat "$out/driver_inside_path.txt")
 driver_libs="$driver_inside:$driver_libs"
 graphics_env=(DISPLAY= WAYLAND_DISPLAY= "VK_DRIVER_FILES=$out/container_nvidia_icd.json"
  "VK_ICD_FILENAMES=$out/container_nvidia_icd.json"
  "__EGL_VENDOR_LIBRARY_FILENAMES=$out/container_nvidia_egl.json")
 while IFS=$'\t' read -r driver_source destination; do
  options+=(--bind "$driver_source:$destination:ro")
 done < "$out/driver_binds.tsv"
 if ! run_check container_vulkan inside 45 env VK_LOADER_DEBUG=error,warn vulkaninfo \
  > "$out/container_vulkan.txt" 2>&1; then
  # Show unresolved dependencies separately from Vulkan instance failures.
  inside 10 ldd -r "$driver_inside/libEGL_nvidia.so.0" > "$out/container_icd_linker.log" 2>&1 || true
 fi
 run_check vulkan_identity check_vulkan_identity > "$out/vulkan_identity.log" 2>&1 || true
 test -f "$out/vulkan_identity.json"
 make_sim_command
 run_check sapien_render inside 110 python "$release/scripts/robotwin_gpu_smoke.py" --stage render --output "$out" > "$out/sapien.log" 2>&1
 python3 - "$out/sim-launch.json" "$gpu_uuid" "${sim_cmd[@]}" <<'JSON'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(gpu_uuid=sys.argv[2],sim_argv=sys.argv[3:]),indent=2)+'\n')
JSON
 exit
fi
exit 1
