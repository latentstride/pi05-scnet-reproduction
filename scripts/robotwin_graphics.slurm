#!/usr/bin/env bash
#SBATCH --job-name=cpf-rw-graphics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/robotwin-graphics-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from personal project root}"
test "$PWD" = "$HOME/code_cpf/pi05"
: "${SLURM_JOB_ID:?Must be an allocated Slurm GPU job}"
: "${ROBOTWIN_IMAGE:?Set isolated research image path}"
: "${ROBOTWIN_PYTHON:?Set Python path INSIDE that image}"
: "${APPTAINER_BIN:=/public/software/apps/apptainer/1.3.4/bin/apptainer}"
case "$ROBOTWIN_IMAGE" in "$PWD"/research/robotwin/*) ;; *) echo 'Research image must be inside personal research directory'; exit 2;; esac
test -e "$ROBOTWIN_IMAGE"
out="$PWD/research/robotwin/graphics-${SLURM_JOB_ID}"
mkdir -p "$out" "$PWD/cache/robotwin/torch_extensions" "$PWD/cache/robotwin/xdg"
export TORCH_EXTENSIONS_DIR="$PWD/cache/robotwin/torch_extensions"
export XDG_CACHE_HOME="$PWD/cache/robotwin/xdg"
export CUDA_CACHE_PATH="$PWD/cache/robotwin/cuda"
options=(--nv --cleanenv --bind "$PWD:$PWD" --pwd "$PWD")
# Explicit optional ICD bind directory: prepared ahead of time, not administrator changes.
if [[ -n ${ROBOTWIN_ICD_DIR:-} ]]; then
  test -d "$ROBOTWIN_ICD_DIR"
  options+=(--bind "$ROBOTWIN_ICD_DIR:/etc/vulkan/icd.d:ro")
fi
inside() {
  local seconds=$1
  shift
  timeout "$seconds" "$APPTAINER_BIN" exec "${options[@]}" "$ROBOTWIN_IMAGE" env \
    TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    CUDA_CACHE_PATH="$CUDA_CACHE_PATH" "$@"
}
failed_stage=initialization
trap 'code=$?; if ((code)); then printf "%s\n" "$failed_stage exit=$code" > "$out/FAILED"; fi' EXIT
failed_stage=nvidia_visibility
timeout 20 nvidia-smi --query-gpu=uuid,name --format=csv,noheader > "$out/host_nvidia.csv"
failed_stage=host_vulkan
timeout 45 vulkaninfo > "$out/host_vulkan.txt" 2>&1
failed_stage=container_vulkan
inside 45 vulkaninfo > "$out/container_vulkan.txt" 2>&1
failed_stage=vulkan_identity
python3 - "$out" <<'PY'
from pathlib import Path
import sys,re,json
p=Path(sys.argv[1])
expected={line.split(',')[0].strip().lower()[4:].replace('-','')
          for line in (p/'host_nvidia.csv').read_text().splitlines()}
seen=[]
for name in ['host','container']:
 text=(p/f'{name}_vulkan.txt').read_text()
 uuids={x.lower().replace('-','') for x in re.findall(r'deviceUUID\s*=\s*([0-9a-fA-F-]+)',text)}
 matches=expected & uuids
 assert matches, f'{name}: no Vulkan device UUID matching NVIDIA management UUID; inspect raw log'
 seen.append(matches)
assert seen[0] & seen[1], 'Host/container do not expose the same allocated NVIDIA GPU'
(p/'vulkan_identity.json').write_text(json.dumps({'pass':True,'matching_uuids':sorted(seen[0]&seen[1])},indent=2))
PY
failed_stage=sapien_render
inside 150 "$ROBOTWIN_PYTHON" scripts/robotwin_gpu_smoke.py --stage render --output "$out" > "$out/sapien.log" 2>&1
failed_stage=curobo_plan
inside 240 "$ROBOTWIN_PYTHON" scripts/robotwin_gpu_smoke.py --stage plan --output "$out" > "$out/curobo.log" 2>&1
failed_stage=summary
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); result={k:json.loads((p/f'{k}.json').read_text()) for k in ['vulkan_identity','render','plan']}
assert all(x['pass'] for x in result.values())
(p/'acceptance.json').write_text(json.dumps({'passed':True,'job_id':p.name.split('-')[-1],'checks':result},indent=2))
print('ROBOTWIN_GRAPHICS_GATE_PASS')
PY
