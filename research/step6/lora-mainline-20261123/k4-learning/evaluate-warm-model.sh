#!/usr/bin/env bash
set -euo pipefail
umask 077
release_home=$(cd "$(dirname "$0")" && pwd -P)
root=$(cd "$release_home/../.." && pwd -P)
cd "$root"
release="$release_home/g2-standalone-arx"
scripts="$release/scripts"
sim="$root/tools/a800-closed-loop"
graphics="$sim/nvidia-580.95.05"
pace="$sim/PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
openpi="$root/source/Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae"
python="$root/tools/pace-model-venv/bin/python"
experiment=${1:?Expected completed learning comparison}
out="$root/results/step6-k4-learning-pure-$(date -u +%Y%m%dT%H%M%SZ)"
model="$out/model"
mkdir "$out"
echo "RESULT=$out"

export G2_SCNET_NOTEBOOK_INSTANCE=2123408832426430466
export PATH="$sim/bin:$root/tools/pace-git/bin:$PATH"
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$openpi"
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1="$pace"
export OPENPI_DATA_HOME="$root/cache/openpi-pace" XDG_CACHE_HOME="$root/cache/notebook"
export HF_HOME="$root/cache/notebook/hf" JAX_COMPILATION_CACHE_DIR="$root/cache/notebook/jax"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_RUNTIME_DIR="$sim/xdg" DOT_SAPIEN_CACHE="$sim/sapien-cache"
export VK_ICD_FILENAMES="$graphics/nvidia-egl.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$graphics/unpacked/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export LD_LIBRARY_PATH="$graphics/unpacked/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

client="$openpi/packages/openpi-client/src"
prefix="$root/tools/pace-smoke-deps:$openpi/src:$openpi/pace/src:$client:$root/cache/pace-prepare/lerobot"
training="$root/tools/step5-execution-aligned"
identity=$("$python" - "$experiment" "$out" "$sim/suite" "$training/initial-references" <<'PY'
import json,shutil,sys
from pathlib import Path
experiment,out,suite,refs=map(Path,sys.argv[1:])
service=json.loads((experiment/'warm-service.json').read_text())
plan=json.loads((suite/'docs/step6/pairs.json').read_text())
seeds=[1,0]
(out/'campaign.json').write_text(json.dumps(dict(plan=plan,seeds=seeds,fixture=False))+'\n')
(out/'candidate.json').write_text(json.dumps(service,indent=2)+'\n')
for seed in seeds:
 label=f'seed-{seed:02d}-frozen'
 (out/label).mkdir()
 shutil.copyfile(refs/f'{label}.json',out/label/'receipt.json')
print(service['metadata']['step6_identity']['training_identity_sha256'])
PY
)
env JAX_PLATFORMS=cpu PYTHONPATH="$sim/deps:$sim/suite:$sim/pr5/scripts:$training:$prefix" \
 "$python" -B -u "$root/tools/shandong/k4-learning/evaluate_pure.py" --pr5-root "$sim/pr5" --pace-root "$pace" \
 --asset-audit "$sim/pr5/research/g2/g2e-20261106-r1/release/asset-audit.json" \
 --output "$out" --seed 1 --seed 0 --variant lora --expected-training-identity "$identity" \
 --lora-url ws://127.0.0.1:51923
