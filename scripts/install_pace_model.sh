#!/usr/bin/env bash
# Run from the SCNet project root inside containers/cuda12-ubuntu22.sif.
set -euo pipefail

root=$PWD
revision=f4e52e6de4cc487a11585a3a7414d610b2883dae
src=$root/source/Openpi-PACE-$revision
stage=$root/cache/pace-prepare
venv=$root/tools/pace-model-venv
uv=$root/tools/uv/bin/uv
python=$root/tools/python/cpython-3.11-linux-x86_64-gnu/bin/python3
export UV_CACHE_DIR="$root/cache/uv" UV_PYTHON_INSTALL_DIR="$root/tools/python"
export OPENPI_DATA_HOME="$root/cache/openpi-pace"
export XDG_CACHE_HOME="$OPENPI_DATA_HOME/xdg" HF_HOME="$OPENPI_DATA_HOME/hf"
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH="$root/sysroot/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

mkdir -p "$stage/lerobot" "$root/logs/pace-prepare"
cd "$src"
echo '5e3a9a0a12d9a6048afea5591f4520c98585499cbd4a8343dcabfe2aaed94e3d  uv.lock' | sha256sum -c -
for entry in aloha:d1dc83afd89ded4379851257fe5d85632d31d5ec libero:f78abd68ee283de9f9be3c8f7e2a9ad60246e95c; do
  name=${entry%:*}
  mkdir -p "third_party/$name"
  tar -xzf "$root/source/${entry/:/-}.tar.gz" --strip-components=1 -C "third_party/$name"
done
tar -xzf "$root/source/lerobot.tar.gz" --strip-components=1 -C "$stage/lerobot"

requirements=$stage/requirements.locked.txt
"$uv" export --frozen --no-dev --no-emit-workspace --no-emit-package lerobot \
  --output-file "$requirements" >/dev/null
# Select the CPython 3.11 Linux wheel already recorded in the lock.
"$python" - "$requirements" <<'PY'
from pathlib import Path
import re
import sys
import tomllib

lock = tomllib.loads(Path("uv.lock").read_text())
av = next(package for package in lock["package"] if package["name"] == "av")
wheel = next(
    wheel for wheel in av["wheels"]
    if wheel["url"].endswith("-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
)
path = Path(sys.argv[1])
text, count = re.subn(
    rf"^av=={re.escape(av['version'])}[^\n]*(?:\n    --hash=[^\n]*)*",
    f"av @ {wheel['url']} --hash={wheel['hash']}",
    path.read_text(),
    flags=re.M,
)
if count != 1:
    raise ValueError("expected one PyAV requirement in the lock export")

# The pinned Gemma module imports pytest when loading the policy configuration.
pytest_packages = ("iniconfig", "pluggy", "pytest")
extra_requirements = []
for name in pytest_packages:
    package = next(package for package in lock["package"] if package["name"] == name)
    wheel = next(wheel for wheel in package["wheels"] if wheel["url"].endswith("-py3-none-any.whl"))
    extra_requirements.append(f"{name} @ {wheel['url']} --hash={wheel['hash']}")
path.write_text(text.rstrip() + "\n" + "\n".join(extra_requirements) + "\n")
PY

test -x "$venv/bin/python" || "$uv" venv --python "$python" "$venv"
py=$venv/bin/python
"$uv" pip install --python "$py" --require-hashes --no-deps -r "$requirements"
"$uv" pip install --python "$py" --no-deps -e "$stage/lerobot" -e "$src/packages/openpi-client" -e "$src"
"$uv" pip check --python "$py"
"$py" - <<'PY'
import jax
import orbax.checkpoint
import sentencepiece
from openpi.policies import policy_config
from openpi.training import config

task = config.get_config("pi05_robotwin_handover_block")
assert task.model.action_horizon == 50
assert jax.devices()[0].platform == "cpu"
print(task.name, jax.__version__, jax.devices())
PY
"$uv" pip freeze --python "$py" >"$root/logs/pace-prepare/model-packages.txt"
echo PACE_MODEL_ENV_VERIFIED
