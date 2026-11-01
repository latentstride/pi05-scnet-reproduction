#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
appt=/public/software/apps/apptainer/1.3.4/bin/apptainer
image=$root/containers/cuda12-ubuntu22.sif
test -s "$image"
"$appt" exec --cleanenv --bind /public:/public "$image" cat /etc/os-release
sha=$(cat source/OPENPI_COMMIT)
cd "source/openpi-$sha"
"$appt" exec --cleanenv --bind /public:/public "$image" env \
  UV_CACHE_DIR="$root/cache/uv" UV_PROJECT_ENVIRONMENT="$root/.venv" \
  GIT_LFS_SKIP_SMUDGE=1 "$root/tools/uv/bin/uv" sync --frozen --python 3.11
"$appt" exec --cleanenv --bind /public:/public "$image" "$root/.venv/bin/python" \
  -c 'from openpi.training import config; from openpi.policies import policy_config; print(config.get_config("pi05_libero").name)'
