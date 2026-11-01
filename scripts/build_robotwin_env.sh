#!/usr/bin/env bash
# CPU-only independent sandbox builder. No sbatch, no policy weights.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
test "$root" = "$HOME/code_cpf/pi05"
a=${APPTAINER_BIN:-/public/software/apps/apptainer/1.3.4/bin/apptainer}
b="$root/research/robotwin/env-build"
image="$root/research/robotwin/robotwin-cu122"
base="$root/containers/cuda12-devel-ubuntu22.sif"
mkdir -p "$b"/{logs,cache/apptainer,tmp}
export APPTAINER_CACHEDIR="$b/cache/apptainer" APPTAINER_TMPDIR="$b/tmp"
if ! test -d "$image/.singularity.d"; then
 "$a" build --sandbox "$image" "$base" > "$b/logs/sandbox-build.log" 2>&1
fi
mkdir -p "$image/public" "$image/opt/robotwin"
sha256sum "$base" > "$b/base-image.sha256"
"$a" exec --writable --cleanenv --bind /public:/public "$image" env PI05_ROOT="$root" \
 bash "$root/scripts/robotwin_env_inside.sh" 2>&1 | tee "$b/logs/install.log"
