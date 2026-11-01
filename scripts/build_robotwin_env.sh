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
# Use the login host's working network route for official source archives.
# No code is installed into the host interpreter.
sources="$image/opt/robotwin/src"
mkdir -p "$sources"
prefetch() {
 local repo=$1 commit=$2 name=$3
 if ! test -s "$sources/$name/setup.py" && ! test -s "$sources/$name/script/requirements.txt"; then
  curl -fL --retry 2 --connect-timeout 20 --max-time 240 \
   "https://codeload.github.com/$repo/tar.gz/$commit" -o "$sources/$name-pinned.tar.gz"
  mkdir -p "$sources/$name"
  tar -xzf "$sources/$name-pinned.tar.gz" --strip-components=1 -C "$sources/$name"
 fi
}
prefetch NVlabs/curobo d64c4b005459db10c5dd867d8b30a87d5bda9bdb curobo
prefetch facebookresearch/pytorch3d 75ebeeaea0908c5527e7b1e305fbc7681382db47 pytorch3d
prefetch RoboTwin-Platform/RoboTwin 13c3c47ff4312dd62484bcd51be034af55c062d1 RoboTwin
"$a" exec --writable --cleanenv --bind /public:/public "$image" env PI05_ROOT="$root" \
 bash "$root/scripts/robotwin_env_inside.sh" 2>&1 | tee "$b/logs/install.log"
