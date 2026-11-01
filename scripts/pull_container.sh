#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p containers cache/apptainer logs
export APPTAINER_CACHEDIR=$PWD/cache/apptainer
appt=/public/software/apps/apptainer/1.3.4/bin/apptainer
image=containers/cuda12-ubuntu22.sif
if test -e "$image"; then
  echo 'Image already exists; verify it before reusing.'
else
  "$appt" pull "$image" docker://nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
fi
"$appt" exec --cleanenv --bind /public:/public "$image" cat /etc/os-release
sha256sum "$image" > containers/SHA256SUMS
