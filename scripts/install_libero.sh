#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tools/uv/bin/uv python install 3.8
mkdir -p config/libero
s=$PWD/source/openpi-$(cat source/OPENPI_COMMIT)/third_party/libero/libero/libero
printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s/../datasets\nassets: %s/assets\n' \
  "$s" "$s" "$s" "$s" "$s" > config/libero/config.yaml
bash scripts/run_container.sh bash "$PWD/scripts/install_libero_inside.sh"
