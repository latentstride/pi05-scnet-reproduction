#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=215abfb217dbac7d5f1273282331b9b1866c0479
mkdir -p source logs
curl -fL --retry 2 --connect-timeout 10 --max-time 120 \
  "https://codeload.github.com/Physical-Intelligence/openpi/tar.gz/$sha" -o openpi-source.tar.gz
tar -xzf openpi-source.tar.gz -C source
printf '%s\n' "$sha" > source/OPENPI_COMMIT
sha256sum openpi-source.tar.gz > source/SHA256SUMS
curl -fL --retry 2 --connect-timeout 10 --max-time 60 \
  "https://api.github.com/repos/Physical-Intelligence/openpi/git/trees/$sha?recursive=1" \
  -o logs/source_tree.json
