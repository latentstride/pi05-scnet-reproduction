#!/usr/bin/env bash
set -euo pipefail
pace=${1:?}
reference=${2:?}
out=${3:?}
uri=${4:?}
scripts=$(cd "$(dirname "$0")" && pwd -P)
python -u "$scripts/pace_client_smoke.py" --pace-root "$pace" --reference-run "$reference" \
 --backend checkpoint --timeout 120 --uri "$uri" --output "$out"
python "$scripts/verify_pace_smoke.py" "$out" --native --backend checkpoint \
 > "$out/artifact-verification.json"
