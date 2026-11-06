#!/usr/bin/env bash
# Run only inside the admitted allocation. This host supervisor launches the existing simulator/model containers.
set -euo pipefail
: "${SLURM_JOB_ID:?An existing allocation is required}"
[[ "${SLURM_JOB_NAME:-}" == cpf-pace-g2 ]]
plan=${1:?Expected effective launch plan}
output=${2:?Expected unused supervisor output directory}
root="$HOME/code_cpf/pi05"
release=$(cd "$(dirname "$0")/.." && pwd -P)
client="$root/source/Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae/packages/openpi-client/src"
export PYTHONPATH="$release/scripts:$root/tools/pace-smoke-deps:$client"
python3 "$release/scripts/g2_host_supervisor.py" "$plan" "$output"
python3 - "$output/g2_acceptance.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
if r.get('complete') is not True or r.get('model_server_starts') != 1:
    raise SystemExit('required stage missing, or shared model did not start exactly once')
PY
