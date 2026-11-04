#!/usr/bin/env bash
#SBATCH --job-name=cpf-pace-g1
#SBATCH --partition=ksagnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --no-requeue
#SBATCH --output=logs/pace-g1-%j.out
set -euo pipefail
release=${1:?Expected fixed release directory}
preflight=${2:?Expected successful G1 CPU preflight result directory}
[[ "$release" == /* && "$preflight" == /* ]]
root="$HOME/code_cpf/pi05"
cd "${SLURM_SUBMIT_DIR:?}"
test "$PWD" = "$root"
: "${SLURM_JOB_ID:?Requires an allocated job}"
out="$root/results/pace-g1-$SLURM_JOB_ID"
mkdir -p "$out"
record_failure() {
 rc=$?
 if [[ ! -f "$out/g1_acceptance.json" ]]; then
  python3 - "$out" "$SLURM_JOB_ID" "$release" "$rc" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, job, release, code = sys.argv[1:]
p = Path(out)
commit = Path(release) / 'RELEASE_COMMIT'
(p / 'g1_acceptance.json').write_text(json.dumps({
    'job_id': job, 'release_commit': commit.read_text().strip() if commit.is_file() else None,
    'passed': False, 'exit_code': int(code),
    'ended_at': datetime.now(timezone.utc).isoformat(),
    'stages': {name: {'status': 'skipped', 'reason': 'release/graphics precondition failed'}
               for name in ('G1.1', 'G1.2', 'G1.3')},
    'precondition_failure': (p / 'FAILED').read_text().strip() if (p / 'FAILED').is_file()
                            else 'release gate or allocation startup'
}, indent=2) + '\n')
PY
 fi
}
trap record_failure EXIT
bash "$release/run_pace_native_smoke.sh" --check-release-gate "$preflight/simulator"
expected=$(sha256sum "$release/SHA256SUMS" | cut -d' ' -f1)
[[ "$(cat "$preflight/PASSED_G1_RELEASE_SHA256")" == "$expected" ]] || exit 1
bash "$release/robotwin_graphics.slurm" --g1 "$release" "$preflight"
