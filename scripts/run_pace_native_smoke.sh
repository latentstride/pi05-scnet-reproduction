#!/usr/bin/env bash
# Invoke inside the RoboTwin container with graphics already configured.
set -euo pipefail
: "${PI05_ROOT:?}" "${SLURM_JOB_ID:?Run under an authorized GPU allocation}"
out=${1:?Expected an existing output directory}
pace="$PI05_ROOT/source/PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
openpi="$PI05_ROOT/source/Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae"
scripts="$PI05_ROOT/scripts"
export PYTHONPATH="$PI05_ROOT/tools/pace-smoke-deps:$openpi/packages/openpi-client/src"
port=$((20000 + SLURM_JOB_ID % 30000))
python -u "$scripts/pace_model_smoke.py" --port "$port" > "$out/fake-policy.log" 2>&1 &
server=$!
cleanup() {
 kill "$server" 2>/dev/null || true
 wait "$server" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((attempt=0; attempt<100; attempt++)); do
 if grep -Fxq "fake policy on ws://127.0.0.1:$port" "$out/fake-policy.log"; then break; fi
 kill -0 "$server"
 sleep 0.1
done
grep -Fxq "fake policy on ws://127.0.0.1:$port" "$out/fake-policy.log"
python -u "$scripts/pace_client_smoke.py" --pace-root "$pace" --uri "ws://127.0.0.1:$port" \
 --output "$out/rollout"
python "$scripts/verify_pace_smoke.py" "$out/rollout" --native > "$out/artifact-verification.json"
