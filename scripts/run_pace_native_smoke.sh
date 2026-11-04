#!/usr/bin/env bash
# Invoke the release copy inside the RoboTwin container.
set -euo pipefail
mode=${1:?Use --cpu-preflight, --check-release-gate or --native}
out=${2:?Expected an existing output directory}
scripts=$(cd "$(dirname "$0")" && pwd -P)
if ! (cd "$scripts" && sha256sum -c SHA256SUMS); then
 echo "PACE release checksum verification failed" >&2
 exit 1
fi
release_hash=$(sha256sum "$scripts/SHA256SUMS" | cut -d' ' -f1)
case "$mode" in
 --check-release-gate|--native)
  gate=$out
  if [[ "$mode" == --native ]]; then gate=${3:?Expected CPU preflight result directory}; fi
  if ! read -r passed_hash < "$gate/PASSED_RELEASE_SHA256" || [[ "$passed_hash" != "$release_hash" ]]; then
   echo "CPU preflight did not pass for this release" >&2
   exit 1
  fi
  if [[ "$mode" == --check-release-gate ]]; then exit 0; fi
  ;;
 --cpu-preflight)
  if [[ -e "$out/PASSED_RELEASE_SHA256" ]]; then
   echo "Expected a fresh CPU preflight result directory" >&2
   exit 1
  fi
  ;;
 *) exit 2 ;;
esac
: "${PI05_ROOT:?}"
pace="$PI05_ROOT/source/PACE-fd2e624cc19325cc9b20a620c82d51c0d8e25cc1"
openpi="$PI05_ROOT/source/Openpi-PACE-f4e52e6de4cc487a11585a3a7414d610b2883dae"
export PYTHONPATH="$PI05_ROOT/tools/pace-smoke-deps:$openpi/packages/openpi-client/src"
export MPLBACKEND=Agg
cp "$scripts/RELEASE_COMMIT" "$out/RELEASE_COMMIT"
printf '%s\n' "$release_hash" > "$out/release-manifest.sha256"
stage=launch
server=
mark() {
 stage=$1
 printf '%s\t%s\tstart\n' "$(date -u +%FT%TZ)" "$stage" >> "$out/runner-phases.tsv"
}
done_stage() {
 printf '%s\t%s\tend\n' "$(date -u +%FT%TZ)" "$stage" >> "$out/runner-phases.tsv"
}
cleanup() {
 status=$?
 if [[ -n "$server" ]]; then
  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
 fi
 if ((status)); then
  printf '%s\t%s\terror=%s\n' "$(date -u +%FT%TZ)" "$stage" "$status" >> "$out/runner-phases.tsv"
 fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$mode" == --cpu-preflight ]]; then
 # CPython emits per-import timings independently of the phase watchdog.
 mark native_cpu_imports
 python -X importtime -u "$scripts/pace_client_smoke.py" --preflight-only \
  --pace-root "$pace" --output "$out/native-preflight" 2> "$out/importtime.log"
 done_stage
fi
port=$((20000 + ${SLURM_JOB_ID:-$$} % 30000))
if [[ "$mode" == --native ]]; then : "${SLURM_JOB_ID:?Expected a GPU allocation}"; fi
mark fake_server_start
python -u "$scripts/pace_model_smoke.py" --port "$port" > "$out/fake-policy.log" 2>&1 &
server=$!
for ((attempt=0; attempt<400; attempt++)); do
 if grep -Fxq "fake policy on ws://127.0.0.1:$port" "$out/fake-policy.log"; then break; fi
 kill -0 "$server"
 sleep 0.1
done
grep -Fxq "fake policy on ws://127.0.0.1:$port" "$out/fake-policy.log"
done_stage
native_args=()
if [[ "$mode" == --native ]]; then native_args+=(--pace-root "$pace"); fi
mark rollout
python -u "$scripts/pace_client_smoke.py" "${native_args[@]}" --uri "ws://127.0.0.1:$port" \
 --output "$out/rollout"
done_stage
verify_args=()
if [[ "$mode" == --native ]]; then verify_args+=(--native); fi
mark artifact_verification
python "$scripts/verify_pace_smoke.py" "$out/rollout" "${verify_args[@]}" > "$out/artifact-verification.json"
done_stage
if [[ "$mode" == --cpu-preflight ]]; then
 printf '%s\n' "$release_hash" > "$out/PASSED_RELEASE_SHA256.tmp"
 mv "$out/PASSED_RELEASE_SHA256.tmp" "$out/PASSED_RELEASE_SHA256"
fi
