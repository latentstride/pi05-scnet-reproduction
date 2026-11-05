#!/usr/bin/env bash
# Free checks in the same isolated runtimes used by the one G1 allocation.
set -euo pipefail
root=${PI05_ROOT:-$PWD}
export PI05_ROOT="$root"
release=$(cd "$(dirname "$0")" && pwd -P)
out=${1:?Expected new absolute output directory}
[[ "$out" == /* ]]
mkdir "$out"
mkdir "$out/simulator"
revision=f4e52e6de4cc487a11585a3a7414d610b2883dae
(cd "$release" && sha256sum -c SHA256SUMS) > "$out/release-integrity.log" 2>&1
set +e
python3 -m unittest discover -s "$release" -p 'test_*.py' -v > "$out/host-tests.log" 2>&1
host_rc=$?
timeout -k 10 720 bash "$root/scripts/run_robotwin_env.sh" bash "$release/run_pace_native_smoke.sh" \
 --cpu-preflight "$out/simulator" > "$out/simulator.log" 2>&1 &
sim_pid=$!
timeout -k 10 1200 bash "$release/pace_g1_model_env.sh" cpu -X importtime "$release/pace_model_smoke.py" \
 --preflight-only --openpi-root "$root/source/Openpi-PACE-$revision" \
 --download-dir "$root/cache/pace-handover" --diagnostics-dir "$out/model" > "$out/model.log" 2>&1 &
model_pid=$!
trap 'kill "$sim_pid" "$model_pid" 2>/dev/null || true' INT TERM
wait "$sim_pid"
sim_rc=$?
wait "$model_pid"
model_rc=$?
set -e
printf 'host_tests\t%s\nsimulator\t%s\nmodel\t%s\n' "$host_rc" "$sim_rc" "$model_rc" > "$out/exit-codes.tsv"
if ((host_rc || sim_rc || model_rc)); then exit 1; fi
bash "$release/run_pace_native_smoke.sh" --check-release-gate "$out/simulator"
sha256sum "$release/SHA256SUMS" | cut -d' ' -f1 > "$out/PASSED_G1_RELEASE_SHA256.tmp"
mv "$out/PASSED_G1_RELEASE_SHA256.tmp" "$out/PASSED_G1_RELEASE_SHA256"
