#!/usr/bin/env bash
# CPU-only checks. Never submits, allocates or launches CUDA.
set -euo pipefail
scripts=$(cd "$(dirname "$0")" && pwd -P)
root=$(dirname "$scripts")
python=${G2_PYTHON:-python3}
client=${OPENPI_CLIENT_SRC:?Set this to the fixed OpenPI-PACE packages/openpi-client/src}
out=${1:?Expected a new output directory}
mkdir "$out"
out=$(cd "$out" && pwd -P)
export PYTHONPATH="$scripts:$root/tests:$client"
"$python" -m unittest discover -s "$root/tests" -v > "$out/tests.log" 2>&1
"$python" "$scripts/g2_workspace_probe.py" --output "$out" > "$out/workspace-probe.log"
"$python" "$scripts/g2_mock_acceptance.py" --output "$out/episodes"
"$python" - "$out" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
files = sorted(p for p in root.rglob("*") if p.is_file())
(root / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in files))
(root / "cpu-status.json").write_text(json.dumps(dict(
    gpu_used=False, passed=True, scope="CPU unit/geometry/synthetic episodes",
    gpu_release_ready=False, formal_run_lock_frozen=False), indent=2) + "\n")
PY
