#!/usr/bin/env bash
# Stage only the files from a Git commit required by the native Fake Policy smoke.
set -euo pipefail
repo=${1:?Expected repository path}
ref=${2:?Expected Git commit}
out=${3:?Expected new release directory}
commit=$(git -C "$repo" rev-parse --verify "$ref^{commit}")
files=(pace_client_smoke.py pace_diagnostics.py pace_model_smoke.py pace_runtime.py
 verify_pace_smoke.py run_pace_native_smoke.sh robotwin_graphics.slurm pace_download.py
 pace_g1.slurm pace_g1_orchestrate.py pace_g1_plan.py pace_g1_model_env.sh
 pace_g1_preflight.sh pace_g1_query.py pace_g1_rollout.sh)
mkdir "$out"
for file in "${files[@]}"; do
 git -C "$repo" show "$commit:scripts/$file" > "$out/$file"
done
printf '%s\n' "$commit" > "$out/RELEASE_COMMIT"
git -C "$repo" show "$commit:pace_download_manifest.json" > "$out/pace_download_manifest.json"
(cd "$out" && sha256sum RELEASE_COMMIT "${files[@]}" pace_download_manifest.json > SHA256SUMS)
chmod a-w "$out" "$out"/*
