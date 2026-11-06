#!/usr/bin/env bash
#SBATCH --job-name=cpf-pace-g2
#SBATCH --partition=ksagnormal01
#SBATCH --nodelist=gnode39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/pace-g2-%j.out
set -euo pipefail

# This entry point runs only inside an allocation. It never submits jobs.
root="$HOME/code_cpf/pi05"
if [[ "${SLURM_JOB_NAME:-}" != cpf-pace-g2 || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "Expected an existing cpf-pace-g2 allocation with a numeric job ID" >&2
    exit 1
fi
out="$root/results/pace-g2-$SLURM_JOB_ID"
mkdir "$out"
stage=entry
record() {
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$stage" "$1" | tee -a "$out/entry-status.tsv"
}
finish() {
    local rc=$?
    if ((rc)); then
        printf '%s\n' "$rc" > "$out/entry-exit-code"
        record "failed exit=$rc" >&2
    fi
}
trap finish EXIT

record start
release=${1:?Expected immutable G2 release directory}
manifest_sha=${2:?Expected reviewed release manifest SHA-256}
if [[ "$release" != "$root/"* || ! "$manifest_sha" =~ ^[0-9a-f]{64}$ ||
      "${SLURM_SUBMIT_DIR:-}" != "$root" ]]; then
    echo "Invalid release path, manifest digest or Slurm submission directory" >&2
    exit 1
fi
stage=release_manifest
record start
cd "$release"
if [[ "$(sha256sum SHA256SUMS | cut -d' ' -f1)" != "$manifest_sha" ]]; then
    echo "Release manifest SHA-256 differs from the reviewed digest" >&2
    exit 1
fi
for item in release-admission.json run_lock.json scripts/run-admitted-g2-plan.sh scripts/g2_host_supervisor.py scripts/g2_single_launch.py scripts/g2_job_admission.py scripts/g2_graphics.sh launch-spec.json; do
    awk -v target="$item" '$2 == target {found=1} END {exit !found}' SHA256SUMS
done
sha256sum --check --strict SHA256SUMS
stage=allocation
record start
python3 "$release/scripts/g2_job_admission.py" "$release" "$out" "$SLURM_JOB_ID" 2>&1 | tee "$out/allocation-admission.log"
stage=graphics
record start
timeout --kill-after=5 "$(cat "$out/graphics-timeout-s")" bash "$release/scripts/g2_graphics.sh" --g2 "$release" "$out/graphics"
stage=launch_plan
record start
python3 "$release/scripts/g2_single_launch.py" "$root" "$release" "$out" "$out/graphics/sim-launch.json" "$SLURM_JOB_ID" "$(cat "$out/allocation-end-epoch")"
stage=supervisor
record start
bash "$release/scripts/run-admitted-g2-plan.sh" "$out/effective-launch-plan.json" "$out/supervisor"
test -s "$out/supervisor/g2_acceptance.json"
record complete
