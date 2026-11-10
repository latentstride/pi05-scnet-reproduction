#!/usr/bin/env bash
#SBATCH --job-name=cpf-pace-g2
#SBATCH --partition=ksagnormal01
#SBATCH --nodelist=gnode39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --time-min=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
set -euo pipefail
phase=slurm_shell
trap 'rc=$?; if (( rc != 0 )); then printf "G2_STARTUP stage=%s outcome=INFRASTRUCTURE_FAILURE exit=%d job=%s\n" "$phase" "$rc" "${SLURM_JOB_ID:-unknown}" >&2; fi' EXIT
umask 077
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
printf 'G2_STARTUP stage=slurm_shell job=%s host=%s\n' "${SLURM_JOB_ID:-unknown}" "$(hostname)"
for required in SLURM_JOB_ID G2_PROJECT_ROOT G2_FINAL_RELEASE G2_FINAL_APPROVAL; do
 if [[ -z ${!required:-} ]]; then
  printf 'Missing required environment variable %s\n' "$required" >&2
  exit 1
 fi
done
root="$G2_PROJECT_ROOT"
if [[ "$root" != "$(pwd -P)" ]]; then
 printf 'Batch working directory differs from G2_PROJECT_ROOT\n' >&2
 exit 1
fi
printf 'G2_STARTUP stage=project_root_checked cwd=%s submit_dir=%s\n' "$root" "${SLURM_SUBMIT_DIR:-unknown}"
release="$G2_FINAL_RELEASE"
approval="$G2_FINAL_APPROVAL"
phase=python_preflight
test -r "$release/scripts/g2_parallel_bootstrap.py"
test -r "$release/scripts/g2_parallel_deploy.py"
test -r "$approval"
phase=python_entry
args=("$release" --root "$root" --authorization "$approval")
if [[ ${G2_STARTUP_PROBE_BATCH+x} ]]; then args+=(--probe-batch "$G2_STARTUP_PROBE_BATCH"); fi
python3 -u "$release/scripts/g2_parallel_bootstrap.py" "$release/scripts/g2_parallel_deploy.py" "${args[@]}" &
child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; exit 143' TERM INT
wait "$child"
