#!/usr/bin/env bash
#SBATCH --job-name=cpf-step5-arx-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue

# Supply an authorized >=48 GiB GPU partition/type and an absolute Slurm log path at submission.
# Arguments are project root, release root, converted H5, data audit, CPU receipt, run directory,
# and the optional literal --resume. Each path must be absolute.
set -euo pipefail
[[ "$#" == 6 || ( "$#" == 7 && "$7" == --resume ) ]]
root=$1
code=$2
dataset=$3
audit=$4
receipt=$5
run=$6
[[ -n "${SLURM_JOB_ID:-}" && "$root" == /* && "$root" == "$(cd "$root" && pwd -P)" && "$root" == */code_cpf/pi05 ]]
[[ "$code" == "$root/"* && "$code" == "$(cd "$code" && pwd -P)" ]]
for path in "$dataset" "$audit" "$receipt" "$run"; do
    [[ "$path" == "$root/"* && "$(realpath -m "$path")" == "$path" ]]
done
[[ -r "$code/scripts/step5_arx/env.sh" && -r "$dataset" && -r "$audit" && -r "$receipt" ]]
[[ -d "$(dirname "$run")" ]]
cd "$root"
printf 'job=%s cwd=%s release=%s device=%s\n' "$SLURM_JOB_ID" "$PWD" "$code" "${CUDA_VISIBLE_DEVICES:?}"
du -sh "$root/cache/pace-handover"
df -h "$(dirname "$run")"
resume=()
[[ "$#" == 6 ]] || resume=(--resume)
stop=$(mktemp "$(dirname "$run")/.step5-stop-${SLURM_JOB_ID}.XXXXXX")
flush_stop() { printf 'stop\n' > "$stop"; }
trap flush_stop USR1 TERM
trap 'unlink "$stop"' EXIT
bash "$code/scripts/step5_arx/env.sh" gpu "$root" -m scripts.step5_arx.train fit \
    --project-root "$root" --dataset "$dataset" --data-audit "$audit" --cpu-receipt "$receipt" \
    --run "$run" --stop-file "$stop" "${resume[@]}" &
child=$!
# A signal interrupts wait even while Python is flushing Orbax. Reap the child afterwards.
while true; do
    status=0
    wait "$child" || status=$?
    if ((status <= 128)) || ! kill -0 "$child" 2>/dev/null; then
        break
    fi
done
exit "$status"
