#!/usr/bin/env bash
#SBATCH --job-name=cpf-step5-arx-lora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue

# Supply an authorized single-GPU partition/type and an absolute Slurm log path at submission.
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
printf 'job=%s cwd=%s release=%s visible=%s allocated=%s\n' \
    "$SLURM_JOB_ID" "$PWD" "$code" "${CUDA_VISIBLE_DEVICES:-cgroup}" "${SLURM_JOB_GPUS:-unknown}"
du -sh "$root/cache/pace-handover"
df -h "$(dirname "$run")"
stop=$(mktemp "$(dirname "$run")/.step5-stop-${SLURM_JOB_ID}.XXXXXX")
wait_signaled=0
flush_stop() {
    wait_signaled=1
    printf 'stop\n' > "$stop"
}
trap flush_stop USR1 TERM
trap 'if [[ -e "$stop" ]]; then unlink "$stop"; fi' EXIT
train_args=(--project-root "$root" --dataset "$dataset" --data-audit "$audit" --cpu-receipt "$receipt"
            --run "$run" --stop-file "$stop")
if [[ "$#" == 7 ]]; then train_args+=(--resume); fi
bash "$code/scripts/step5_arx/env.sh" gpu "$root" -m scripts.step5_arx.train fit "${train_args[@]}" &
child=$!
# A trapped signal can interrupt wait before the child finishes saving.
while true; do
    wait_signaled=0
    status=0
    wait "$child" || status=$?
    if ((status <= 128 || !wait_signaled)); then
        break
    fi
done
exit "$status"
