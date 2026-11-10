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
umask 077
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
root="${SLURM_SUBMIT_DIR:?requires its own Slurm allocation}"
release="${G2_FINAL_RELEASE:?candidate path required}"
exec python3 "$release/scripts/g2_parallel_deploy.py" "$release" \
  --root "$root" --authorization "${G2_FINAL_APPROVAL:?job-bound approval required}"
