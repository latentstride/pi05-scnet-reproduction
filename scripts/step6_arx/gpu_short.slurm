#!/usr/bin/env bash
#SBATCH --job-name=cpf-step6-arx-short
#SBATCH --partition=ksagnormal01
#SBATCH --nodelist=gnode39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:RTX4090-PCIE-24GB-LS:1
#SBATCH --time=01:10:00
#SBATCH --signal=B:USR1@90
#SBATCH --no-requeue

set -euo pipefail
root=${1:?Expected canonical project root}
bundle=${2:?Expected fixed short-window bundle}
run=${3:?Expected completed Step 5 training run}
cd "$root"
export PYTHONPATH="$bundle/suite" PYTHONDONTWRITEBYTECODE=1
exec python3 -B -u -m scripts.step6_arx.gpu_short run "$root" "$run"
