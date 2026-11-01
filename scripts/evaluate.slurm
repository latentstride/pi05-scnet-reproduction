#!/usr/bin/env bash
#SBATCH --job-name=pi05-libero15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/evaluation-%j.out
#SBATCH --error=logs/evaluation-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
out=results/evaluation-$SLURM_JOB_ID
mkdir -p "$out"
port=$((20000 + SLURM_JOB_ID % 20000))
export PI05_GPU=1
bash scripts/run_container.sh /usr/bin/nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash scripts/run_container.sh .venv/bin/python scripts/serve_evaluation.py --port "$port" --output "$out" > "$out/policy_server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
ready=0
for ((i=0;i<180;i++)); do
  kill -0 "$server" 2>/dev/null || { tail -n 30 "$out/policy_server.log"; exit 1; }
  if grep -q POLICY_READY "$out/policy_server.log"; then ready=1; break; fi
  sleep 5
done
test "$ready" = 1
PI05_RENDER_BACKEND=osmesa bash scripts/run_container.sh .venv-libero/bin/python scripts/evaluate_libero.py --port "$port" --output "$out"
test -s "$out/summary.json"
cat "$out/summary.json"
