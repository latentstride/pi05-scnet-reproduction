#!/usr/bin/env bash
#SBATCH --job-name=pi05-inference
#SBATCH --partition=ksagnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the pi05 root}"
date -Is
hostname
nvidia-smi
export PI05_GPU=1
bash scripts/run_container.sh .venv/bin/python -c 'import jax,torch; print("JAX",jax.devices(),"TORCH",torch.cuda.get_device_name(),flush=True); assert any(d.platform=="gpu" for d in jax.devices())'
bash scripts/run_container.sh .venv-libero/bin/python scripts/capture_libero.py
bash scripts/run_container.sh .venv/bin/python inference_test.py \
  --checkpoint cache/checkpoints/pi05_libero \
  --observation results/observation/libero_observation.npz \
  --output results/inference
date -Is
