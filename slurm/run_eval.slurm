#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 8:00:00
#SBATCH -A cis260094p
#SBATCH -J eeg_eval
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/classification_cnn_baseline.yaml}"
CHECKPOINT_PATH="${2:-}"

cd /jet/home/jye10/Guided-Project
mkdir -p logs

module purge
module load anaconda3
conda deactivate || true
conda activate /ocean/projects/cis260094p/jye10/shared-envs/IDLS26
export PYTHONNOUSERSITE=1

echo "========================================"
echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Config path: ${CONFIG_PATH}"
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint path: ${CHECKPOINT_PATH}"
fi
echo "========================================"

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  python -m scripts.eval_classification \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}"
else
  python -m scripts.eval_classification \
    --config "${CONFIG_PATH}"
fi

echo "========================================"
echo "Job finished on $(date)"
echo "========================================"
