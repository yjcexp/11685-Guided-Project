#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 24:00:00
#SBATCH -A cis260094p
#SBATCH -J eeg_train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/classification_cnn_baseline.yaml}"

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
echo "========================================"

python -m scripts.train_classification \
  --config "${CONFIG_PATH}"

echo "========================================"
echo "Job finished on $(date)"
echo "========================================"
