#!/bin/bash
#SBATCH --job-name=pi0_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm/train_%j.out
#SBATCH --error=results/slurm/train_%j.err

# ---------------------------------------------------------------------------
# Single-GPU LoRA fine-tuning script for pi0.
#
# Usage:
#   sbatch scripts/submit_train.sbatch
#
# Environment variables (optional overrides):
#   TRAIN_CONFIG     config name (default: pi0_libero_low_mem_finetune)
#   DATA_DIR         path to LeRobot dataset
#   OUTPUT_DIR       checkpoint output directory
#   NUM_STEPS        training steps (default: 30000)
# ---------------------------------------------------------------------------

set -euo pipefail

TRAIN_CONFIG="${TRAIN_CONFIG:-pi0_libero_low_mem_finetune}"
DATA_DIR="${DATA_DIR:-data/libero_lerobot/libero_spatial}"
OUTPUT_DIR="${OUTPUT_DIR:-results/finetune/lora}"
NUM_STEPS="${NUM_STEPS:-30000}"

module load cuda/12.1
source ~/.bashrc
conda activate pi0

mkdir -p "${OUTPUT_DIR}"
mkdir -p results/slurm

echo "=== Single-GPU LoRA Training ==="
echo "Config:  ${TRAIN_CONFIG}"
echo "Data:    ${DATA_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "Steps:   ${NUM_STEPS}"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "================================="

python finetune/train.py \
    --config "${TRAIN_CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-steps "${NUM_STEPS}" \
    --backend jax

echo "Training complete. Checkpoints in: ${OUTPUT_DIR}"
