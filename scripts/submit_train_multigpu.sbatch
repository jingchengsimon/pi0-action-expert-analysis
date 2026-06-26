#!/bin/bash
#SBATCH --job-name=pi0_train_multigpu
#SBATCH --partition=gpu-redhat
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=results/slurm/train_multigpu_%j.out
#SBATCH --error=results/slurm/train_multigpu_%j.err

# ---------------------------------------------------------------------------
# Multi-GPU training script for pi0 fine-tuning.
#
# Supports two backends:
#   1. JAX FSDP:  --fsdp_devices=2 (shards model across 2 GPUs)
#   2. PyTorch DDP: torchrun --nproc_per_node=2 (data parallel)
#
# Usage:
#   sbatch scripts/submit_train_multigpu.sbatch
#
# Environment variables (optional overrides):
#   TRAIN_BACKEND    jax|pytorch (default: jax)
#   TRAIN_CONFIG     config name (default: pi0_libero)
#   DATA_DIR         path to LeRobot dataset
#   OUTPUT_DIR       checkpoint output directory
#   NUM_STEPS        training steps (default: 30000)
#   NUM_GPUS         number of GPUs (default: 2)
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Defaults ---
TRAIN_BACKEND="${TRAIN_BACKEND:-jax}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi0_libero}"
DATA_DIR="${DATA_DIR:-data/libero_lerobot/libero_spatial}"
OUTPUT_DIR="${OUTPUT_DIR:-results/finetune/multigpu}"
NUM_STEPS="${NUM_STEPS:-30000}"
NUM_GPUS="${NUM_GPUS:-2}"

# --- Environment setup ---
module load cuda/12.1
source ~/.bashrc
conda activate pi0

# --- Create output directory ---
mkdir -p "${OUTPUT_DIR}"
mkdir -p results/slurm

# --- Print job info ---
echo "=== Multi-GPU Training Job ==="
echo "Backend:    ${TRAIN_BACKEND}"
echo "Config:     ${TRAIN_CONFIG}"
echo "Data:       ${DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Steps:      ${NUM_STEPS}"
echo "GPUs:       ${NUM_GPUS}"
echo "Node:       ${SLURM_NODELIST}"
echo "GPU IDs:    ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "================================"

# --- GPU info ---
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# --- Launch training ---
if [ "${TRAIN_BACKEND}" = "pytorch" ]; then
    echo "Launching PyTorch DDP training with ${NUM_GPUS} GPUs..."
    srun torchrun \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port=29500 \
        finetune/train.py \
        --config "${TRAIN_CONFIG}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-steps "${NUM_STEPS}" \
        --backend pytorch \
        --num-gpus "${NUM_GPUS}"

elif [ "${TRAIN_BACKEND}" = "jax" ]; then
    echo "Launching JAX FSDP training with ${NUM_GPUS} GPUs..."
    srun python finetune/train.py \
        --config "${TRAIN_CONFIG}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-steps "${NUM_STEPS}" \
        --backend jax \
        --fsdp-devices "${NUM_GPUS}"

else
    echo "ERROR: Unknown backend '${TRAIN_BACKEND}'. Use 'jax' or 'pytorch'."
    exit 1
fi

echo "Training complete. Checkpoints in: ${OUTPUT_DIR}"
