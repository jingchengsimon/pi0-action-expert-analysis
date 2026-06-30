#!/bin/bash
#SBATCH --job-name=pi0_pipeline
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/pipeline_%j.out

# ---------------------------------------------------------------------------
# Pipeline orchestrator submission script.
#
# Submits the full experiment pipeline (data_prep → train → eval)
# as chained Slurm jobs via pipeline/run_experiment.py.
#
# Usage:
#   sbatch scripts/submit_pipeline.sbatch
#
# Environment variables:
#   VARIANT          experiment variant (default: lora_both)
#   DATA_DIR         LeRobot dataset path
#   OUTPUT_DIR       pipeline output root
#   NUM_STEPS        training steps
#   EVAL_SUITES      space-separated LIBERO suites
# ---------------------------------------------------------------------------

set -euo pipefail

VARIANT="${VARIANT:-lora_both}"
DATA_DIR="${DATA_DIR:-data/libero_lerobot/libero_spatial}"
OUTPUT_DIR="${OUTPUT_DIR:-results/experiments}"
NUM_STEPS="${NUM_STEPS:-30000}"
EVAL_SUITES="${EVAL_SUITES:-libero_spatial libero_object}"

export BASHRCSOURCED=1
source ~/.bashrc
conda activate pi0

mkdir -p "${OUTPUT_DIR}"
mkdir -p results/slurm

echo "=== Pipeline Submission ==="
echo "Variant:    ${VARIANT}"
echo "Data:       ${DATA_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "Steps:      ${NUM_STEPS}"
echo "Suites:     ${EVAL_SUITES}"
echo "=========================="

python pipeline/run_experiment.py \
    --variant "${VARIANT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-steps "${NUM_STEPS}" \
    --eval-suites ${EVAL_SUITES}

echo "Pipeline jobs submitted. Check ${OUTPUT_DIR}/${VARIANT}/pipeline_manifest.json"
