#!/bin/bash
#SBATCH --job-name=pi0-eval
#SBATCH --partition=gpu-redhat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out

# Amarel-specific module loads
module purge
module load cuda/12.1

set -euo pipefail

# Activate environment
export BASHRCSOURCED=1
source ~/.bashrc
conda activate pi0

# Ensure openpi is installed (editable from submodule)
pip install -e third_party/openpi --quiet 2>/dev/null || true

# Run the evaluation harness
cd ~/projects/pi0-action-expert-analysis
python eval_harness/run_eval.py \
    --config configs/eval_libero.yaml \
    --serving-config configs/serving.yaml \
    --results-dir results/
