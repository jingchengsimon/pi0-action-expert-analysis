#!/bin/bash
#SBATCH --job-name=pi0-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm-%j.out

# TODO: adjust partition / account / module loads for your cluster.
set -euo pipefail

# source your env here, e.g.:
# source ~/.bashrc && conda activate pi0

python sim/libero_runner.py --config configs/eval_libero.yaml
