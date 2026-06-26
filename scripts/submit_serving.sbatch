#!/bin/bash
#SBATCH --job-name=pi0-serving
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out

# Amarel-specific module loads
module purge
module load cuda/12.1

set -euo pipefail

# Activate environment
source ~/.bashrc
conda activate pi0

# Ensure openpi is installed (editable from submodule)
pip install -e third_party/openpi --quiet 2>/dev/null || true

# Launch the policy server
python serving/launch_policy_server.py --config configs/serving.yaml
#!/bin/bash
#SBATCH --job-name=pi0-serving
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%j.out

# TODO: adjust partition / account / module loads for your cluster.
set -euo pipefail

# source your env here, e.g.:
# source ~/.bashrc && conda activate pi0

python serving/launch_policy_server.py --config configs/serving.yaml
