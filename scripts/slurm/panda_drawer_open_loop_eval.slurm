#!/bin/bash
#SBATCH --job-name=groot-panda-drawer-eval
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Open-loop evaluation of a panda_drawer NEW_EMBODIMENT checkpoint.
#
# Before first use:
#   1. bash scripts/slurm/prefetch_panda_drawer.sh   (run on the LOGIN node,
#      if not already done for the fine-tune job -- makes sure the dataset
#      is present so this job never touches the network)
#   2. mkdir -p logs
#
# Usage:
#   sbatch --export=CHECKPOINT=outputs/panda_drawer/checkpoint-2000 \
#       scripts/slurm/panda_drawer_open_loop_eval.slurm

set -euo pipefail

# Non-interactive batch shells don't source ~/.bashrc, so `uv` needs to be put
# on PATH explicitly here rather than relying on --export=ALL to propagate it
# from the submitting session.
export PATH="$HOME/.local/bin:$PATH"

if [ -z "${CHECKPOINT:-}" ]; then
    echo "Set CHECKPOINT (relative to \$HOME/groot) via --export=CHECKPOINT=..." >&2
    exit 1
fi

TRAJ_IDS="${TRAJ_IDS:-0}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"

module load devel/cuda/12.8

cd "$HOME/groot"
source .venv/bin/activate

uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path examples/panda_drawer/panda_drawer_many_ee_fullres \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path "$HOME/groot/$CHECKPOINT" \
    --traj-ids $TRAJ_IDS \
    --execution-horizon "$EXECUTION_HORIZON" \
    --save-plot-path "$HOME/groot/outputs/open_loop_eval"
