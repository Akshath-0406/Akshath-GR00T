#!/bin/bash
#SBATCH --job-name=groot-panda-drawer-eval
#SBATCH --account=ka
#SBATCH --partition=gpu_a100_il
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
# Usage (CHECKPOINT can be relative to $HOME/groot, or an absolute path --
# e.g. a workspace path from `ws_find groot_ckpts`, since the fine-tune job
# now writes checkpoints there instead of $HOME -- see panda_drawer_finetune.slurm):
#   sbatch --export=CHECKPOINT=outputs/panda_drawer/checkpoint-2000 \
#       scripts/slurm/panda_drawer_open_loop_eval.slurm
#   sbatch --export=CHECKPOINT=$(ws_find groot_ckpts)/outputs/panda_drawer/checkpoint-2000 \
#       scripts/slurm/panda_drawer_open_loop_eval.slurm

set -euo pipefail

# torchcodec (GR00T's video decoder) needs FFmpeg 4-7's shared libraries at
# runtime; see scripts/slurm/prefetch_panda_drawer.sh for how the ffmpeg-libs
# conda env gets created.
export LD_LIBRARY_PATH="$HOME/.conda/envs/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"

if [ -z "${CHECKPOINT:-}" ]; then
    echo "Set CHECKPOINT (relative to \$HOME/groot) via --export=CHECKPOINT=..." >&2
    exit 1
fi

TRAJ_IDS="${TRAJ_IDS:-0}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-16}"

# Accept CHECKPOINT as either absolute (workspace paths) or relative to
# $HOME/groot (the old convention, kept for compatibility).
if [[ "$CHECKPOINT" = /* ]]; then
    MODEL_PATH="$CHECKPOINT"
else
    MODEL_PATH="$HOME/groot/$CHECKPOINT"
fi

cd "$HOME/groot"
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi

python gr00t/eval/open_loop_eval.py \
    --dataset-path examples/panda_drawer/panda_drawer_many_ee_fullres \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path "$MODEL_PATH" \
    --traj-ids $TRAJ_IDS \
    --execution-horizon "$EXECUTION_HORIZON" \
    --save-plot-path "$HOME/groot/outputs/open_loop_eval"
