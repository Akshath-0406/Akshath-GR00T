#!/bin/bash
#SBATCH --job-name=groot-panda-drawer-eval
#SBATCH --account=ka
#SBATCH --partition=gpu_h100_short,gpu_h100,gpu_a100_il,gpu_h100_il,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Open-loop evaluation of a GR00T NEW_EMBODIMENT checkpoint. Despite the
# filename, this script is dataset-agnostic: DATASET_PATH and EMBODIMENT_TAG
# are overridable via --export (defaulting to this repo's original
# panda_drawer setup), same pattern as panda_drawer_finetune.slurm -- no
# cloning needed for a new dataset.
#
# Runs on whichever of all 5 GPU partitions frees up first (single forward
# passes over a handful of trajectories -- light enough that even the 40GB
# gpu_a100_short is fine here, unlike the real fine-tune's global_batch_size=32
# which isn't confirmed safe there; see panda_drawer_finetune.slurm). Override
# with `sbatch --partition=...` if you want to pin it to one partition.
#
# Before first use:
#   1. run the new dataset's prefetch/prepare step on the LOGIN node (e.g.
#      prefetch_panda_drawer.sh for the original dataset) -- makes sure it's
#      present so this job never touches the network
#   2. mkdir -p logs
#
# Usage (CHECKPOINT can be relative to $HOME/groot, or an absolute path --
# e.g. a workspace path from `ws_find groot_ckpts`, since the fine-tune job
# now writes checkpoints there instead of $HOME -- see panda_drawer_finetune.slurm):
#   sbatch --export=CHECKPOINT=outputs/panda_drawer/checkpoint-2000 \
#       scripts/slurm/panda_drawer_open_loop_eval.slurm
#   sbatch --export=CHECKPOINT=$(ws_find groot_ckpts)/outputs/panda_drawer/checkpoint-2000 \
#       scripts/slurm/panda_drawer_open_loop_eval.slurm
# For a different dataset, also override DATASET_PATH (and EMBODIMENT_TAG if
# it differs):
#   sbatch --export=CHECKPOINT=$(ws_find groot_ckpts)/outputs/cup_stacking/checkpoint-2000,DATASET_PATH=examples/cup_stacking/cup_stacking_dataset \
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
DATASET_PATH="${DATASET_PATH:-examples/panda_drawer/panda_drawer_many_ee_fullres}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
# open_loop_eval.py's own default (4) is on the low end for a flow-matching
# action head -- fewer denoising steps means coarser sampling noise
# independent of how well-trained the model is. Override higher (e.g. 10-20)
# to check whether prediction jerkiness is a sampling artifact rather than a
# training-quality issue.
DENOISING_STEPS="${DENOISING_STEPS:-4}"

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

# Named after the checkpoint being evaluated (e.g. checkpoint-4000.png) so a
# folder of these self-documents progress over time, and an explicit .png
# extension (matplotlib infers PNG from a missing extension anyway, but a
# bare extensionless filename confuses some viewers).
# NOTE: re-running eval on the SAME checkpoint (e.g. to try a different
# --denoising-steps) overwrites this same file rather than keeping both --
# rename/move the old one first if you want to keep it.
# NOTE: open_loop_eval.py's plot_trajectory_results() saves every trajectory
# in --traj-ids to this same path (its per-trajectory filenames comment isn't
# actually implemented) -- if you ever pass multiple TRAJ_IDS in one job, only
# the last trajectory's plot survives. Submit one job per trajectory if you
# need separate plots for each.
CKPT_NAME="$(basename "$MODEL_PATH")"
PLOT_PATH="$HOME/groot/outputs/open_loop_eval/${CKPT_NAME}.png"
mkdir -p "$(dirname "$PLOT_PATH")"

python gr00t/eval/open_loop_eval.py \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --model-path "$MODEL_PATH" \
    --traj-ids $TRAJ_IDS \
    --execution-horizon "$EXECUTION_HORIZON" \
    --denoising-steps "$DENOISING_STEPS" \
    --save-plot-path "$PLOT_PATH"

echo "Plot saved to: $PLOT_PATH"
