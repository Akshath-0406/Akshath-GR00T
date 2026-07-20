#!/bin/bash
#SBATCH --job-name=groot-panda-drawer
#SBATCH --account=ka
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# bwUniCluster3.0 fine-tuning job for the panda_drawer NEW_EMBODIMENT setup.
#
# Before first use:
#   1. bash scripts/slurm/prefetch_panda_drawer.sh   (run on the LOGIN node --
#      downloads the base model + dataset so this job never touches the
#      network; compute nodes are not expected to have internet access)
#   2. mkdir -p logs   (SLURM needs this to exist before the job starts, since
#                        --output/--error are opened before this script runs)
#
# Smoke test before committing to the full run (matches MolmoAct2's own
# smoke-test-first practice) -- override partition/time/steps at submit time:
#   sbatch --partition=gpu_h100_short --time=00:30:00 \
#       --export=ALL,MAX_STEPS=10,SAVE_STEPS=10 \
#       scripts/slurm/panda_drawer_finetune.slurm
#
# Full run:
#   sbatch scripts/slurm/panda_drawer_finetune.slurm

set -euo pipefail

MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"

module load devel/cuda/12.8

cd "$HOME/groot"
source .venv/bin/activate

mkdir -p "$HOME/groot/outputs"

CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/panda_drawer/panda_drawer_many_ee_fullres \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/panda_drawer/panda_drawer_config.py \
    --num-gpus 1 \
    --output-dir "$HOME/groot/outputs/panda_drawer" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit 5 \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --dataloader-num-workers 4
