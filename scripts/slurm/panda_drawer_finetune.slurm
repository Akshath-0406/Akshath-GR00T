#!/bin/bash
#SBATCH --job-name=groot-panda-drawer
#SBATCH --account=ka
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=20:00:00
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
# NOTE ON PARTITIONS: avoid gpu_h100_short and gpu_h100 entirely. They share
# the exact same physical node pool (uc3n[071-081,103]), and every single
# "user env retrieval failed requeued held" hold across this whole project
# happened on that pool -- never once on the Ice Lake pool (gpu_a100_il /
# gpu_h100_il, same physical nodes, different GPU populated per node).
# Excluding just gpu_h100_short doesn't help if gpu_h100 is still in the
# list -- it's the identical hardware.
#
# Smoke test before committing to the full run (matches MolmoAct2's own
# smoke-test-first practice) -- runs on whichever of gpu_a100_il,
# gpu_a100_short, gpu_h100_il frees up first (fine to include gpu_a100_short
# here -- light workload, already confirmed to load and run there; --time
# must fit the smallest MaxTime among them, 30min):
#   sbatch --partition=gpu_a100_il,gpu_a100_short,gpu_h100_il \
#       --time=00:30:00 \
#       --export=MAX_STEPS=10,SAVE_STEPS=10 \
#       scripts/slurm/panda_drawer_finetune.slurm
#
# Full run (single uninterrupted job, no chaining -- needs a partition with a
# long enough MaxTime, e.g. gpu_a100_il alone, 48h):
#   sbatch --partition=gpu_a100_il --time=48:00:00 scripts/slurm/panda_drawer_finetune.slurm
#
# Chained short-leg run (self-resubmits every leg via
# `sbatch --dependency=afterany` until MAX_STEPS is reached, matching
# MolmoAct2's panda_full_1epoch.slurm pattern -- each leg is submitted
# against CHAIN_PARTITIONS, so it starts on whichever partition frees up
# first). CHAIN_PARTITIONS deliberately excludes gpu_a100_short too (its 40GB
# A100s aren't confirmed safe for global_batch_size=32 -- see comment below).
# TRAIN_TIMEOUT_SECONDS must leave enough headroom under the SLURM --time for
# this script to reclaim control, scan checkpoints, and resubmit before the
# hard kill:
#   sbatch --partition=gpu_a100_il,gpu_h100_il \
#       --time=00:28:00 \
#       --export=MAX_STEPS=2000,SAVE_STEPS=20,TRAIN_TIMEOUT_SECONDS=1500 \
#       scripts/slurm/panda_drawer_finetune.slurm
#
# DATALOADER_NUM_WORKERS=0 was tested as a factor in the "user env retrieval
# failed" holds (MolmoAct2's known-good scripts always use --num_workers=0;
# ours defaults to 4) and did NOT change anything -- the holds happened with
# workers=0 too. Ruled out; the node pool above is the real factor.

set -euo pipefail

# torchcodec (GR00T's video decoder) needs FFmpeg 4-7's shared libraries at
# runtime, which bwUniCluster3.0 doesn't provide as a module or system
# package. scripts/slurm/prefetch_panda_drawer.sh creates a conda-forge env
# containing a pinned ffmpeg<8 for exactly this purpose (not used for Python
# at all, just to get loadable libs onto disk).
export LD_LIBRARY_PATH="$HOME/.conda/envs/ffmpeg-libs/lib:${LD_LIBRARY_PATH:-}"

MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
# MolmoAct2's known-good scripts always use --num_workers=0 (no multiprocessing
# dataloader workers); ours defaults to >0 (spawns persistent worker
# subprocesses -- see gr00t/experiment/trainer.py). Override to 0 to test
# whether that's a factor in the "user env retrieval failed" holds.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
# Default is ~19.7h, safely under this file's own 20h #SBATCH --time so a
# plain `sbatch panda_drawer_finetune.slurm` behaves as one uninterrupted
# run. Override lower (e.g. 1500) together with --time/--partition to chain
# short legs on gpu_h100_short instead.
TRAIN_TIMEOUT_SECONDS="${TRAIN_TIMEOUT_SECONDS:-71000}"
# Partitions the self-resubmitted chain legs are eligible for -- SLURM starts
# the job on whichever of these frees up first.
# Deliberately excludes gpu_h100_short AND gpu_h100: they share the exact same
# physical node pool (uc3n[071-081,103]) and every "user env retrieval failed
# requeued held" hold across this entire project happened on that pool --
# never once on the Ice Lake pool (gpu_a100_il/gpu_h100_il, same nodes,
# different GPU populated per node). This isn't just gpu_h100_short being
# flaky; gpu_h100 is the identical hardware, so keeping it in the list
# wouldn't avoid anything.
# Also excludes gpu_a100_short (different reason): its A100s are the 40GB
# Cascade Lake variant vs 80GB on the two Ice Lake partitions --
# global_batch_size=32 at bf16 is only confirmed safe on the larger-memory
# nodes. Fine for smoke tests/eval (lighter, already confirmed to run there),
# just not for the real chained run.
CHAIN_PARTITIONS="${CHAIN_PARTITIONS:-gpu_a100_il,gpu_h100_il}"


# Needed for CUDA_HOME: accelerate's unwrap_model() unconditionally imports
# deepspeed (just to type-check the model, even though we're not using it --
# single GPU, no deepspeed backend), and deepspeed's own import-time CUDA
# compatibility check hard-crashes with MissingCUDAException if CUDA_HOME
# isn't set. Not needed for torch/flash-attn (their wheels bundle their own
# CUDA runtime), but deepspeed's import path needs the real toolkit present.
module load devel/cuda/12.8

cd "$HOME/groot"
source .venv/bin/activate

# Diagnostics -- mirrors the pattern the panda_drawer MolmoAct2 scripts use
# (experiments/slurm/panda_smoke.slurm etc.), which have run cleanly on
# gpu_h100_short for multi-day stretches. Cheap, and gives us something to
# compare against if a job dies or gets held before reaching training.
echo "Host: $(hostname)"
echo "Python: $(which python)"
python --version
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

OUTPUT_DIR="$HOME/groot/outputs/panda_drawer"
SLURM_SCRIPT="$HOME/groot/scripts/slurm/panda_drawer_finetune.slurm"
mkdir -p "$OUTPUT_DIR"

# Gr00tTrainer.train() hard-fails if resume_from_checkpoint=True and no
# checkpoint-* exists yet (see gr00t/experiment/trainer.py) -- only resume
# once a prior leg has actually saved one.
RESUME_ARGS=()
if compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
    RESUME_ARGS=(--resume-from-checkpoint)
    echo "Existing checkpoint(s) found in $OUTPUT_DIR -- resuming."
else
    echo "No existing checkpoint in $OUTPUT_DIR -- starting fresh."
fi

# `timeout` lets this script reclaim control before SLURM's own hard kill at
# --time, so the chaining logic below actually gets to run (GR00T has no
# built-in wall-clock stopping criterion the way MolmoAct2's trainer does).
set +e
# --letter-box-transform is required: panda_drawer's three cameras don't share
# one aspect ratio (gripper_cam 4:3 vs left_cam/right_cam 16:9), which crashes
# single-observation inference (torch.stack shape mismatch across views)
# without it. Must match between train and eval -- it's saved into the
# checkpoint's processor_config.json and restored automatically at eval time.
CUDA_VISIBLE_DEVICES=0 timeout "$TRAIN_TIMEOUT_SECONDS" python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/panda_drawer/panda_drawer_many_ee_fullres \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/panda_drawer/panda_drawer_config.py \
    --num-gpus 1 \
    --letter-box-transform \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit 5 \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
    "${RESUME_ARGS[@]}"
train_exit=$?
set -e

# 124 = timeout fired (expected when a leg runs out of wall-clock); anything
# else nonzero is a real failure -- don't mask it by chaining another leg.
if [[ "$train_exit" -ne 0 && "$train_exit" -ne 124 ]]; then
    echo "launch_finetune.py exited with code $train_exit (not a timeout) -- stopping, not chaining." >&2
    exit "$train_exit"
fi

latest_step=0
for ckpt in "$OUTPUT_DIR"/checkpoint-*; do
    [[ -d "$ckpt" ]] || continue
    step_name="${ckpt##*checkpoint-}"
    if [[ "$step_name" =~ ^[0-9]+$ ]] && (( step_name > latest_step )); then
        latest_step="$step_name"
    fi
done
echo "Latest completed checkpoint: checkpoint-${latest_step} (target: ${MAX_STEPS})"

if (( latest_step < MAX_STEPS )); then
    echo "Target not yet reached; chaining next leg on whichever of $CHAIN_PARTITIONS frees up first."
    next_job_output="$(sbatch --partition="$CHAIN_PARTITIONS" --time=00:28:00 \
        --export=MAX_STEPS="$MAX_STEPS",SAVE_STEPS="$SAVE_STEPS",GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE",TRAIN_TIMEOUT_SECONDS="$TRAIN_TIMEOUT_SECONDS",DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS",CHAIN_PARTITIONS="$CHAIN_PARTITIONS" \
        --dependency=afterany:"$SLURM_JOB_ID" \
        "$SLURM_SCRIPT")"
    echo "$next_job_output"
else
    echo "Reached target of ${MAX_STEPS} steps."
fi
