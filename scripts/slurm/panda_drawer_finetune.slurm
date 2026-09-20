#!/bin/bash
#SBATCH --job-name=groot-panda-drawer
#SBATCH --account=ka
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
# --gres=gpu:1 is the default for the common single-GPU case. For multi-GPU
# (NUM_GPUS>1 below, launched via torchrun), you MUST ALSO override this at
# the `sbatch` command line to match -- e.g.
# `sbatch --gres=gpu:4 --export=NUM_GPUS=4,... panda_drawer_finetune.slurm`.
# This #SBATCH pragma only sets a default; it does not read NUM_GPUS (SLURM
# pragmas can't reference shell variables computed inside the script), so the
# two must be kept in sync manually. gpu_h100 nodes have 4 GPUs each
# (confirmed via `scontrol show partition gpu_h100`: gres/gpu=48 over
# TotalNodes=12) -- torchrun's single-node launch model needs all requested
# GPUs on the one node SLURM allocates, so NUM_GPUS must not exceed that.
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
# Both sized for the single-GPU case (NUM_GPUS=1 below). For multi-GPU,
# torchrun launches NUM_GPUS separate processes, each spawning its own
# DATALOADER_NUM_WORKERS worker pool -- so CPU-side memory/worker pressure
# scales with NUM_GPUS even though these two flat #SBATCH values don't
# (observed directly: a NUM_GPUS=4 run got killed by the Linux OOM-killer on
# system RAM, not a CUDA OOM, with these defaults still in place). Override
# at the `sbatch` command line using the SAME directive type as this file
# (--mem=/--cpus-per-task=), computed as NUM_GPUS x 120G / NUM_GPUS x 16 --
# NOT --mem-per-gpu/--cpus-per-gpu: SLURM treats --mem/--mem-per-cpu/
# --mem-per-gpu (and similarly --cpus-per-task/--cpus-per-gpu) as mutually
# exclusive regardless of whether one comes from this file's #SBATCH lines
# and the other from the command line -- passing --mem-per-gpu here doesn't
# override the file's --mem=120G, it fatals with "mutually exclusive"
# (confirmed directly). e.g. for NUM_GPUS=2:
#   sbatch --gres=gpu:2 --mem=240G --cpus-per-task=32 --export=NUM_GPUS=2,...
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# bwUniCluster3.0 fine-tuning job. Despite the filename, this script is
# dataset-agnostic: DATASET_PATH, MODALITY_CONFIG_PATH, EMBODIMENT_TAG, and
# BASE_MODEL_PATH are all overridable via --export, defaulting to this repo's
# original panda_drawer setup. For a new dataset, no cloning needed -- just
# export different values for those four (and WS_SUBDIR, so it lands in its
# own workspace subfolder instead of panda_drawer's). Example, smoke-testing a
# hypothetical "cup_stacking" dataset starting from the panda_drawer checkpoint
# (same embodiment -- see BASE_MODEL_PATH comment below for when that applies):
#   sbatch --partition=gpu_a100_il,gpu_a100_short,gpu_h100_il,gpu_h100_short,gpu_h100 \
#       --time=00:30:00 --job-name=groot-cup-stacking \
#       --export=MAX_STEPS=10,SAVE_STEPS=10,WS_SUBDIR=cup_stacking,DATASET_PATH=examples/cup_stacking/cup_stacking_dataset,MODALITY_CONFIG_PATH=examples/cup_stacking/cup_stacking_config.py,BASE_MODEL_PATH=$(ws_find groot_ckpts)/outputs/panda_drawer/checkpoint-2000 \
#       scripts/slurm/panda_drawer_finetune.slurm
# (a new dataset still needs its own prefetch/prepare step first -- that
# logic, e.g. prefetch_panda_drawer.sh, is inherently dataset-specific and
# isn't made generic here.)
#
# Before first use:
#   1. bash scripts/slurm/prefetch_panda_drawer.sh   (run on the LOGIN node --
#      downloads the base model + dataset so this job never touches the
#      network; compute nodes are not expected to have internet access)
#   2. mkdir -p logs   (SLURM needs this to exist before the job starts, since
#                        --output/--error are opened before this script runs)
#   3. ws_allocate groot_ckpts 60   (one-time: allocates the workspace
#      checkpoints are written to -- see WORKSPACE NOTE below. Re-run
#      `ws_extend groot_ckpts 60` before it expires if the run is still going.)
#
# WORKSPACE NOTE: checkpoints are large (~25GB each with optimizer state) and
# saved every SAVE_STEPS -- writing them to $HOME triggers bwHPC's automated
# high-I/O notice (HOME is backed up and not meant for this volume/frequency
# of writes). OUTPUT_DIR instead resolves to a workspace (`ws_find`), which is
# local-ish, not backed up, and meant for exactly this. This also means
# checkpoints are NOT permanent -- workspaces expire (default allocated for 60
# days above) and get deleted after a grace period. Retrieve/transfer whatever
# you need before it expires; see the retrieval commands in this repo's
# conversation history / ask for them again if needed.
# WS_SUBDIR defaults to "panda_drawer" (this script's dataset); a future
# script for a different dataset can reuse the same workspace by just
# exporting a different WS_SUBDIR (or WS_NAME for a fully separate workspace)
# -- no script changes needed.
#
# NOTE ON PARTITIONS: gpu_h100/gpu_h100_short (same physical node pool,
# uc3n[071-081,103]) intermittently hit "user env retrieval failed requeued
# held" -- but it's genuinely intermittent, not deterministic: job 5975739
# ran a full 25-minute leg on gpu_h100_short (uc3n105) with zero issues, at
# roughly 3x the steps/sec of the Ice Lake A100/H100 nodes. Given that speed
# advantage and the checkpoint-quarantine safeguard below (which makes a
# mid-save kill on a flaky node self-healing instead of corrupting the resume
# chain), all 4 partitions are included by default so SLURM can race them and
# start on whichever frees up first. If a leg does land in a hold, release it
# manually (`scontrol release <jobid>`) or run a watcher loop -- see
# ~/watch_and_release.sh on the cluster.
#
# Smoke test before committing to the full run (matches MolmoAct2's own
# smoke-test-first practice) -- runs on whichever of gpu_a100_il,
# gpu_a100_short, gpu_h100_il, gpu_h100_short, gpu_h100 frees up first (fine
# to include gpu_a100_short here -- light workload, already confirmed to load
# and run there; --time must fit the smallest MaxTime among them, 30min):
#   sbatch --partition=gpu_a100_il,gpu_a100_short,gpu_h100_il,gpu_h100_short,gpu_h100 \
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
# first). CHAIN_PARTITIONS deliberately excludes gpu_a100_short (its 40GB
# A100s aren't confirmed safe for global_batch_size=32 -- see comment below).
# TRAIN_TIMEOUT_SECONDS must leave enough headroom under the SLURM --time for
# this script to reclaim control, scan checkpoints, and resubmit before the
# hard kill:
#   sbatch --partition=gpu_h100_short,gpu_h100,gpu_a100_il,gpu_h100_il \
#       --time=00:28:00 \
#       --export=MAX_STEPS=2000,SAVE_STEPS=50,TRAIN_TIMEOUT_SECONDS=1500 \
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

# Dataset/model identity -- override all four together to reuse this exact
# script for a different dataset/embodiment without cloning it (only the
# prefetch step -- downloading/preparing the new dataset -- still needs a
# dataset-specific script, since that logic isn't generic). Defaults are this
# repo's original panda_drawer setup, kept for backward compatibility.
DATASET_PATH="${DATASET_PATH:-examples/panda_drawer/panda_drawer_many_ee_fullres}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-examples/panda_drawer/panda_drawer_config.py}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-NEW_EMBODIMENT}"
# nvidia/GR00T-N1.7-3B for an unrelated dataset/embodiment, or a local
# checkpoint dir (e.g. $(ws_find groot_ckpts)/outputs/panda_drawer/checkpoint-2000)
# to continue from an already-adapted checkpoint for the same embodiment.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
# panda_drawer needs this (mixed camera aspect ratios -- see comment at the
# python invocation below); a new dataset with uniform-aspect-ratio cameras
# can turn it off with LETTER_BOX_TRANSFORM_FLAG=--no-letter-box-transform.
LETTER_BOX_TRANSFORM_FLAG="${LETTER_BOX_TRANSFORM_FLAG:---letter-box-transform}"
# Defaults reproduce this repo's original lightweight-finetune behavior
# (backbone frozen, only the projector/action-head/vision-language-bridge
# trainable -- see gr00t/configs/finetune_config.py's tune_llm/tune_visual
# defaults). Override to --tune-llm/--tune-visual for a full-parameter
# finetune of the whole model, including the pretrained VLM backbone. If you
# do, also lower LEARNING_RATE below -- there is no separate/lower learning
# rate for backbone parameters, so the default 1e-4 (tuned for the small
# projector/action-head) risks damaging the pretrained backbone's features
# rather than usefully adapting them.
TUNE_LLM_FLAG="${TUNE_LLM_FLAG:---no-tune-llm}"
TUNE_VISUAL_FLAG="${TUNE_VISUAL_FLAG:---no-tune-visual}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# Off by default (matches this repo's original behavior -- no run so far has
# ever produced a live wandb run, only a static wandb_config.json copy).
# Override to --use-wandb to enable. Compute nodes have no internet access
# (see prefetch note above), so WANDB_MODE=offline is forced unconditionally
# below regardless of this flag -- wandb writes run data to local files under
# $HOME/groot/wandb/ instead of trying (and hanging/failing) to reach
# api.wandb.ai. Sync them from the LOGIN node once you have something to look
# at (needs `wandb login` once on the login node first). The wandb.init() call
# now also passes a stable id=experiment_name + resume="allow" (see
# gr00t/experiment/experiment.py) so every chained leg of a long run appends
# to the SAME run instead of starting a new fragmented one each leg -- but
# that also means `wandb sync --sync-all` re-uploads each resumed run's full
# history every time it's invoked (wandb's synced-marker doesn't reliably
# stick for resumed offline runs). Use `scripts/wandb_sync_new.sh` instead --
# it tracks which run directories it has already synced in a manifest file so
# repeat invocations only upload genuinely new runs.
USE_WANDB_FLAG="${USE_WANDB_FLAG:---no-use-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7}"
export WANDB_MODE=offline

# 1 (default) launches with plain `python`, pinned to GPU 0 -- unchanged from
# this script's original behavior. >1 switches to `torchrun`, splitting
# GLOBAL_BATCH_SIZE across GPUs (gr00t/experiment/experiment.py:
# per_device_train_batch_size = global_batch_size // num_gpus), so e.g.
# NUM_GPUS=4 with GLOBAL_BATCH_SIZE=32 gives each GPU the same per-device
# batch of 8 as a single-GPU GLOBAL_BATCH_SIZE=8 run, while still reaching a
# global/effective batch of 32 (before gradient accumulation). Must evenly
# divide GLOBAL_BATCH_SIZE (launch_finetune.py asserts this and errors
# otherwise). Remember to also pass --gres=gpu:$NUM_GPUS on the `sbatch`
# command line -- see the note above #SBATCH --gres=gpu:1.
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

MAX_STEPS="${MAX_STEPS:-2000}"
# 0 disables milestone saving (default). Set > 0 (e.g. 5000) to additionally
# copy every checkpoint whose step is a multiple of this value into
# <OUTPUT_DIR>/milestones/checkpoint-<step>, kept indefinitely regardless of
# save_total_limit -- lets you compare performance at each milestone without
# it being rotated away by the regular SAVE_STEPS/save_total_limit cleanup.
MILESTONE_STEPS="${MILESTONE_STEPS:-0}"
# 50 balances I/O against progress-loss risk: at ~25GB/checkpoint this is
# ~5x fewer bytes written to the workspace than the SAVE_STEPS=20 this project
# used earlier (which contributed to a bwHPC high-I/O notice on $HOME before
# checkpoints moved to a workspace -- see WORKSPACE NOTE above), while still
# capping any mid-save-kill progress loss at 50 steps.
SAVE_STEPS="${SAVE_STEPS:-50}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
# Effective batch = GLOBAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS. Only
# GLOBAL_BATCH_SIZE=32 is confirmed to fit in memory on this model/dataset
# (3B params, 3 full-res cameras) -- to test a larger effective batch, raise
# this instead of GLOBAL_BATCH_SIZE, since accumulation reaches a bigger
# effective batch via multiple safe-sized forward/backward passes per
# optimizer step, not a bigger single pass (no extra memory needed, just more
# wall-clock time per step -- roughly proportional to this value).
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
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
# Includes gpu_h100_short/gpu_h100 (uc3n[071-081,103] EPYC pool): flaky with
# intermittent "user env retrieval failed requeued held" holds, but genuinely
# intermittent (job 5975739 ran a full clean 25-minute leg there at ~3x the
# steps/sec of the Ice Lake nodes) and no longer a resume-safety risk now that
# incomplete checkpoints are quarantined above -- a hold or a mid-save kill on
# this pool just costs a requeue/retry, it can't corrupt the resume chain.
# Excludes gpu_a100_short (different reason): its A100s are the 40GB Cascade
# Lake variant vs 80GB/94GB on the other three partitions -- global_batch_size
# =32 at bf16 is only confirmed safe on the larger-memory nodes. Fine for
# smoke tests/eval (lighter, already confirmed to run there), just not for the
# real chained run.
CHAIN_PARTITIONS="${CHAIN_PARTITIONS:-gpu_h100_short,gpu_h100,gpu_a100_il,gpu_h100_il}"


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

# Echo the resolved run config so a --export override that silently didn't
# take (typo'd var name, wrong quoting, etc.) is visible in the log instead
# of only discoverable by inspecting the finished checkpoint's config.json.
echo "--- Run config ---"
echo "DATASET_PATH: $DATASET_PATH"
echo "MODALITY_CONFIG_PATH: $MODALITY_CONFIG_PATH"
echo "EMBODIMENT_TAG: $EMBODIMENT_TAG"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "MAX_STEPS: $MAX_STEPS"
echo "SAVE_STEPS: $SAVE_STEPS"
echo "MILESTONE_STEPS: $MILESTONE_STEPS"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
echo "TUNE_LLM_FLAG: $TUNE_LLM_FLAG"
echo "TUNE_VISUAL_FLAG: $TUNE_VISUAL_FLAG"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "USE_WANDB_FLAG: $USE_WANDB_FLAG"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "NUM_GPUS: $NUM_GPUS"
echo "MASTER_PORT: $MASTER_PORT"
echo "------------------"

# Workspace-backed output dir -- see WORKSPACE NOTE near the top of this file.
# OUTPUT_DIR itself takes priority if already set (this is how chained legs
# below propagate the exact resolved path to the next leg, so only the very
# first leg needs to call ws_find / needs WS_NAME+WS_SUBDIR to be correct).
WS_NAME="${WS_NAME:-groot_ckpts}"
WS_SUBDIR="${WS_SUBDIR:-panda_drawer}"
OUTPUT_DIR="${OUTPUT_DIR:-$(ws_find "$WS_NAME")/outputs/$WS_SUBDIR}"
SLURM_SCRIPT="$HOME/groot/scripts/slurm/panda_drawer_finetune.slurm"
mkdir -p "$OUTPUT_DIR"

# HF's get_last_checkpoint() (called by Gr00tTrainer.train() to resolve
# resume_from_checkpoint=True) picks the highest-numbered checkpoint-N
# directory purely by name -- it does not check whether the save actually
# completed. If `timeout` below fires mid-save (a SAVE_STEPS checkpoint can
# land right at the wall-clock deadline), the highest-numbered checkpoint can
# be a truncated partial write (missing shards/index/trainer_state.json),
# which crashes resume with FileNotFoundError instead of falling back to the
# last good one. trainer_state.json is the last file HF's _save_checkpoint
# writes, so its absence means that save never finished. Quarantine any such
# checkpoint (rename so it no longer matches transformers' `^checkpoint-\d+$`
# pattern) before resume detection runs, so get_last_checkpoint() and the
# chaining step's own "latest_step" scan both skip it automatically.
# Skip anything already quarantined -- the checkpoint-* glob below also
# matches previously-renamed checkpoint-N.incomplete dirs, and without this
# guard every run would re-quarantine them, growing another ".incomplete"
# suffix each time (observed: checkpoint-320.incomplete.incomplete).
for ckpt in "$OUTPUT_DIR"/checkpoint-*; do
    [[ -d "$ckpt" ]] || continue
    [[ "$ckpt" == *.incomplete ]] && continue
    if [[ ! -f "$ckpt/trainer_state.json" ]]; then
        echo "Quarantining incomplete checkpoint (no trainer_state.json): $ckpt -> ${ckpt}.incomplete"
        mv "$ckpt" "${ckpt}.incomplete"
    fi
done

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
LAUNCH_ARGS=(
    gr00t/experiment/launch_finetune.py
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --num-gpus "$NUM_GPUS"
    "$LETTER_BOX_TRANSFORM_FLAG"
    "$TUNE_LLM_FLAG"
    "$TUNE_VISUAL_FLAG"
    --learning-rate "$LEARNING_RATE"
    "$USE_WANDB_FLAG"
    --wandb-project "$WANDB_PROJECT"
    --output-dir "$OUTPUT_DIR"
    --max-steps "$MAX_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit 5
    --milestone-steps "$MILESTONE_STEPS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    "${RESUME_ARGS[@]}"
)

# NUM_GPUS=1 (default): plain python, pinned to GPU 0 -- unchanged original
# behavior. NUM_GPUS>1: torchrun, one process per GPU on this single node
# (see NUM_GPUS comment above for how GLOBAL_BATCH_SIZE splits across them).
# Mirrors examples/finetune.sh's launcher-selection pattern.
if [[ "$NUM_GPUS" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" timeout "$TRAIN_TIMEOUT_SECONDS" python "${LAUNCH_ARGS[@]}"
else
    timeout "$TRAIN_TIMEOUT_SECONDS" torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${LAUNCH_ARGS[@]}"
fi
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
    # Pass overrides as real environment variables (prefix-assignment) rather
    # than cramming them into --export=VAR1=val1,VAR2=val2 -- sbatch's --export
    # splits on EVERY comma to separate different variables, so a
    # comma-containing value like CHAIN_PARTITIONS
    # (gpu_h100_short,gpu_h100,gpu_a100_il,gpu_h100_il) gets silently truncated
    # to just its first fragment ("gpu_h100_short") when embedded that way --
    # which then pins every subsequent chained leg to one partition instead of
    # racing all four (observed: job pinned to gpu_h100_short alone, scheduled
    # 3 days out while the other three partitions sat idle). --export=ALL
    # (sbatch's default even without the flag) forwards the whole calling
    # environment, so prefix-assigning these here is sufficient -- no
    # comma-splitting involved since they're real env vars, not a
    # single-string list.
    # --job-name is passed explicitly (SLURM doesn't forward it via
    # --export=ALL -- job name isn't an env var) so every chained leg keeps
    # the name the FIRST leg was submitted with, instead of falling back to
    # this script's #SBATCH --job-name default. Without this, `sacct
    # --name=<custom-name>` only ever finds the first leg of a chain.
    # --mem/--cpus-per-task are likewise passed explicitly and scaled by
    # NUM_GPUS -- without this, a chained leg silently falls back to this
    # file's #SBATCH --mem=120G/--cpus-per-task=16 (the NUM_GPUS=1 sizing)
    # regardless of the actual NUM_GPUS, even if the FIRST leg was submitted
    # with a correctly-scaled --mem/--cpus-per-task at the command line
    # (confirmed directly: a NUM_GPUS=2 chain's first leg ran fine under a
    # manually-passed --mem=240G, but the next self-chained leg -- which only
    # inherited --gres=gpu:2, not --mem -- got Linux-OOM-killed on a
    # DataLoader worker mid-checkpoint-save under the fallen-back 120G).
    next_job_output="$(MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" MILESTONE_STEPS="$MILESTONE_STEPS" GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" TRAIN_TIMEOUT_SECONDS="$TRAIN_TIMEOUT_SECONDS" DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS" CHAIN_PARTITIONS="$CHAIN_PARTITIONS" OUTPUT_DIR="$OUTPUT_DIR" DATASET_PATH="$DATASET_PATH" MODALITY_CONFIG_PATH="$MODALITY_CONFIG_PATH" EMBODIMENT_TAG="$EMBODIMENT_TAG" BASE_MODEL_PATH="$BASE_MODEL_PATH" LETTER_BOX_TRANSFORM_FLAG="$LETTER_BOX_TRANSFORM_FLAG" TUNE_LLM_FLAG="$TUNE_LLM_FLAG" TUNE_VISUAL_FLAG="$TUNE_VISUAL_FLAG" LEARNING_RATE="$LEARNING_RATE" USE_WANDB_FLAG="$USE_WANDB_FLAG" WANDB_PROJECT="$WANDB_PROJECT" NUM_GPUS="$NUM_GPUS" MASTER_PORT="$MASTER_PORT" \
        sbatch --partition="$CHAIN_PARTITIONS" --time=00:28:00 --gres=gpu:"$NUM_GPUS" \
        --mem="$((NUM_GPUS * 120))G" --cpus-per-task="$((NUM_GPUS * 16))" \
        --job-name="$SLURM_JOB_NAME" \
        --export=ALL \
        --dependency=afterany:"$SLURM_JOB_ID" \
        "$SLURM_SCRIPT")"
    echo "$next_job_output"
else
    echo "Reached target of ${MAX_STEPS} steps."
fi
