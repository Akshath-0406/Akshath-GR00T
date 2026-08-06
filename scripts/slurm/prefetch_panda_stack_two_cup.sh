#!/bin/bash
# Run this ONCE on a bwUniCluster3.0 LOGIN node -- NOT inside a SLURM job --
# before submitting a panda_stack_two_cup finetune/eval job (see
# scripts/slurm/panda_drawer_finetune.slurm's own header comment for how to
# reuse that script against a different dataset via --export).
#
# Mirrors scripts/slurm/prefetch_panda_drawer.sh, with one extra step: the
# source dataset (kitalr/panda_stack_two_cup_ee_fullres_v3.0) ships in
# LeRobot v3.0 format on the Hub, but GR00T's LeRobotEpisodeLoader (and
# examples/panda_stack_two_cup/prepare_panda_stack_two_cup_dataset.py) expect
# v2.1's per-episode-parquet layout. scripts/lerobot_conversion/convert_v3_to_v2.py
# handles both the download AND the v3->v2.1 conversion in one call, but needs
# its OWN sub-venv (separate pyproject.toml/lerobot install, incompatible with
# groot's main .venv) -- see scripts/lerobot_conversion/README.md.
#
# Usage (from $HOME/groot, with .venv already set up via `uv sync`, and
# already `hf auth login`-ed):
#   bash scripts/slurm/prefetch_panda_stack_two_cup.sh

set -euo pipefail

cd "$HOME/groot"

echo "Checking for the ffmpeg-libs conda env (see prefetch_panda_drawer.sh for why)..."
module load devel/miniforge
if ! conda env list | grep -q '^ffmpeg-libs '; then
    conda create -n ffmpeg-libs -c conda-forge 'ffmpeg<8' -y
else
    echo "ffmpeg-libs already exists, skipping."
fi

echo "Downloading base model nvidia/GR00T-N1.7-3B (skipped if already cached)..."
source .venv/bin/activate
uv run hf download nvidia/GR00T-N1.7-3B
deactivate

echo "Converting kitalr/panda_stack_two_cup_ee_fullres_v3.0 from LeRobot v3.0 to v2.1..."
echo "(uses its own sub-venv, not the main groot .venv -- see scripts/lerobot_conversion/README.md)"
mkdir -p examples/panda_stack_two_cup
pushd scripts/lerobot_conversion
if [ ! -d .venv ]; then
    uv venv
    source .venv/bin/activate
    uv pip install -e . --verbose
else
    source .venv/bin/activate
fi
python convert_v3_to_v2.py \
    --repo-id kitalr/panda_stack_two_cup_ee_fullres_v3.0 \
    --root "$HOME/groot/examples/panda_stack_two_cup"
deactivate
popd

DATASET_PATH="$HOME/groot/examples/panda_stack_two_cup/kitalr/panda_stack_two_cup_ee_fullres_v3.0"

echo ""
echo "Spot-checking column names against what prepare_panda_stack_two_cup_dataset.py expects..."
source .venv/bin/activate
python -c "
import pandas as pd
from pathlib import Path
ep = sorted(Path('$DATASET_PATH').glob('data/chunk-*/episode_*.parquet'))[0]
df = pd.read_parquet(ep)
print('Columns:', df.columns.tolist())
print('Rows in first episode:', len(df))
"
python -c "
import json
info = json.load(open('$DATASET_PATH/meta/info.json'))
print('total_episodes:', info.get('total_episodes'))
print('total_frames:', info.get('total_frames'))
print('features:', list(info['features'].keys()))
"
echo ""
echo "^^ If the printed columns don't match observation.proprio.{ee_pos,ee_rot,gripper,joint_pos} /"
echo "   action.{ee_pos,ee_rot,gripper} / observation.images.{gripper_cam.rgb,left_cam.left,right_cam.left},"
echo "   STOP and report back before running the next step -- the prep script assumes this exact schema."
echo ""

echo "Preparing dataset (concatenating state/action columns, writing modality.json)..."
python examples/panda_stack_two_cup/prepare_panda_stack_two_cup_dataset.py \
    --dataset-path "$DATASET_PATH"

echo "Done. Everything the SLURM job needs is now cached under \$HOME."
echo "Dataset path for --export=DATASET_PATH=...: $DATASET_PATH"
