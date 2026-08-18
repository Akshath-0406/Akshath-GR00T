#!/bin/bash
# Run this ONCE on a bwUniCluster3.0 LOGIN node -- NOT inside a SLURM job --
# before submitting a panda_sort_three finetune/eval job.
#
# panda_sort_three is four separate HF dataset repos (full/no_green/no_red/
# no_yellow color-exclusion variants of the same sort-three-cubes task),
# each downloaded and prepared independently under its own subdirectory.
# GR00T's launch_finetune.py accepts --dataset-path as an os.pathsep-joined
# list of multiple dataset roots, so these are combined at finetune time
# (join all four prepared paths with ':') rather than merged into one
# dataset here.
#
# Unlike panda_stack_two_cup, these repo names don't have a "_v3.0" suffix,
# so their LeRobot codebase version isn't known up front -- this script
# checks meta/info.json's codebase_version per-repo (cheap: downloads just
# that one file first) and only invokes the v3->v2.1 conversion sub-venv
# (scripts/lerobot_conversion, see its own README) if actually needed.
#
# Usage (from $HOME/groot, with .venv already set up via `uv sync`, and
# already `hf auth login`-ed):
#   bash scripts/slurm/prefetch_panda_sort_three.sh

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

mkdir -p examples/panda_sort_three

declare -A VARIANTS=(
    [full]="kitalr/panda_sort_three_continued_ee_fullres_full"
    [no_green]="kitalr/panda_sort_three_continued_ee_fullres_no_green"
    [no_red]="kitalr/panda_sort_three_continued_ee_fullres_no_red"
    [no_yellow]="kitalr/panda_sort_three_continued_ee_fullres_no_yellow"
)

for variant in "${!VARIANTS[@]}"; do
    repo_id="${VARIANTS[$variant]}"
    dest="$HOME/groot/examples/panda_sort_three/$variant"
    echo ""
    echo "=== $variant ($repo_id) ==="

    if [ -f "$dest/meta/modality.json" ]; then
        echo "Already prepared at $dest -- skipping. Delete meta/modality.json there to force a redo."
        continue
    fi

    source .venv/bin/activate
    codebase_version="$(python -c "
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download(repo_id='$repo_id', repo_type='dataset', filename='meta/info.json')
print(json.load(open(path)).get('codebase_version', 'unknown'))
")"
    deactivate
    echo "Detected codebase_version: $codebase_version"

    if [ "$codebase_version" = "v3.0" ]; then
        echo "v3.0 detected -- converting via scripts/lerobot_conversion (downloads + converts in one step)..."
        pushd scripts/lerobot_conversion
        if [ ! -d .venv ]; then
            uv venv
            source .venv/bin/activate
            uv pip install -e . --verbose
        else
            source .venv/bin/activate
        fi
        export PATH="$HOME/.conda/envs/ffmpeg-libs/bin:$PATH"
        python convert_v3_to_v2.py --repo-id "$repo_id" --root "$HOME/groot/examples/panda_sort_three/$variant"
        deactivate
        popd
        # convert_v3_to_v2.py's --root places the result at <root>/<repo_id>
        # (org/name nested), matching its own convention -- move it up to
        # the flat per-variant path this script/the finetune configs expect.
        mv "$HOME/groot/examples/panda_sort_three/$variant/$repo_id" "$dest.tmp"
        rm -rf "$HOME/groot/examples/panda_sort_three/$variant"
        mv "$dest.tmp" "$dest"
    else
        echo "Already v2.1 (or non-v3) -- downloading directly..."
        source .venv/bin/activate
        python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo_id', repo_type='dataset', local_dir='$dest')
"
        deactivate
    fi

    echo "Spot-checking column names for $variant..."
    source .venv/bin/activate
    python -c "
import pandas as pd
from pathlib import Path
ep = sorted(Path('$dest').glob('data/chunk-*/episode_*.parquet'))[0]
df = pd.read_parquet(ep)
print('Columns:', df.columns.tolist())
"
    python -c "
import json
info = json.load(open('$dest/meta/info.json'))
print('total_episodes:', info.get('total_episodes'))
print('total_frames:', info.get('total_frames'))
"
    echo "^^ If columns don't match observation.proprio.{ee_pos,ee_rot,gripper,joint_pos} /"
    echo "   action.{ee_pos,ee_rot,gripper} / observation.images.{gripper_cam.rgb,left_cam.left,right_cam.left},"
    echo "   STOP and report back before preparing -- the prep script assumes this exact schema."

    echo "Preparing $variant (concatenating state/action columns, writing modality.json)..."
    python examples/panda_sort_three/prepare_panda_sort_three_dataset.py --dataset-path "$dest"
    deactivate
done

echo ""
echo "Done. Combined --dataset-path for finetuning (all four variants, colon-joined):"
echo "examples/panda_sort_three/full:examples/panda_sort_three/no_green:examples/panda_sort_three/no_red:examples/panda_sort_three/no_yellow"
