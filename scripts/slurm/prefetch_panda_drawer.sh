#!/bin/bash
# Run this ONCE on a bwUniCluster3.0 LOGIN node -- NOT inside a SLURM job --
# before submitting scripts/slurm/panda_drawer_finetune.slurm or
# panda_drawer_open_loop_eval.slurm.
#
# HPC compute nodes conventionally have no internet access (only login
# nodes do); this hasn't been explicitly confirmed for bwUniCluster3.0's
# docs specifically, but it's near-universal practice and worth defending
# against regardless. This script pre-downloads everything network-dependent
# into the shared $HOME cache (visible from compute nodes too, since $HOME
# is a cluster-wide Lustre filesystem) so the actual SLURM job never needs
# to reach the network.
#
# Usage (from $HOME/groot, with .venv already set up via `uv sync`):
#   bash scripts/slurm/prefetch_panda_drawer.sh

set -euo pipefail

cd "$HOME/groot"
source .venv/bin/activate

echo "Checking Hugging Face authentication..."
uv run python -c "from huggingface_hub import whoami; print('Logged in as:', whoami()['name'])" || {
    echo "Not authenticated. Run 'uv run hf auth login' first." >&2
    echo "(nvidia/GR00T-N1.7-3B pulls in the gated nvidia/Cosmos-Reason2-2B backbone --" >&2
    echo " request access at https://huggingface.co/nvidia/Cosmos-Reason2-2B first if you haven't.)" >&2
    exit 1
}

echo "Downloading base model nvidia/GR00T-N1.7-3B..."
uv run hf download nvidia/GR00T-N1.7-3B

echo "Downloading dataset kitalr/panda_drawer_many_ee_fullres..."
mkdir -p examples/panda_drawer
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'kitalr/panda_drawer_many_ee_fullres',
    repo_type='dataset',
    local_dir='examples/panda_drawer/panda_drawer_many_ee_fullres',
)
"

echo "Preparing dataset (concatenating state/action columns, writing modality.json)..."
uv run python examples/panda_drawer/prepare_panda_drawer_dataset.py \
    --dataset-path examples/panda_drawer/panda_drawer_many_ee_fullres

echo "Done. Everything the SLURM job needs is now cached under \$HOME."
echo "You can now: mkdir -p logs && sbatch scripts/slurm/panda_drawer_finetune.slurm"
