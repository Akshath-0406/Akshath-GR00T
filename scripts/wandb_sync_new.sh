#!/usr/bin/env bash
# Syncs only the offline wandb run directories that this script hasn't synced
# before, tracked in a manifest file next to them.
#
# `wandb sync --sync-all` re-uploads full history for every offline run each
# time it's invoked when those runs were resumed across processes (id=<name>
# resume="allow", used here for cross-leg SLURM continuity -- see
# gr00t/experiment/experiment.py and panda_drawer_finetune.slurm). wandb's own
# already-synced marker is unreliable for resumed/appended offline runs
# (see https://github.com/wandb/wandb/issues/5865 and
# community.wandb.ai/t/synced-runs-still-shows-as-unsynced/6478), so this
# script tracks synced run directories itself instead of trusting wandb.
#
# Usage: scripts/wandb_sync_new.sh [WANDB_DIR]   (default: $HOME/groot/wandb)
set -euo pipefail

WANDB_DIR="${1:-$HOME/groot/wandb}"
MANIFEST="$WANDB_DIR/.synced_runs"
touch "$MANIFEST"

for run_dir in "$WANDB_DIR"/offline-run-*/ "$WANDB_DIR"/run-*/; do
    [ -d "$run_dir" ] || continue
    name="$(basename "$run_dir")"
    grep -qxF "$name" "$MANIFEST" && continue

    echo "Syncing $name..."
    if wandb sync "$run_dir"; then
        echo "$name" >>"$MANIFEST"
    else
        echo "Failed to sync $name -- will retry next invocation" >&2
    fi
done
