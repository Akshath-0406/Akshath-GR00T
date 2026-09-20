#!/usr/bin/env bash
# Syncs offline wandb run directories that have new data since this script
# last synced them, tracked (name -> last-synced mtime) in a manifest file.
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
# A run directory is only skipped if none of its files have changed since the
# last successful sync -- tracking a plain "already synced" boolean instead
# would permanently skip a run that was still being appended to (a leg still
# training, or a resumed run that later got another chained leg) after its
# first, partial sync, silently dropping every step logged afterward.
#
# Usage: scripts/wandb_sync_new.sh [WANDB_DIR]   (default: $HOME/groot/wandb)
set -euo pipefail

WANDB_DIR="${1:-$HOME/groot/wandb}"
MANIFEST="$WANDB_DIR/.synced_runs"
touch "$MANIFEST"

for run_dir in "$WANDB_DIR"/offline-run-*/ "$WANDB_DIR"/run-*/; do
    [ -d "$run_dir" ] || continue
    name="$(basename "$run_dir")"

    current_mtime=$(find "$run_dir" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
    current_mtime=${current_mtime%.*}
    [ -z "$current_mtime" ] && continue

    recorded_mtime=$(awk -F'\t' -v n="$name" '$1==n{v=$2} END{print v}' "$MANIFEST")

    if [ -n "$recorded_mtime" ] && [ "$recorded_mtime" -ge "$current_mtime" ] 2>/dev/null; then
        continue
    fi

    echo "Syncing $name..."
    if wandb sync "$run_dir"; then
        awk -F'\t' -v n="$name" '$1!=n' "$MANIFEST" >"$MANIFEST.tmp" 2>/dev/null || true
        mv "$MANIFEST.tmp" "$MANIFEST"
        printf '%s\t%s\n' "$name" "$current_mtime" >>"$MANIFEST"
    else
        echo "Failed to sync $name -- will retry next invocation" >&2
    fi
done
