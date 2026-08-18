#!/usr/bin/env python3
"""Prepare a panda_sort_three_* dataset variant for GR00T N1.7 (NEW_EMBODIMENT).

Same Panda P4 rig, same imikit teleop/recording pipeline, and same GR00T
NEW_EMBODIMENT setup as examples/panda_drawer/prepare_panda_drawer_dataset.py
-- this script mirrors it exactly (17-D state, 10-D action, same three
camera keys) rather than genericizing across datasets, per this repo's
per-dataset example convention.

This is intended to be run once per downloaded variant (full / no_green /
no_red / no_yellow), each under its own subdirectory -- see
scripts/slurm/prefetch_panda_sort_three.sh, which loops over all four HF
repos (kitalr/panda_sort_three_continued_ee_fullres_{full,no_green,no_red,
no_yellow}) and calls this script once per variant. GR00T's
--dataset-path accepts an os.pathsep-joined list of multiple dataset
roots, so the four prepared variants are combined at finetune time rather
than merged into one directory here.

Before running this against a fresh download, spot-check that the raw
column names actually match panda_drawer's (observation.proprio.ee_pos/
ee_rot/gripper/joint_pos, action.ee_pos/ee_rot/gripper,
observation.images.{gripper_cam.rgb,left_cam.left,right_cam.left}) --
e.g. `python -c "import pandas as pd; print(pd.read_parquet('<dataset>/data/chunk-000/episode_000000.parquet').columns.tolist())"`
-- since that assumption hasn't been independently verified for this
dataset. If any column names differ, adjust process_episode() below and
the "video" section of write_modality_json() to match before running.

Usage:
    python prepare_panda_sort_three_dataset.py --dataset-path <path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


STATE_DIM = 17  # eef_pose(9) + gripper(1) + joint_pos(7)
ACTION_DIM = 10  # eef_pose(9) + gripper(1)


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a batch of [w, x, y, z] quaternions to GR00T's 6D rotation format.

    Mirrors gr00t.data.state_action.pose.EndEffectorPose._matrix_to_rot6d:
    the first two rows of the rotation matrix, flattened.
    """
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]  # scipy expects x, y, z, w
    matrices = Rotation.from_quat(quat_xyzw).as_matrix()  # (N, 3, 3)
    return matrices[:, :2, :].reshape(-1, 6).astype(np.float32)


def build_eef_pose(pos_col: pd.Series, quat_col: pd.Series) -> np.ndarray:
    pos = np.stack(pos_col.to_numpy()).astype(np.float32)  # (N, 3)
    quat = np.stack(quat_col.to_numpy()).astype(np.float32)  # (N, 4)
    rot6d = quat_wxyz_to_rot6d(quat)  # (N, 6)
    return np.concatenate([pos, rot6d], axis=1).astype(np.float32)  # (N, 9)


def process_episode(df: pd.DataFrame) -> pd.DataFrame:
    state_eef = build_eef_pose(df["observation.proprio.ee_pos"], df["observation.proprio.ee_rot"])
    state_gripper = df["observation.proprio.gripper"].to_numpy(dtype=np.float32).reshape(-1, 1)
    state_joint = np.stack(df["observation.proprio.joint_pos"].to_numpy()).astype(np.float32)
    state = np.concatenate([state_eef, state_gripper, state_joint], axis=1)
    assert state.shape[1] == STATE_DIM, state.shape

    action_eef = build_eef_pose(df["action.ee_pos"], df["action.ee_rot"])
    action_gripper = df["action.gripper"].to_numpy(dtype=np.float32).reshape(-1, 1)
    action = np.concatenate([action_eef, action_gripper], axis=1)
    assert action.shape[1] == ACTION_DIM, action.shape

    df = df.copy()
    df["observation.state"] = list(state)
    df["action"] = list(action)
    return df


def update_info_json(dataset_path: Path) -> None:
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"]["observation.state"] = {"shape": [STATE_DIM], "dtype": "float32"}
    info["features"]["action"] = {"shape": [ACTION_DIM], "dtype": "float32"}

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


def write_modality_json(dataset_path: Path) -> None:
    modality = {
        "state": {
            "eef_pose": {"start": 0, "end": 9},
            "gripper": {"start": 9, "end": 10},
            "joint_pos": {"start": 10, "end": 17},
        },
        "action": {
            "eef_pose": {"start": 0, "end": 9},
            "gripper": {"start": 9, "end": 10},
        },
        "video": {
            "gripper_cam": {"original_key": "observation.images.gripper_cam.rgb"},
            "left_cam": {"original_key": "observation.images.left_cam.left"},
            "right_cam": {"original_key": "observation.images.right_cam.left"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    modality_path = dataset_path / "meta" / "modality.json"
    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    args = parser.parse_args()

    dataset_path: Path = args.dataset_path
    episode_paths = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    if not episode_paths:
        raise SystemExit(f"No episode parquet files found under {dataset_path / 'data'}")

    rows_updated = 0
    for episode_path in episode_paths:
        df = pd.read_parquet(episode_path)
        df = process_episode(df)
        df.to_parquet(episode_path, index=False)
        rows_updated += len(df)

    update_info_json(dataset_path)
    write_modality_json(dataset_path)

    print(f"Prepared dataset: {dataset_path}")
    print(f"Episodes processed: {len(episode_paths)}")
    print(f"Rows updated: {rows_updated}")
    print("Added columns:")
    print(
        "  observation.state (17,) = "
        "proprio.eef_pose[xyz+rot6d](9) + proprio.gripper(1) + proprio.joint_pos(7)"
    )
    print("  action (10,) = action.eef_pose[xyz+rot6d](9) + action.gripper(1)")
    print("Wrote meta/modality.json")
    print("Updated meta/info.json features")


if __name__ == "__main__":
    main()
