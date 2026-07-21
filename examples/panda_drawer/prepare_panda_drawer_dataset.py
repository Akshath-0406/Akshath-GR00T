#!/usr/bin/env python3
"""Prepare panda_drawer_many_ee_fullres for GR00T N1.7 (NEW_EMBODIMENT).

GR00T's LeRobotEpisodeLoader expects a single concatenated `observation.state`
and `action` column per row (see getting_started/data_preparation.md), sliced
back into named fields via meta/modality.json. This dataset instead stores
each modality as its own column (observation.proprio.ee_pos, action.ee_rot,
etc.), so this script builds the concatenated columns GR00T needs without
touching the original columns.

End-effector fields (ee_pos + ee_rot) are packed into a single 9-D
[xyz, rot6d] field ("eef_pose"), matching gr00t.data.state_action.pose.
EndEffectorPose.xyz_rot6d / EndEffectorPose.from_action_format exactly:
translation first (3), then the first two rows of the rotation matrix
flattened (6). GR00T's EEF action math (EndEffectorActionChunk.from_array,
RelativeActionLoader in gr00t/data/stats.py) requires position and rotation
combined into one field like this -- not as separate keys -- because the
relative-pose transform operates on the pair as a single homogeneous
transformation.

Quaternion order: this dataset's ee_rot is [w, x, y, z] (scalar-first).
Verified against imikit's control/ik.py (`pin.Quaternion` -> [q.w, q.x, q.y,
q.z]), which is identical across every branch of that repo's full history
(main, factr, intervention_guidance, molmoact-inference, simple_guidance,
vla_uq_eval, xvla-ee-inference all carry the same line; the two branches
without the file don't use this kinematics solver at all).

Per instructions from the dataset owner (kitalr):
  - action = ee_pos + ee_rot + gripper (action.joint_pos is ignored)
  - observation.target.* fields are ignored
  - state fields are up to GR00T's own conventions; this script follows
    NVIDIA's own DROID embodiment config (gr00t/configs/data/
    embodiment_configs.py, "oxe_droid_relative_eef_relative_joint"), which
    uses eef pose + gripper + joint_position together for state.

Usage:
    python prepare_panda_drawer_dataset.py --dataset-path <path>
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
