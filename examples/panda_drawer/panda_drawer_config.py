"""Modality config for the panda_drawer_many_ee_fullres dataset (NEW_EMBODIMENT).

Mirrors NVIDIA's own DROID config (gr00t/configs/data/embodiment_configs.py,
"oxe_droid_relative_eef_relative_joint") for the "state" side: end-effector
pose + gripper + joint positions together. The "action" side follows the
dataset owner's (kitalr) instruction to use ee_pos + ee_rot + gripper only
(action.joint_pos is intentionally excluded).

See examples/panda_drawer/prepare_panda_drawer_dataset.py for how
meta/modality.json's "eef_pose" field (9-D: xyz + 6D rotation) is built from
the dataset's raw ee_pos/ee_rot columns.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


panda_drawer_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["gripper_cam", "left_cam", "right_cam"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_pose", "gripper", "joint_pos"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["eef_pose", "gripper"],
        action_configs=[
            # eef_pose: relative to the current end-effector pose (N1.7's
            # default recommendation for cross-embodiment generalization).
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_pose",
            ),
            # gripper: absolute target (binary open/close works better absolute).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="gripper",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(panda_drawer_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
