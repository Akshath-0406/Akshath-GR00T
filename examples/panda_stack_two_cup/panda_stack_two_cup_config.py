"""Modality config for the panda_stack_two_cup_ee_fullres dataset (NEW_EMBODIMENT).

Identical to examples/panda_drawer/panda_drawer_config.py -- same Panda P4
rig, same imikit recording pipeline, same state/action layout produced by
examples/panda_stack_two_cup/prepare_panda_stack_two_cup_dataset.py. Kept as
its own file (rather than importing panda_drawer_config) per this repo's
per-dataset example convention, and so the two can diverge later if this
task's data turns out to need different handling.
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


panda_stack_two_cup_config = {
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

register_modality_config(panda_stack_two_cup_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
