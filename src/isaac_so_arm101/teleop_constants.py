"""Shared teleop action layout for PingTi (keyboard now, SO-ARM101 later).

Device contract used by ``teleop_keyboard.py`` and the teleop env::

    advance() -> Tensor[7]  # dx, dy, dz, droll, dpitch, dyaw, gripper

Gripper from Isaac Lab ``Se3Keyboard`` is binary: ``+1`` open, ``-1`` close.
"""

from __future__ import annotations

PINGTI_ARM_JOINTS = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_roll",
)
PINGTI_GRIPPER_JOINT = "gripper_moving"
PINGTI_EE_BODY = "moving_gripper"
PINGTI_JOINTS = (*PINGTI_ARM_JOINTS, PINGTI_GRIPPER_JOINT)

# URDF limits for gripper_moving (radians).
GRIPPER_CLOSED_RAD = 0.0
GRIPPER_OPEN_RAD = 1.5708

SE3_ACTION_DIM = 7
SE3_GRIPPER_OPEN_CMD = 1.0
SE3_GRIPPER_CLOSE_CMD = -1.0

# Default table-scene spawn (not the palm-garden coordinates on PING_TI_CFG).
PINGTI_TABLE_POS = (0.0, 0.0, 0.0)
PINGTI_TABLE_ROT = (1.0, 0.0, 0.0, 0.0)

# Palm-garden teleop spawn (not written into the garden USD).
# Lab 3 ArticulationCfg.InitialStateCfg.rot is (x, y, z, w). Euler XYZ 0,0,0
# is identity (0, 0, 0, 1). Kit P bake 2026-09-19:
#   PINGTI_PALM_POS=(1.00000, 1.52000, 4.65058)
#   PINGTI_PALM_ROT=(0.00000, 0.00000, 0.00000, 1.00000)  # Lab3 xyzw
# USD xformOp:orient for that pose is Gf.Quatd(w=1, x=0, y=0, z=0).
PINGTI_PALM_POS = (1.0, 1.52, 4.65058)
PINGTI_PALM_ROT = (0.0, 0.0, 0.0, 1.0)


def gripper_joint_target_from_se3(command: float) -> float:
    """Map Se3Keyboard gripper (+1 open / -1 close) to a PingTi joint target."""
    if command < 0.0:
        return GRIPPER_CLOSED_RAD
    return GRIPPER_OPEN_RAD
