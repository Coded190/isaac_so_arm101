"""Shared teleop action layout for PingTi.

Keyboard (``--teleop_device keyboard``) uses Isaac Lab ``Se3Keyboard``::

    advance() -> Tensor[7]  # dx, dy, dz, droll, dpitch, dyaw, gripper

SO-ARM101 leader (``--teleop_device so101leader``) uses 6-D joint position::

    Tensor[6]  # base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, gripper_moving

Gripper from Isaac Lab ``Se3Keyboard`` is binary: ``+1`` open, ``-1`` close.
Lab AppLauncher already owns ``--device`` (cuda/cpu); hardware is ``--teleop_device``.
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
JOINT_POS_ACTION_DIM = 6
SE3_GRIPPER_OPEN_CMD = 1.0
SE3_GRIPPER_CLOSE_CMD = -1.0

# SO-ARM101 leader motors (LeRobot / Feetech ids 1-6) → PingTi URDF joints.
SO101_LEADER_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_TO_PINGTI = {
    "shoulder_pan": "base_yaw",
    "shoulder_lift": "shoulder_pitch",
    "elbow_flex": "elbow_pitch",
    "wrist_flex": "wrist_pitch",
    "wrist_roll": "wrist_roll",
    "gripper": "gripper_moving",
}
SO101_LEADER_ARM_RANGE = (-100.0, 100.0)
SO101_LEADER_GRIPPER_RANGE = (0.0, 100.0)
# Exact limits from PingTi_Arm_5DOF_v4_copy.urdf (radians).
PINGTI_JOINT_LIMITS_RAD = {
    "base_yaw": (-1.5708, 1.5708),
    "shoulder_pitch": (-1.69313, 1.44846),
    "elbow_pitch": (-1.44846, 1.69313),
    "wrist_pitch": (-1.5708, 1.5708),
    "wrist_roll": (-3.14159, 3.14159),
    "gripper_moving": (-0.0872665, 1.5708),
}
# Hardware motor count per URDF joint. PingTi has 8 Feetech motors, SO101 has 6.
# The extra two are mechanically coupled dual-drives (shoulder_pitch 2x STS3250,
# elbow_pitch 2x STS3215). Sim / URDF expose one joint each, so the leader still
# maps 6 motors → 6 joints. A real PingTi follower expands those 6 commands to
# 8 Goal_Position writes (same target on both motors of a dual joint).
PINGTI_JOINT_MOTOR_COUNTS = {
    "base_yaw": 1,
    "shoulder_pitch": 2,
    "elbow_pitch": 2,
    "wrist_pitch": 1,
    "wrist_roll": 1,
    "gripper_moving": 1,
}
PINGTI_PHYSICAL_MOTOR_COUNT = 8
SO101_PHYSICAL_MOTOR_COUNT = 6

# Real PingTi follower Feetech ids (base → gripper). Dual-drive joints share one
# commanded angle; both motors get that same Goal_Position. Override ids only if
# the bus was set up differently (`ISAAC_SO_ARM101_PINGTI_PORT`).
PINGTI_FOLLOWER_MOTOR_IDS = {
    "base_yaw": 1,
    "shoulder_pitch_1": 2,
    "shoulder_pitch_2": 3,
    "elbow_pitch_1": 4,
    "elbow_pitch_2": 5,
    "wrist_pitch": 6,
    "wrist_roll": 7,
    "gripper_moving": 8,
}
PINGTI_FOLLOWER_MOTORS = tuple(PINGTI_FOLLOWER_MOTOR_IDS)
# LeRobot Feetech handshake pings model numbers (sts3215=777, sts3250=2825).
# Shoulder duals are STS3250; the rest are STS3215 (see PING_TI_CFG comments).
PINGTI_FOLLOWER_MOTOR_MODELS = {
    "base_yaw": "sts3215",
    "shoulder_pitch_1": "sts3250",
    "shoulder_pitch_2": "sts3250",
    "elbow_pitch_1": "sts3215",
    "elbow_pitch_2": "sts3215",
    "wrist_pitch": "sts3215",
    "wrist_roll": "sts3215",
    "gripper_moving": "sts3215",
}
PINGTI_JOINT_TO_FOLLOWER_MOTORS = {
    "base_yaw": ("base_yaw",),
    "shoulder_pitch": ("shoulder_pitch_1", "shoulder_pitch_2"),
    "elbow_pitch": ("elbow_pitch_1", "elbow_pitch_2"),
    "wrist_pitch": ("wrist_pitch",),
    "wrist_roll": ("wrist_roll",),
    "gripper_moving": ("gripper_moving",),
}
PINGTI_DUAL_JOINTS = ("shoulder_pitch", "elbow_pitch")

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
