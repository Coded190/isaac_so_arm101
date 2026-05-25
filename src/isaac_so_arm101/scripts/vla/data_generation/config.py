# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration constants for VLA data generation."""

import numpy as np


# ─── Kinematic / FSM constants ────────────────────────────────────────────
ACTION_CLAMP = 0.25      # recording-style slow motion (was 0.5; smoother per-step deltas)
POSITION_GAIN = 0.35     # recording-style slow motion (was 0.75)
SPRAY_DURATION = 60

REST_POSE_VALUES = [
    0.0,                       # base_yaw
    float(np.deg2rad(-45.0)),  # shoulder_pitch  — chain sums to -90° world pitch:
    float(np.deg2rad(-45.0)),  # elbow_pitch     — gripper points straight down at rest
    -0.2,                      # wrist_pitch     — slight back-tilt (~11°) toward observer
    0.0,                       # wrist_roll
    0.0,                       # gripper_moving
]

# Pose the arm transitions to during settle (state 2) and holds during
# spray/hold (states 3/4). Shoulder pulled further back, elbow folded down,
# wrist at lower limit so the gripper looks straight down at the crown.
SPRAY_POSE_VALUES = [
    0.0,                       # base_yaw (locked)
    float(np.deg2rad(-70.0)),  # shoulder_pitch  — pulled back hard
    float(np.deg2rad(-78.0)),  # elbow_pitch     — folded down
    -1.5708,                   # wrist_pitch     — at URDF lower limit (looks down)
    0.0,                       # wrist_roll
    0.0,                       # gripper_moving
]

LEAF_CULL_Z_OFFSET = 0.03
# When an episode culls leaves, the keep-ratio is sampled uniformly from
# [LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX]. Lower keep-ratio = more leaves
# removed: 0.5 culls 50% of the top leaves, 0.8 culls 20%.
LEAF_KEEP_RATIO_MIN = 0.5
LEAF_KEEP_RATIO_MAX = 0.8
HOVER_OFFSET = np.array([0.0, 0.0, 0.35])  # back to original reachable height
# Meters to pull the hover target from the crown XY back toward the robot XY.
# 0.0 = directly above crown (used to overshoot past). Increase to land short of crown.
HOVER_PULLBACK_M = 0.0

# ─── Dataset metadata ─────────────────────────────────────────────────────
TASK_DESCRIPTION = "Move end effector above palm crown, angle end effector downward, and hold while end effector is spraying."
MAX_TOTAL_SAVED_FRAMES = 25000  # cap across all envs; script exits after

DOWN_QUAT = np.array([0.2588, 0.0, -0.9659, 0.0])
# Bumped 0.2 → 0.5 so the wrist rotates toward DOWN_QUAT ~2.5× faster
# per step. With ANGLE_RANDOM_RANGE shrunk to ±45° the IK has easy
# reach, so the orientation command no longer needs to be conservative.
ORIENTATION_CLAMP = 1.0   # aggressive wrist-down correction (only fires in spray/hold; EE already parked)

# Force a re-randomization only when the FSM has already given up
# (state 5 = fail_hold) AND the EE is sitting close to a leaf. This
# guards against the auto-reset firing during a normal approach where
# the EE passes through the canopy on its way to the hover target.
LEAF_STUCK_DISTANCE = 0.05
LEAF_STUCK_STEPS = 30
LEAF_STUCK_FAIL_STATE = 5

# Verbose console output: FSM state transitions and stuck-reset events.
DEBUG_VERBOSE = True

# Recording camera follows the robot. Each reset picks a random left/right
# side of the robot's "robot → tree" axis and points the camera at the
# midpoint, lifted to frame both the base and the canopy.
CAMERA_FOLLOWS_ROBOT = True
CAMERA_LATERAL_OFFSET = 1.10  # meters out to the side of the robot
CAMERA_HEIGHT_OFFSET = 0.75   # meters above the base height
# Look-at point sits at the midpoint XY between robot and crown, lifted
# this much above the base height to favor the canopy in the framing.
CAMERA_TARGET_LIFT = 0.20

# Per-episode root randomization. The robot is placed at a random angle on
# a circle centered on the tree, keeping its distance from the tree (and
# its world-Z height) constant. Yaw is set so the robot faces the tree.
ANGLE_RANDOM_RANGE = float(np.deg2rad(45.0))  # ±45° forward arc.
# Positive value moves the robot closer to the tree (subtracted from the
# default radius). NEGATIVE pushes it further away. 0.0 keeps default.
TREE_INWARD_OFFSET = 0.50  # positive = pull robot CLOSER to tree (reachable distance)
# Clamp the final spawn radius to this range (meters)
MIN_TREE_RADIUS = 0.55  # Don't spawn too close (arm can reach better from distance)
MAX_TREE_RADIUS = 0.60  # Don't spawn too far (arm needs reasonable reach)
# Reject any placement whose XY position lands within this many meters of
# any palm leaf — the base would otherwise spawn inside / through a leaf.
LEAF_CLEARANCE = 0.10
PLACEMENT_MAX_ATTEMPTS = 15
PALM_ROOT_NAME = "palm_tree_crown"

# HDRI lighting setup
HDRI_FOLDER_PATH = "/home/cirplab/moore/isaac_data/palm_tree_models/blender/pretoria_gardens_4k/hdri"

# Palm tree randomization ranges
GIRTH_SCALE_RANGE = (0.85, 1.15)
HEIGHT_SCALE_RANGE = (0.85, 1.25)
CROWN_SHAFT_HEIGHT_RANGE = (0.7, 1.3)
CROWN_SHAFT_GIRTH_RANGE = (0.96, 1.04)
CANOPY_MULTIPLIER_RANGE = (0.8, 1.25)
LEAF_VARIANCE_RANGE = (0.9, 1.1)

# Joint control indices and limits
BASE_YAW_LIMIT = float(np.deg2rad(10.0))   # cap base_yaw joint to ±10° each step
WRIST_PITCH_REST = REST_POSE_VALUES[3]   # +1.2 rad — wrist up at rest
WRIST_PITCH_DOWN = -1.5708               # lower limit — EE points straight down
WRIST_LERP_STEPS = 90                    # slow wrist rotation (≈3s @ 30Hz; smoother trajectory)

# Palm physics tuning
PALM_LEAF_MASS = 0.05
JOINT_STIFFNESS_OVERRIDE = 0.5
JOINT_DAMPING_OVERRIDE = 0.1

# Lighting tuning
LIGHTING_SPECULAR_MULTIPLIER = 1.0
LIGHTING_DIFFUSE_BOOST = 10.0
AMBIENT_FILL_INTENSITY = 500.0
HDRI_INTENSITY_RANGE = (600.0, 1200.0)

# FSM state names
FSM_STATE_NAMES = {
    0: "approach_wp",
    1: "approach_hover",
    2: "settle",
    3: "spray",
    4: "success_hold",
    5: "fail_hold"
}

# FSM configuration
FSM_POSITION_THRESHOLD = 0.20
FSM_MAX_STATE_STEPS = 400

# Episode configuration
EPISODE_LENGTH_S = 22.0

# Noise filter patterns
NOISE_FILTER_DROP_PATTERNS = [
    r"PxJoint::setActors",
    r"CreateJoint - cannot create",
    r"FabricManager::initializePointInstancer",
    r"primvars:displayColor:indices not found",
    r"omni\.kit\.notification_manager.*PhysX",
    r"omni\.kit\.notification_manager.*Physics USD Load",
    r"omni\.hydra.*update topology",
    r"gpu::unstable::IMemoryBudgetManagerFactory",
]
