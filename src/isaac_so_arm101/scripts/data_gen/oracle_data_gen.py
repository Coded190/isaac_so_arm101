# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run VLA Inference in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# Parse arguments and boot Omniverse first
parser = argparse.ArgumentParser(description="VLA Inference for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate, default is 1.")
parser.add_argument("--task", type=str, default="None", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

# Suppress verbose Omniverse logging
import carb
carb.settings.get_settings().set_string("/log/level", "error")
simulation_app = app_launcher.app

# Standard library and ML imports
import os
import cv2
import torch
import numpy as np
import gymnasium as gym

# Project-specific imports
import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg

# Kinematic limits and timing constants.
# The action layout for this env is:
#   action[0:3]  -> relative position delta (meters) for the DifferentialIKController
#   action[3:6]  -> relative orientation delta (axis-angle, radians) — left at zero here
#   action[6]    -> absolute gripper joint target (radians), NOT a spray flag
ACTION_CLAMP = 0.05      # Max Cartesian delta per step (meters). Small → stable IK, larger → faster motion.
HOVER_OFFSET_Z = 0.10    # Vertical offset above the target for the approach waypoint.
SPRAY_DURATION = 60      # Sim steps to "spray" at the target (~2 s at 30 Hz).

# Gripper joint targets. Since action[6] is the absolute joint position, we open
# the gripper during spray (nozzle active) and close it otherwise. Adjust to
# match your end-effector's mechanical range if different.
GRIPPER_OPEN = 0.0       # "Spray on" joint position
GRIPPER_CLOSED = 0.0     # "Spray off" joint position (leave equal if gripper is unused)

# Path to the palm under which all leaf/trunk prims live.
PALM_ROOT_PATH = "/World/envs/env_0/Scene/Palm"

# Crunched rest configuration the arm snaps to at startup and after every reset.
# Joint order matches robot.data.joint_names for the SO-ARM101:
# [base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, gripper_moving]
REST_POSE_VALUES = [
    0.0,    # base_yaw         (0 = facing forward)
   -1.8,    # shoulder_pitch   (negative = tucked down)
    2.8,    # elbow_pitch      (folded up)
   -1.5,    # wrist_pitch      (curled in)
    0.0,    # wrist_roll
    0.0,    # gripper_moving
]

# Leaf culling parameters. Only leaves whose world Z is above (crown_z + offset)
# are candidates for removal. A positive offset culls only the very topmost leaves,
# preserving the bulk of the canopy for visual realism.
LEAF_CULL_Z_OFFSET = 0.03   # Start culling 3 cm above the crown centroid
LEAF_KEEP_RATIO = 0.8       # Keep 80% of the top leaves; only remove the highest 20%

# Spray target tuning, expressed as an offset from the computed crown centroid.
TARGET_OFFSET = np.array([0.0, 0.0, 0.20])


def _iter_leaf_prims(stage):
    """Yield every leaf_* and leaf_b_* prim under the palm root."""
    from pxr import UsdGeom
    palm = stage.GetPrimAtPath(PALM_ROOT_PATH)
    if not palm:
        return
    for child in palm.GetChildren():
        name = child.GetName()
        if name.startswith("leaf_") and UsdGeom.Xformable(child):
            yield child


def _leaf_world_positions(stage):
    """Return (prim, world_xyz) tuples for every leaf with a valid transform."""
    from pxr import UsdGeom
    out = []
    for prim in _iter_leaf_prims(stage):
        xf = UsdGeom.Xformable(prim)
        try:
            wp = xf.ComputeLocalToWorldTransform(0).ExtractTranslation()
            out.append((prim, np.array([wp[0], wp[1], wp[2]], dtype=np.float64)))
        except Exception:
            continue
    return out


def get_crown_centroid(stage):
    """
    Return the centroid of the palm's leaf cluster in world coordinates.
    This is the geometric center of the crown, used as the base spray target.
    """
    leaves = _leaf_world_positions(stage)
    if not leaves:
        print("[WARN] No leaves found under palm — using fallback crown centroid.")
        return np.array([0.0, 0.0, 5.0])
    positions = np.stack([p for _, p in leaves], axis=0)
    return positions.mean(axis=0)


def remove_top_leaves(stage, crown_z, z_threshold_offset=LEAF_CULL_Z_OFFSET, keep_ratio=LEAF_KEEP_RATIO):
    """
    Deactivate the topmost leaf prims to clear an approach corridor above the crown.
    Leaves below (crown_z + z_threshold_offset) are always preserved; above that,
    the highest-Z leaves are culled first until (1 - keep_ratio) of the candidate
    set has been removed.
    """
    cull_z = crown_z + z_threshold_offset
    leaves = _leaf_world_positions(stage)

    top_leaves = [(prim, pos[2]) for prim, pos in leaves if pos[2] > cull_z]
    top_leaves.sort(key=lambda x: -x[1])

    n_remove = int(len(top_leaves) * (1.0 - keep_ratio))
    for prim, _ in top_leaves[:n_remove]:
        prim.SetActive(False)

    print(f"[INFO] Culled {n_remove}/{len(top_leaves)} top leaves above z={cull_z:.3f} "
          f"(total leaves seen: {len(leaves)})")


def spawn_target_marker(stage, position_world, radius=0.04, color=(1.0, 0.0, 0.0)):
    """
    Place a small red sphere at the given world coordinate to visualize the spray
    target. Non-colliding, non-physical — pure visual aid. Idempotent.
    """
    from pxr import UsdGeom, Gf, Sdf
    path = "/World/debug_target_marker"
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)

    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(path))
    sphere.GetRadiusAttr().Set(radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(position_world[0]),
                                          float(position_world[1]),
                                          float(position_world[2])))
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    print(f"[DEBUG] Target marker at {np.asarray(position_world).round(3)}")


def set_rest_pose(env, rest_pose_tensor):
    """Snap the arm to its rest configuration with zero joint velocity."""
    robot = env.unwrapped.scene["robot"]
    zero_vel = torch.zeros_like(rest_pose_tensor)
    robot.write_joint_state_to_sim(rest_pose_tensor, zero_vel)


def get_deterministic_target(stage, offset=TARGET_OFFSET):
    """
    Compute the spray target from the current palm crown centroid plus a
    configurable offset. Re-reading the stage means the target tracks the
    palm if it moves (e.g. under domain randomization).
    """
    return get_crown_centroid(stage) + offset


class SprayOracle:
    """
    Finite State Machine (FSM) producing expert actions for VLA dataset collection.
    Actions are Cartesian position deltas for a DifferentialIKController in
    relative mode — each step commands a small displacement from the current EE
    pose, clamped by ACTION_CLAMP.

    States: 0 (Approach Hover), 1 (Descend), 2 (Spray), 3 (Retract), 4 (Complete)
    """
    POSITION_THRESHOLD = 0.05  # Distance tolerance (meters) to advance state
    MAX_STATE_STEPS = 400      # Failsafe bound so a stuck state cannot hang the episode.

    def __init__(self):
        self.state = 0
        self.spray_counter = 0
        self.state_steps = 0

    def _advance(self, next_state):
        self.state = next_state
        self.state_steps = 0

    def compute_action(self, ee_pos, spray_target):
        """
        Return a 7D action vector:
          [0:3] relative position delta toward the current waypoint
          [3:6] zero orientation delta (no reorient during spray)
          [6]   absolute gripper target (open during spray, closed otherwise)
        """
        action = np.zeros(7)
        action[6] = GRIPPER_CLOSED  # gripper default

        hover_pos = spray_target.copy()
        hover_pos[2] += HOVER_OFFSET_Z

        err_to_hover = hover_pos - ee_pos
        err_to_target = spray_target - ee_pos
        self.state_steps += 1

        if self.state == 0:
            # Approach hover waypoint above the target
            action[0:3] = err_to_hover
            if np.linalg.norm(err_to_hover) < self.POSITION_THRESHOLD or self.state_steps >= self.MAX_STATE_STEPS:
                self._advance(1)

        elif self.state == 1:
            # Descend to the target
            action[0:3] = err_to_target
            if np.linalg.norm(err_to_target) < self.POSITION_THRESHOLD or self.state_steps >= self.MAX_STATE_STEPS:
                self.spray_counter = SPRAY_DURATION
                self._advance(2)

        elif self.state == 2:
            # Hold position and "spray" by opening the gripper
            action[0:3] = err_to_target
            action[6] = GRIPPER_OPEN
            self.spray_counter -= 1
            if self.spray_counter <= 0:
                self._advance(3)

        elif self.state == 3:
            # Retract back to the hover waypoint
            action[0:3] = err_to_hover
            if np.linalg.norm(err_to_hover) < self.POSITION_THRESHOLD or self.state_steps >= self.MAX_STATE_STEPS:
                self._advance(4)

        return action


def main():
    print("[INFO]: Setting up Isaac Lab Environment...")
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    obs, _ = env.reset()

    # Cache the sim device and robot handle; used throughout the main loop.
    sim_device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]

    # Acquire the USD stage — needed for leaf enumeration, target calc, marker.
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # ---------------------------------------------------------
    # STATIC BASE POSITIONING
    # Calculate a specific shifted coordinate exactly ONCE
    # to prevent cumulative positional drift on environment resets.
    # ---------------------------------------------------------
    shifted_pos = robot.data.root_pos_w.clone()
    shifted_pos[:, 0] += 0.2032 # +X (in towards tree): 8 inches
    shifted_pos[:, 2] += 0.127  # +Z (upwards): 5 inches
    shifted_quat = robot.data.root_quat_w.clone()

    absolute_shifted_pose = torch.cat([shifted_pos, shifted_quat], dim=-1)
    robot.write_root_pose_to_sim(absolute_shifted_pose)
    # ---------------------------------------------------------

    # Dump joint metadata so REST_POSE_VALUES can be verified against actual DOF order.
    print(f"[INFO] Joint names: {robot.data.joint_names}")
    print(f"[INFO] Default joint pos: {robot.data.default_joint_pos[0].cpu().numpy()}")

    # Build the rest-pose tensor on the sim device, padding/truncating to match the
    # robot's real DOF count so a mis-sized REST_POSE_VALUES list can't crash the sim.
    num_dof = robot.data.default_joint_pos.shape[-1]
    rest_vals = REST_POSE_VALUES[:num_dof]
    if len(rest_vals) < num_dof:
        rest_vals = rest_vals + [0.0] * (num_dof - len(rest_vals))
    REST_POSE = torch.tensor(rest_vals, device=sim_device, dtype=torch.float32).unsqueeze(0)

    # Compute the crown centroid once from the pristine stage.
    crown_centroid = get_crown_centroid(stage)
    print(f"[INFO] Palm crown centroid: {crown_centroid.round(3)}")

    # Cull only the very topmost leaves to clear an approach corridor.
    remove_top_leaves(stage, crown_z=crown_centroid[2])

    # Initialize the arm in the crunched rest configuration before the first episode.
    set_rest_pose(env, REST_POSE)
    
    # Hold the rest pose for a few seconds so you can see it before motion starts.
    import time
    print("[INFO] Holding rest pose for 3 seconds...")
    for _ in range(90):  # ~3 seconds at 30 Hz — step the sim with a zero action to render
        zero_action = torch.zeros((1, 7), dtype=torch.float32, device=sim_device)
        env.step(zero_action)
    time.sleep(0.5)

    # Dataset directory initialization
    dataset_dir = "vla_tree_dataset"
    img_dir = os.path.join(dataset_dir, "images")
    act_dir = os.path.join(dataset_dir, "actions")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(act_dir, exist_ok=True)
    print(f"[INFO]: Saving images to ./{img_dir}/ and actions to ./{act_dir}/")

    # Initialize the trajectory planner and visualize where it's aiming.
    oracle = SprayOracle()
    current_spray_target = get_deterministic_target(stage)
    spawn_target_marker(stage, current_spray_target)

    # Resolve the gripper body index once to avoid repeated name lookups in the hot loop.
    gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper")
    gripper_idx = gripper_indices[0]

    step = 0
    global_record_step = 0

    print("[INFO]: Starting Oracle Data Generation Loop...")
    while simulation_app.is_running():
        # Pull the absolute world coordinate of the end-effector
        ee_position = env.unwrapped.scene["robot"].data.body_pos_w[0, gripper_idx].cpu().numpy()

        # The oracle returns a relative Cartesian delta toward the current waypoint.
        # The IK controller is configured with use_relative_mode=True, so we send
        # this delta directly — no conversion to absolute coordinates.
        action = oracle.compute_action(ee_position, current_spray_target)
        action[:3] = np.clip(action[:3], -ACTION_CLAMP, ACTION_CLAMP)

        # Upload to the sim device.
        action_tensor = torch.tensor(action, dtype=torch.float32, device=sim_device).unsqueeze(0)

        # Record visual and state data while the episode is active
        if oracle.state < 4:
            img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]
            img_numpy = img_tensor.cpu().numpy()

            if img_numpy.shape[-1] == 4:
                img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

            step_str = str(global_record_step).zfill(5)

            cv2.imwrite(os.path.join(img_dir, f"frame_{step_str}.jpg"), img_bgr)
            np.save(os.path.join(act_dir, f"action_{step_str}.npy"), action)
            global_record_step += 1

            if global_record_step >= 25000:
                print(f"[INFO]: Successfully collected {global_record_step} frames! Shutting down.")
                simulation_app.close()
                break

        # Telemetry heartbeat
        if step % 50 == 0:
            print(f"[STEP {step:05d}] state={oracle.state} | "
                  f"ee={ee_position.round(3)} | target={current_spray_target.round(3)} | "
                  f"action={action_tensor[0].cpu().numpy().round(3)}")

        # Advance physics engine
        env.step(action_tensor)
        step += 1

        # Episode completion reset
        if oracle.state == 4 or oracle.state_steps >= oracle.MAX_STATE_STEPS:
            env.reset()

            # Snap the base back to the exact pre-calculated shifted coordinate
            env.unwrapped.scene["robot"].write_root_pose_to_sim(absolute_shifted_pose)

            # Restore the crunched joint configuration so every episode starts
            # from a consistent rest state rather than wherever the arm finished.
            set_rest_pose(env, REST_POSE)

            oracle = SprayOracle()
            current_spray_target = get_deterministic_target(stage)
            step = 0


if __name__ == "__main__":
    main()
