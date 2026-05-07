# Copyright (c) 2022-2025, The Isaac Lab Project Developers. 
# All rights reserved. 
# SPDX-License-Identifier: BSD-3-Clause 

"""Script to run VLA Inference in Isaac Lab with LeRobot Data Generation.""" 

import argparse 
from isaaclab.app import AppLauncher 

# Parse arguments and boot Omniverse first 
parser = argparse.ArgumentParser(description="VLA Inference for Isaac Lab.") 
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.") 
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate, default is 1.") 
parser.add_argument("--task", type=str, default="None", help="Name of the task.") 
parser.add_argument("--save_data", action="store_true", help="Enable saving data with LeRobotDataset. If not set, data is not saved.")
parser.add_argument("--top_leaf_cull_prob", type=float, default=0.5, help="Probability of culling the top leaves for a given episode.")
parser.add_argument("--dataset_root", type=str, default="outputs/vla_palm_dataset", help="Root folder where per-env LeRobot datasets are stored.")
AppLauncher.add_app_launcher_args(parser) 
args_cli = parser.parse_args() 

app_launcher = AppLauncher(args_cli) 

# Suppress verbose Omniverse logging 
import carb 
carb.settings.get_settings().set_string("/log/level", "error") 
simulation_app = app_launcher.app 

# Standard library and ML imports 
import os
import shutil
import cv2
import torch 
import numpy as np 
import gymnasium as gym 
from PIL import Image 

try: 
    from lerobot.datasets.lerobot_dataset import LeRobotDataset 
except ImportError: 
    print("[ERROR]: 'lerobot' package not found. Please install it in your Isaac Lab environment using 'pip install lerobot'.") 
    simulation_app.close() 
    exit(1) 

# Project-specific imports 
import isaac_so_arm101.tasks 
from isaaclab_tasks.utils import parse_env_cfg 

# Kinematic limits and timing constants. 
# The action layout for this env is: 
# action[0:3] -> relative position delta (meters) for the DifferentialIKController 
# action[3:6] -> relative orientation delta (axis-angle, radians) — left at zero here 
# action[6] -> absolute gripper joint target (radians), NOT a spray flag 
ACTION_CLAMP = 0.05 # Max Cartesian delta per step (meters). Small → stable IK, larger → faster motion. 
POSITION_GAIN = 0.75 # Proportional gain for Cartesian position tracking. 
HOVER_OFFSET_Z = 0.13 # Vertical offset above the target for the approach waypoint.

# End-effector orientation target: gripper "nozzle" pointing straight down (-world_Z).
# Quaternion is [w, x, y, z]. The right value depends on which local axis of the
# gripper body is the nozzle direction. Common options to try:
#   [0.7071, 0.0, 0.7071, 0.0]  -> 90 deg around Y (local +X -> -world_Z)  [default]
#   [0.7071, 0.7071, 0.0, 0.0]  -> 90 deg around X (local +Y -> -world_Z)
#   [0.0, 1.0, 0.0, 0.0]        -> 180 deg around X (flips Z, points local +Z down)
#   [0.0, 0.0, 1.0, 0.0]        -> 180 deg around Y
DOWN_QUAT = np.array([0.2588, 0.9659, 0.0, 0.0])
ORIENTATION_CLAMP = 0.2 # Max rotation delta per step (radians), matches ACTION_CLAMP spirit.

SPRAY_DURATION = 60 # Sim steps to "spray" at the target (~2 s at 30 Hz). 

# Gripper joint targets. Since action[6] is the absolute joint position, we open 
# the gripper during spray (nozzle active) and close it otherwise. Adjust to 
# match your end-effector's mechanical range if different. 
GRIPPER_OPEN = 0.0 # "Spray on" joint position 
GRIPPER_CLOSED = 0.0 # "Spray off" joint position (leave equal if gripper is unused) 

# Crunched rest configuration the arm snaps to at startup and after every reset. 
# Joint order matches robot.data.joint_names for the SO-ARM101: 
# [base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, gripper_moving] 
REST_POSE_VALUES = [ 
    0.0,                           # base_yaw 
    float(np.deg2rad(48.5)),       # shoulder_pitch: 48.5 degrees
    float(np.deg2rad(-58.6)),      # elbow_pitch: -58.6 degrees
    1.2,                           # wrist_pitch: (keep as radians or change to np.deg2rad(68.7))
    0.0,                           # wrist_roll 
    0.0,                           # gripper_moving 
]

# Leaf culling parameters. Only leaves whose world Z is above (crown_z + offset) 
# are candidates for removal. A positive offset culls only the very topmost leaves, 
# preserving the bulk of the canopy for visual realism. 
LEAF_CULL_Z_OFFSET = 0.03 # Start culling 3 cm above the crown centroid 
LEAF_KEEP_RATIO = 0.8 # Keep 80% of the top leaves; only remove the highest 20% 

# Spray target tuning, expressed as an offset from the computed crown centroid. 
HOVER_OFFSET = np.array([0.0, 0.0, 0.35]) # Crown centroid + 35cm Z-height
TASK_DESCRIPTION = "Move end effector above palm crown, angle end effector downward, and hold while end effector is spraying."
MAX_TOTAL_SAVED_FRAMES = 25000


def get_palm_root_path(env_id):
    return f"/World/envs/env_{env_id}/Scene/Palm"


def _iter_leaf_prims(stage, palm_root_path): 
    """Yield every leaf_* and leaf_b_* prim under the palm root.""" 
    from pxr import UsdGeom 
    palm = stage.GetPrimAtPath(palm_root_path) 
    if not palm: 
        return 
    for child in palm.GetChildren(): 
        name = child.GetName() 
        if name.startswith("leaf_") and UsdGeom.Xformable(child): 
            yield child 


def _leaf_world_positions(stage, palm_root_path): 
    """Return (prim, world_xyz) tuples for every leaf with a valid transform.""" 
    from pxr import UsdGeom 
    out = [] 
    for prim in _iter_leaf_prims(stage, palm_root_path): 
        xf = UsdGeom.Xformable(prim) 
        try: 
            wp = xf.ComputeLocalToWorldTransform(0).ExtractTranslation() 
            out.append((prim, np.array([wp[0], wp[1], wp[2]], dtype=np.float64))) 
        except Exception: 
            continue 
    return out 

def set_leaf_prims_active(stage, palm_root_path, active=True):
    """Set all leaf prims under a palm root active or inactive."""
    for prim in _iter_leaf_prims(stage, palm_root_path):
        prim.SetActive(active)


def get_crown_centroid(stage, palm_root_path): 
    """ 
    Return the centroid of the palm's leaf cluster in world coordinates. 
    This is the geometric center of the crown, used as the base spray target. 
    """ 
    leaves = _leaf_world_positions(stage, palm_root_path) 
    if not leaves: 
        print("[WARN] No leaves found under palm — using fallback crown centroid.") 
        return np.array([0.0, 0.0, 5.0]) 
    positions = np.stack([p for _, p in leaves], axis=0) 
    return positions.mean(axis=0) 


def remove_top_leaves(stage, palm_root_path, crown_z, z_threshold_offset=LEAF_CULL_Z_OFFSET, keep_ratio=LEAF_KEEP_RATIO): 
    """ 
    Deactivate the topmost leaf prims to clear an approach corridor above the crown. 
    Leaves below (crown_z + z_threshold_offset) are always preserved; above that, 
    the highest-Z leaves are culled first until (1 - keep_ratio) of the candidate 
    set has been removed. 
    """ 
    cull_z = crown_z + z_threshold_offset 
    leaves = _leaf_world_positions(stage, palm_root_path) 

    top_leaves = [(prim, pos[2]) for prim, pos in leaves if pos[2] > cull_z] 
    top_leaves.sort(key=lambda x: -x[1]) 

    n_remove = int(len(top_leaves) * (1.0 - keep_ratio)) 
    for prim, _ in top_leaves[:n_remove]: 
        prim.SetActive(False) 

    print(f"[INFO] Culled {n_remove}/{len(top_leaves)} top leaves above z={cull_z:.3f} " 
          f"(total leaves seen: {len(leaves)})") 


def spawn_target_marker(stage, position_world, marker_path, radius=0.04, color=(1.0, 0.0, 0.0)): 
    """ 
    Place a small red sphere at the given world coordinate to visualize the spray 
    target. Non-colliding, non-physical — pure visual aid. Idempotent. 
    """ 
    from pxr import UsdGeom, Gf, Sdf 
    if stage.GetPrimAtPath(marker_path): 
        stage.RemovePrim(marker_path) 

    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(marker_path)) 
    sphere.GetRadiusAttr().Set(radius) 
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(position_world[0]), 
                                         float(position_world[1]), 
                                         float(position_world[2]))) 
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)]) 
    print(f"[DEBUG] Target marker at {np.asarray(position_world).round(3)} -> {marker_path}") 


def spawn_target_markers(stage, target_positions, env_ids=None, marker_type="spray", color=(1.0, 0.0, 0.0)):
    """Spawn one target marker per environment for a specific type."""
    if env_ids is None:
        env_ids = range(target_positions.shape[0])
    for env_id in env_ids:
        # Include the marker_type in the path so they don't overwrite each other
        marker_path = f"/World/debug_target_markers/{marker_type}/env_{env_id}"
        spawn_target_marker(stage, target_positions[env_id], marker_path=marker_path, color=color)


def set_rest_pose(env, rest_pose_tensor, env_ids=None): 
    """Snap the arm to its rest configuration with zero joint velocity.""" 
    robot = env.unwrapped.scene["robot"] 
    if env_ids is None:
        joint_pos = rest_pose_tensor.expand(env.unwrapped.num_envs, -1)
        zero_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, zero_vel)
        return

    env_ids_t = torch.as_tensor(env_ids, device=rest_pose_tensor.device, dtype=torch.long)
    joint_pos = rest_pose_tensor.expand(env_ids_t.shape[0], -1)
    zero_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, zero_vel, env_ids=env_ids_t)


def get_deterministic_target(stage, palm_root_path, offset=HOVER_OFFSET): 
    """ 
    Compute the spray target from the current palm crown centroid plus a 
    configurable offset. Re-reading the stage means the target tracks the 
    palm if it moves (e.g. under domain randomization). 
    """ 
    return get_crown_centroid(stage, palm_root_path) + offset 


def prepare_episode_targets(stage, palm_root_paths, episode_rng, cull_prob, env_ids=None):
    """Prepare per-env crown targets and optional top-leaf culling decisions."""
    if env_ids is None:
        env_ids = list(range(len(palm_root_paths)))
    targets = []
    cull_count = 0
    for env_id in env_ids:
        palm_root_path = palm_root_paths[env_id]
        set_leaf_prims_active(stage, palm_root_path, active=True)
        crown_centroid = get_crown_centroid(stage, palm_root_path)
        should_cull = episode_rng.random() < cull_prob
        if should_cull:
            remove_top_leaves(stage, palm_root_path, crown_z=crown_centroid[2])
            cull_count += 1
        targets.append(get_deterministic_target(stage, palm_root_path))
    print(f"[INFO] Episode prep complete: culled top leaves in {cull_count}/{len(env_ids)} envs.")
    return np.asarray(targets, dtype=np.float32)

DOWN_QUAT = np.array([0.2588, 0.0, -0.9659, 0.0])
ORIENTATION_CLAMP = 0.2

def _quat_multiply(q1, q2):
    """Hamilton product of two quaternions in [w, x, y, z] order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _quat_conjugate(q):
    """Conjugate of a unit quaternion [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def _quat_to_axis_angle(q):
    """Convert a unit quaternion [w, x, y, z] to an axis-angle 3-vector."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-9: return np.zeros(3)
    q = q / n
    if q[0] < 0.0: q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-6: return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = q[1:4] / sin_half
    return axis * angle

class SprayOracle: 
    """ 
    Finite State Machine (FSM) producing expert actions for VLA dataset collection. 
    States: 
      0 (Approach Waypoint), 1 (Approach Hover), 2 (Descend/Settle), 
      3 (Spray), 4 (Success Hold), 5 (Fail Hold) 
    """ 
    POSITION_THRESHOLD = 0.50 # Distance tolerance (meters) to advance state 
    MAX_STATE_STEPS = 530 

    def __init__(self): 
        self.state = 0 
        self.spray_counter = 0 
        self.state_steps = 0 
        self.completed = False 
        self.timed_out = False 
        self.approach_waypoint = None
        self.fail_pos = None # Anchor point to prevent drift upon failure

    def _advance(self, next_state): 
        self.state = next_state 
        self.state_steps = 0 

    @staticmethod
    def _cap_vector_norm(vector, max_norm):
        norm = np.linalg.norm(vector)
        if norm <= max_norm or norm == 0.0:
            return vector
        return vector * (max_norm / norm)

    def _position_command(self, error_vector):
        return self._cap_vector_norm(POSITION_GAIN * error_vector, ACTION_CLAMP)

    def compute_action(self, ee_pos, ee_quat, hover_target):
        action = np.zeros(7) 
        action[6] = 0.0 

        # Calculate waypoint exactly once at the start of the sequence
        if self.approach_waypoint is None:
            self.approach_waypoint = np.copy(hover_target)
            self.approach_waypoint[0] = (ee_pos[0] + hover_target[0]) / 2.0
            self.approach_waypoint[1] = (ee_pos[1] + hover_target[1]) / 2.0
        
        # Always maintain downward orientation to prevent wrist drift
        err_quat = _quat_multiply(DOWN_QUAT, _quat_conjugate(ee_quat))
        err_axis_angle = _quat_to_axis_angle(err_quat)
        action[3:6] = self._cap_vector_norm(err_axis_angle, ORIENTATION_CLAMP)
        
        self.state_steps += 1

        if self.state == 0: 
            err_to_waypoint = self.approach_waypoint - ee_pos 
            action[0:3] = self._position_command(err_to_waypoint) 
            if np.linalg.norm(err_to_waypoint) < self.POSITION_THRESHOLD: 
                self._advance(1) 
            elif self.state_steps >= self.MAX_STATE_STEPS: 
                self.timed_out = True 
                self.fail_pos = np.copy(ee_pos) # Lock the failed position
                self._advance(5) 

        elif self.state == 1: 
            err_to_hover = hover_target - ee_pos 
            action[0:3] = self._position_command(err_to_hover) 
            if np.linalg.norm(err_to_hover) < self.POSITION_THRESHOLD: 
                self._advance(2) 
            elif self.state_steps >= self.MAX_STATE_STEPS: 
                self.timed_out = True 
                self.fail_pos = np.copy(ee_pos) # Lock the failed position
                self._advance(5) 
                
        elif self.state == 2: 
            err_to_hover = hover_target - ee_pos 
            action[0:3] = self._position_command(err_to_hover) 
            if self.state_steps >= 30: 
                self.spray_counter = SPRAY_DURATION 
                self._advance(3)

        elif self.state == 3: 
            err_to_hover = hover_target - ee_pos 
            action[0:3] = self._position_command(err_to_hover) 
            self.spray_counter -= 1 
            if self.spray_counter <= 0: 
                self.completed = True 
                self._advance(4) # Move to Success Hold
                
        elif self.state == 4: 
            # Success Hold: actively track the hover target to fight gravity sag
            err_to_hover = hover_target - ee_pos 
            action[0:3] = self._position_command(err_to_hover) 

        elif self.state == 5:
            # Fail Hold: actively track the exact spot where it failed to fight gravity sag
            err_to_fail = self.fail_pos - ee_pos
            action[0:3] = self._position_command(err_to_fail)

        return action


def main(): 
    print("[INFO]: Setting up Isaac Lab Environment...") 
    env_cfg = parse_env_cfg( 
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric 
    ) 
    
    # Offset the default start position so Isaac Lab resets the robot here automatically
    default_pos = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.robot.init_state.pos = (
        default_pos[0] + 0.05,
        default_pos[1] - 0.179,
        default_pos[2] + 0.1524
    )
    
    # Map the REST_POSE_VALUES directly to the starting joint state
    env_cfg.scene.robot.init_state.joint_pos = {
        "base_yaw": REST_POSE_VALUES[0], 
        "shoulder_pitch": REST_POSE_VALUES[1], 
        "elbow_pitch": REST_POSE_VALUES[2], 
        "wrist_pitch": REST_POSE_VALUES[3], 
        "wrist_roll": REST_POSE_VALUES[4], 
        "gripper_moving": REST_POSE_VALUES[5]
    }
    
    # Extend the maximum episode length (e.g., to 15 seconds) so the 
    # SprayOracle FSM has enough time to reach the target before the environment resets.
    env_cfg.episode_length_s = 18.33
    
    env = gym.make(args_cli.task, cfg=env_cfg) 

    obs, _ = env.reset() 

    # Cache the sim device and robot handle; used throughout the main loop. 
    sim_device = env.unwrapped.device 
    robot = env.unwrapped.scene["robot"] 
    num_envs = env.unwrapped.num_envs

    # Acquire the USD stage — needed for leaf enumeration, target calc, marker. 
    import omni.usd 
    stage = omni.usd.get_context().get_stage() 
    palm_root_paths = [get_palm_root_path(env_id) for env_id in range(num_envs)]
    episode_rng = np.random.default_rng(getattr(args_cli, "seed", None)) 

    # --------------------------------------------------------- 
    # STATIC BASE POSITIONING 
    # Calculate a specific shifted coordinate exactly ONCE 
    # to prevent cumulative positional drift on environment resets. 
    # --------------------------------------------------------- 
    shifted_pos = robot.data.root_pos_w.clone() 
    shifted_pos[:, 0] += 0.2032 # +X (in towards tree): 8 inches 
    shifted_pos[:, 2] += 0.127 # +Z (upwards): 5 inches 
    shifted_quat = robot.data.root_quat_w.clone() 

    # absolute_shifted_pose = torch.cat([shifted_pos, shifted_quat], dim=-1) 
    # robot.write_root_pose_to_sim(absolute_shifted_pose) 
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
    current_hover_targets = np.zeros((num_envs, 3), dtype=np.float32)
    current_hover_targets[:] = prepare_episode_targets(
        stage=stage,
        palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
        cull_prob=args_cli.top_leaf_cull_prob,
    )
    print(f"[INFO] Prepared per-env crown targets for {num_envs} environments.")

    # Initialize the arm in the crunched rest configuration before the first episode. 
    # set_rest_pose(env, REST_POSE) 
    # Hold the rest pose for a few seconds so you can see it before motion starts. 
    import time 
    print("[INFO] Holding rest pose for 3 seconds...") 
    for _ in range(90): # ~3 seconds at 30 Hz — step the sim with a zero action to render 
        zero_action = torch.zeros((num_envs, 7), dtype=torch.float32, device=sim_device) 
        env.step(zero_action) 
    time.sleep(0.5) 

    # --------------------------------------------------------- 
    # LeRobot Dataset Initialization (Conditional)
    # --------------------------------------------------------- 
    datasets = None
    if args_cli.save_data:
        img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0] 
        IMG_H, IMG_W = img_tensor.shape[0], img_tensor.shape[1] 
        features = { 
            "env_id": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "observation.state": { 
                "dtype": "float32", 
                "shape": (num_dof,), 
                "names": [f"joint_{i}" for i in range(num_dof)] 
            }, 
            "observation.state.ee_pose": { 
                "dtype": "float32", 
                "shape": (7,), # pos (3) + quat (4) 
                "names": ["x", "y", "z", "qw", "qx", "qy", "qz"] 
            }, 
            "observation.images.wrist_camera": { 
                "dtype": "video", 
                "shape": (3, IMG_H, IMG_W), # Channels first 
                "names": ["c", "h", "w"] 
            }, 
            "action": { 
                "dtype": "float32", 
                "shape": (7,), 
                "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"] 
            } 
        } 
        datasets = []
        for env_id in range(num_envs):
            env_repo_id = f"local/vla_palm_dataset_env_{env_id:04d}"
            env_root = os.path.join(args_cli.dataset_root, f"env_{env_id:04d}")
            env_dataset = LeRobotDataset.create(
                repo_id=env_repo_id,
                root=env_root,
                fps=30,
                features=features,
            )
            datasets.append(env_dataset)
        
        abs_save_path = os.path.abspath(args_cli.dataset_root)
        print(f"[INFO]: Initialized {num_envs} per-env LeRobot datasets.")
        print(f"[INFO]: >>> SAVING DATA TO DIRECTORY: {abs_save_path} <<<")
    else:
        print("[INFO]: Running in TEST MODE. Data will NOT be saved.")

    oracles = [SprayOracle() for _ in range(num_envs)]
    
    # # Spawn Blue Hover Targets
    # spawn_target_markers(stage, current_hover_targets, marker_type="hover", color=(0.0, 0.0, 1.0))  
    
    # Resolve the gripper body index once to avoid repeated name lookups in the hot loop. 
    moving_gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper") 
    moving_gripper_idx = moving_gripper_indices[0] 

    step = 0 
    episode_frame_count = np.zeros(num_envs, dtype=np.int64)
    saved_frame_count = np.zeros(num_envs, dtype=np.int64)
    prev_dist_to_target = np.full(num_envs, np.nan, dtype=np.float64)

    print("[INFO]: Starting Oracle Data Generation Loop...") 
    while simulation_app.is_running(): 
        # State Extraction 
        ee_pos_all = env.unwrapped.scene["robot"].data.body_pos_w[:, moving_gripper_idx].cpu().numpy() 
        ee_quat_all = env.unwrapped.scene["robot"].data.body_quat_w[:, moving_gripper_idx].cpu().numpy() 
        joint_positions_all = env.unwrapped.scene["robot"].data.joint_pos.cpu().numpy() 
        dist_to_target_all = np.linalg.norm(current_hover_targets - ee_pos_all, axis=1)

        # Action Computation 
        action_batch = np.zeros((num_envs, 7), dtype=np.float32)
        for env_id in range(num_envs):
            env_action = oracles[env_id].compute_action(ee_pos_all[env_id], ee_quat_all[env_id], hover_target=current_hover_targets[env_id])
            env_action[:3] = np.clip(env_action[:3], -ACTION_CLAMP, ACTION_CLAMP)
            action_batch[env_id] = env_action

        # Upload to the sim device. 
        action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=sim_device) 

        # Dataset Recording (Conditional)
        for env_id in range(num_envs):
            if oracles[env_id].state < 4: # Only save frames during active approach/spray phases, not the hold states
                if args_cli.save_data:
                    assert datasets is not None
                    img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][env_id]
                    img_numpy = img_tensor.cpu().numpy()

                    if img_numpy.shape[-1] == 4:
                        img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2RGB)
                    else:
                        img_rgb = img_numpy

                    ee_pose_env = np.concatenate([ee_pos_all[env_id], ee_quat_all[env_id]])
                    frame_dict = {
                        "env_id": np.array([env_id], dtype=np.int64),
                        "task": TASK_DESCRIPTION,
                        "observation.state": joint_positions_all[env_id].astype(np.float32),
                        "observation.state.ee_pose": ee_pose_env.astype(np.float32),
                        "observation.images.wrist_camera": Image.fromarray(img_rgb),
                        "action": action_batch[env_id].astype(np.float32),
                    }
                    datasets[env_id].add_frame(frame_dict)
                episode_frame_count[env_id] += 1

        total_saved_frames = int(saved_frame_count.sum())
        if (total_saved_frames if args_cli.save_data else step) >= MAX_TOTAL_SAVED_FRAMES: 
            if args_cli.save_data and datasets is not None:
                for env_id in range(num_envs):
                    # Keep quality high: drop in-progress partial episodes at shutdown.
                    if episode_frame_count[env_id] > 0:
                        datasets[env_id].clear_episode_buffer()
                    datasets[env_id].finalize()
                abs_save_path = os.path.abspath(args_cli.dataset_root)
                print(f"[INFO]: Collected {total_saved_frames} high-quality frames.")
                print(f"[INFO]: >>> DATA SUCCESSFULLY SAVED TO: {abs_save_path} <<<")
            else:
                print(f"[INFO]: Reached {step} test frames! Shutting down.") 
            simulation_app.close() 
            break 

        # Telemetry heartbeat 
        if step % 50 == 0: 
            state_values = np.array([oracle.state for oracle in oracles], dtype=np.int64)
            state_counts = np.bincount(state_values, minlength=6)
            print(
                f"[STEP {step:05d}] states=0:{state_counts[0]} 1:{state_counts[1]} 2:{state_counts[2]} "
                f"3:{state_counts[3]} 4(Succ):{state_counts[4]} 5(Fail):{state_counts[5]} | saved_frames={int(saved_frame_count.sum())}"
            )
            for env_id in range(num_envs):
                prev_dist = prev_dist_to_target[env_id]
                curr_dist = float(dist_to_target_all[env_id])
                delta_str = "na" if np.isnan(prev_dist) else f"{curr_dist - prev_dist:+.3f}"
                
                err_to_hover = current_hover_targets[env_id] - ee_pos_all[env_id]
                
                if oracles[env_id].state < 4:
                    state_norm = float(np.linalg.norm(err_to_hover))
                    state_norm_name = "hover"
                elif oracles[env_id].state == 4:
                    state_norm = 0.0
                    state_norm_name = "success"
                else:
                    state_norm = 0.0
                    state_norm_name = "fail"
                
                print(
                    f"[E{env_id:04d}] s={oracles[env_id].state} steps={oracles[env_id].state_steps:03d} "
                    f"dist_moving->hover={np.linalg.norm(err_to_hover):.3f} "
                    f"state_norm({state_norm_name})={state_norm:.3f} d_dist={delta_str}"
                )
            prev_dist_to_target[:] = dist_to_target_all

        # Advance physics engine 
        obs, _, terminated, truncated, _ = env.step(action_tensor)
        step += 1 

        terminated_t = torch.as_tensor(terminated, device=sim_device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=sim_device, dtype=torch.bool)
        done_mask = torch.logical_or(terminated_t, truncated_t).reshape(-1)
        done_env_ids_t = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        done_env_ids = done_env_ids_t.cpu().tolist() if done_env_ids_t.numel() > 0 else []

        # Per-env lifecycle reset driven by env termination signals.
        if done_env_ids:
            if args_cli.save_data and datasets is not None:
                for env_id in done_env_ids:
                    if oracles[env_id].completed and not oracles[env_id].timed_out:
                        if episode_frame_count[env_id] > 0:
                            datasets[env_id].save_episode()
                            saved_frame_count[env_id] += episode_frame_count[env_id]
                    else:
                        if episode_frame_count[env_id] > 0:
                            datasets[env_id].clear_episode_buffer()
            print(
                f"[INFO]: Resetting {len(done_env_ids)} env(s) on termination/truncation. "
                f"Total frames saved so far: {int(saved_frame_count.sum())}"
            )

            done_env_ids_reset_t = torch.as_tensor(done_env_ids, device=sim_device, dtype=torch.long)
            # robot.write_root_pose_to_sim(
            #     absolute_shifted_pose.index_select(0, done_env_ids_reset_t),
            #     env_ids=done_env_ids_reset_t,
            # )
            # set_rest_pose(env, REST_POSE, env_ids=done_env_ids)

            refreshed_targets = prepare_episode_targets(
                stage=stage,
                palm_root_paths=palm_root_paths,
                episode_rng=episode_rng,
                cull_prob=args_cli.top_leaf_cull_prob,
                env_ids=done_env_ids,
            )
            for i, env_id in enumerate(done_env_ids):
                current_hover_targets[env_id] = refreshed_targets[i] 
                oracles[env_id] = SprayOracle()
                episode_frame_count[env_id] = 0
            
            # # Refresh Blue Hover Targets
            # spawn_target_markers(stage, current_hover_targets, env_ids=done_env_ids, marker_type="hover", color=(0.0, 0.0, 1.0))

    import sys
    sys.exit(0)

if __name__ == "__main__": 
    main()
