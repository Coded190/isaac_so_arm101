# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""VLA data-generation entry point for Isaac Lab.

Refactored with modular architecture.

Typical launch:
    uv run vla_data_gen_v2 \
        --task Isaac-PING-TI-VLA-v0 --num_envs 10 \
        --enable_cameras --save_data
"""

import argparse
import os
import sys
import numpy as np
import torch
import gymnasium as gym
import cv2

# Install noise filtering BEFORE any Isaac imports
import config
from noise_filter import NoiseFilter

sys.stderr = NoiseFilter(sys.stderr, config.NOISE_FILTER_DROP_PATTERNS)
sys.stdout = NoiseFilter(sys.stdout, config.NOISE_FILTER_DROP_PATTERNS)

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="VLA palm-spray data generation for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                   help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1,
                   help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="None",
                   help="Name of the task.")
parser.add_argument("--save_data", action="store_true",
                   help="Enable saving data with LeRobotDataset.")
parser.add_argument("--top_leaf_cull_prob", type=float, default=0.5,
                   help="Probability of culling the top leaves per episode.")
parser.add_argument("--dataset_root", type=str, default="outputs/vla_palm_dataset",
                   help="Root folder for LeRobot datasets.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# PhysX configuration
sys.argv.append("--/persistent/omni/physx/persistentErrorMaxCount=10000000")
sys.argv.append("--/persistent/omni/physx/rejectUnsupportedActors=false")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set_string("/log/level", "error")
carb.settings.get_settings().set_int("/physics/numThreads", 0)
carb.settings.get_settings().set_string("/log/channels/omni.physx.plugin", "fatal")
carb.settings.get_settings().set_string("/log/channels/omni.kit.notification_manager.manager", "fatal")

# Import modular components
import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg
from math_utils import rotate_vec_z, yaw_from_quat_wxyz
from geometry_utils import get_crown_centroid
from leaf_manager import LeafManager
from palm_randomizer import PalmRandomizer
from physics_setup import PhysicsSetup
from lighting_manager import LightingManager
from camera_controller import CameraController
from robot_controller import RobotController
from spray_oracle import SprayOracle
from dataset_manager import DatasetManager


def setup_environment():
    """Initialize the Isaac Lab environment and configuration."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Set neutral initial position
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
    env_cfg.scene.robot.init_state.joint_pos = {
        "base_yaw": config.REST_POSE_VALUES[0],
        "shoulder_pitch": config.REST_POSE_VALUES[1],
        "elbow_pitch": config.REST_POSE_VALUES[2],
        "wrist_pitch": config.REST_POSE_VALUES[3],
        "wrist_roll": config.REST_POSE_VALUES[4],
        "gripper_moving": config.REST_POSE_VALUES[5],
    }
    env_cfg.episode_length_s = config.EPISODE_LENGTH_S
    
    # 4-DOF IK configuration
    try:
        env_cfg.actions.arm_action.joint_names = [
            "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll",
        ]
        env_cfg.actions.arm_action.body_name = "moving_gripper"
    except Exception as e:
        print(f"[setup] couldn't configure arm action: {e}", flush=True)
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    return env


def main():
    """Main simulation loop."""
    # Setup environment
    env = setup_environment()
    sim_device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    
    # Initialize managers
    leaf_manager = LeafManager(stage, debug_verbose=config.DEBUG_VERBOSE)
    palm_randomizer = PalmRandomizer(stage, debug_verbose=config.DEBUG_VERBOSE)
    physics_setup = PhysicsSetup(stage, debug_verbose=config.DEBUG_VERBOSE)
    lighting_manager = LightingManager(stage, debug_verbose=config.DEBUG_VERBOSE)
    camera_controller = CameraController(debug_verbose=config.DEBUG_VERBOSE)
    robot_controller = RobotController(stage, debug_verbose=config.DEBUG_VERBOSE)
    
    # Setup palm trees
    palm_root_paths = [f"/World/envs/env_{env_id}/Scene/{config.PALM_ROOT_NAME}"
                       for env_id in range(num_envs)]
    # for palm_path in palm_root_paths:
    #     physics_setup.disable_palm_physics(stage, palm_path)
    
    # Initialize random seed
    episode_rng = np.random.default_rng(getattr(args_cli, "seed", None))
    
    # Setup lighting and initial randomization
    lighting_manager.randomize_lighting(config.HDRI_FOLDER_PATH, env_ids=range(num_envs))
    leaf_manager.cull_episode_leaves(
        stage=stage,
        palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
        cull_prob=args_cli.top_leaf_cull_prob,
    )
    
    # Position robot
    robot_controller.randomize_robot_root_pose(
        env=env, stage=stage, palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
    )
    
    # Prepare hover targets
    robot = env.unwrapped.scene["robot"]
    current_hover_targets = np.zeros((num_envs, 3), dtype=np.float32)
    robot_xys_now = robot.data.root_pos_w[:, :2].cpu().numpy()
    current_hover_targets[:] = robot_controller.prepare_episode_targets(
        stage=stage,
        palm_root_paths=palm_root_paths,
        robot_xys=robot_xys_now,
    )
    
    # Initialize FSM oracles
    oracles = [SprayOracle() for _ in range(num_envs)]
    
    # Setup dataset recording if enabled
    dataset_manager = None
    if args_cli.save_data:
        robot_asset = env.unwrapped.scene["robot"]
        num_dof = robot_asset.data.default_joint_pos.shape[-1]
        img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]
        IMG_H, IMG_W = img_tensor.shape[0], img_tensor.shape[1]
        
        dataset_manager = DatasetManager(args_cli.dataset_root, fps=30,
                                        debug_verbose=config.DEBUG_VERBOSE)
        dataset_manager.initialize_datasets(num_envs, num_dof, (3, IMG_H, IMG_W))
    else:
        print("[INFO]: Running in TEST MODE. Data will NOT be saved.")
    
    # Get body indices
    moving_gripper_indices, _ = robot.find_bodies("moving_gripper")
    moving_gripper_idx = moving_gripper_indices[0]
    
    # Tracking variables
    episode_frame_count = np.zeros(num_envs, dtype=np.int64)
    saved_frame_count = np.zeros(num_envs, dtype=np.int64)
    leaf_stuck_steps = [0] * num_envs
    pending_reset_env_ids = set()
    step_counter = 0
    prev_states = [oracle.state for oracle in oracles]
    first_step = True
    
    # Get joint indices for pose control
    joint_names = robot.joint_names
    wrist_pitch_idx = joint_names.index("wrist_pitch")
    
    # Main simulation loop
    while simulation_app.is_running():
        robot_data = robot.data
        ee_pos_all = robot_data.body_pos_w[:, moving_gripper_idx].cpu().numpy()
        ee_quat_all = robot_data.body_quat_w[:, moving_gripper_idx].cpu().numpy()
        root_quat_all = robot_data.root_quat_w.cpu().numpy()
        
        # Debug output on first step
        if first_step and config.DEBUG_VERBOSE:
            robot_base_all = robot_data.root_pos_w.cpu().numpy()
            for env_id in range(num_envs):
                ee_pos = ee_pos_all[env_id]
                target_pos = current_hover_targets[env_id]
                base_pos = robot_base_all[env_id]
                dist_to_target = float(np.linalg.norm(ee_pos - target_pos))
                print(f"[first_step env {env_id}] ee={ee_pos} target={target_pos} "
                      f"dist={dist_to_target:.3f}m base={base_pos}", flush=True)
            first_step = False
        
        # Compute actions for all environments
        action_batch = np.zeros((num_envs, 7), dtype=np.float32)
        for env_id in range(num_envs):
            env_action = oracles[env_id].compute_action(
                ee_pos_all[env_id], ee_quat_all[env_id],
                hover_target=current_hover_targets[env_id],
            )
            
            # Rotate to body frame
            base_yaw = yaw_from_quat_wxyz(root_quat_all[env_id])
            env_action[0:3] = rotate_vec_z(env_action[0:3], -base_yaw)
            env_action[3:6] = rotate_vec_z(env_action[3:6], -base_yaw)
            env_action[:3] = np.clip(env_action[:3], -config.ACTION_CLAMP, config.ACTION_CLAMP)
            action_batch[env_id] = env_action
            
            # Debug state transitions
            if config.DEBUG_VERBOSE and oracles[env_id].state != prev_states[env_id]:
                hover = current_hover_targets[env_id]
                dist = float(np.linalg.norm(ee_pos_all[env_id] - hover))
                old = config.FSM_STATE_NAMES.get(prev_states[env_id], str(prev_states[env_id]))
                new = config.FSM_STATE_NAMES.get(oracles[env_id].state, str(oracles[env_id].state))
                print(f"[env {env_id} step {step_counter:5d}] state {old} -> {new}  "
                      f"dist_to_hover={dist:.3f}m  yaw={np.rad2deg(base_yaw):+7.2f}deg",
                      flush=True)
                prev_states[env_id] = oracles[env_id].state
        
        step_counter += 1
        
        # Record frames to dataset if in approach/spray states
        if dataset_manager is not None:
            joint_positions_all = robot_asset.data.joint_pos.cpu().numpy()
            for env_id in range(num_envs):
                if oracles[env_id].state < 4:
                    img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][env_id]
                    img_numpy = img_tensor.cpu().numpy()
                    
                    ee_pose_env = np.concatenate([ee_pos_all[env_id], ee_quat_all[env_id]])
                    
                    dataset_manager.add_frame(
                        env_id=env_id,
                        joint_positions=joint_positions_all[env_id],
                        ee_position=ee_pos_all[env_id],
                        ee_quaternion=ee_quat_all[env_id],
                        camera_image=img_numpy,
                        action=action_batch[env_id],
                    )
                    episode_frame_count[env_id] += 1
        
        # Check hard frame cap
        total_saved_frames = int(saved_frame_count.sum())
        if total_saved_frames >= config.MAX_TOTAL_SAVED_FRAMES:
            if dataset_manager is not None:
                dataset_manager.finalize()
                abs_save_path = os.path.abspath(args_cli.dataset_root)
                print(f"[INFO]: Collected {total_saved_frames} high-quality frames.")
            try:
                env.close()
            except Exception:
                pass
            os._exit(0)
        
        # Step environment
        action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=sim_device)
        _, _, terminated, truncated, _ = env.step(action_tensor)
        
        terminated_t = torch.as_tensor(terminated, device=sim_device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=sim_device, dtype=torch.bool)
        done_mask = torch.logical_or(terminated_t, truncated_t).reshape(-1)
        done_env_ids_t = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        done_env_ids = done_env_ids_t.cpu().tolist() if done_env_ids_t.numel() > 0 else []
        
        # Detect stuck episodes
        ee_pos_post = robot_data.body_pos_w[:, moving_gripper_idx].cpu().numpy()
        stuck_env_ids = []
        # (Leaf stuck detection would go here - simplified for now)
        
        pending_reset_env_ids.update(done_env_ids)
        pending_reset_env_ids.update(stuck_env_ids)
        
        # Reset all environments if any are done
        if len(pending_reset_env_ids) == num_envs and pending_reset_env_ids:
            reset_env_ids = list(range(num_envs))
            
            # Save episodes
            if dataset_manager is not None:
                for env_id in reset_env_ids:
                    keep = (
                        oracles[env_id].completed
                        and not oracles[env_id].timed_out
                        and env_id not in stuck_env_ids
                    )
                    dataset_manager.save_episode(env_id, save=keep)
                    if keep:
                        saved_frame_count[env_id] += episode_frame_count[env_id]
                
                print(f"[INFO]: Resetting all {len(reset_env_ids)} envs; "
                      f"total frames saved so far: {int(saved_frame_count.sum())}")
            
            # Re-randomize for next episode
            lighting_manager.randomize_lighting(config.HDRI_FOLDER_PATH, env_ids=range(num_envs))
            leaf_manager.cull_episode_leaves(
                stage=stage,
                palm_root_paths=palm_root_paths,
                episode_rng=episode_rng,
                cull_prob=args_cli.top_leaf_cull_prob,
                env_ids=reset_env_ids,
            )
            
            for env_id in reset_env_ids:
                specific_palm_path = f"/World/envs/env_{env_id}/{config.PALM_ROOT_NAME}"
                palm_randomizer.randomize_palm_dimensions(specific_palm_path)
            
            robot_controller.randomize_robot_root_pose(
                env=env, stage=stage, palm_root_paths=palm_root_paths,
                episode_rng=episode_rng, env_ids=reset_env_ids,
            )
            
            robot_xys_now = robot.data.root_pos_w[:, :2].cpu().numpy()
            refreshed_targets = robot_controller.prepare_episode_targets(
                stage=stage,
                palm_root_paths=palm_root_paths,
                robot_xys=robot_xys_now,
                env_ids=reset_env_ids,
            )
            
            for i, env_id in enumerate(reset_env_ids):
                current_hover_targets[env_id] = refreshed_targets[i]
                oracles[env_id].reset()
                prev_states[env_id] = 0
                leaf_stuck_steps[env_id] = 0
                episode_frame_count[env_id] = 0
            
            pending_reset_env_ids.clear()
    
    # Cleanup
    try:
        env.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
