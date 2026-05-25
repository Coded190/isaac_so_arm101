# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Robot positioning and control for base and arm movement."""

import numpy as np
import torch
from typing import Optional, List

from .geometry_utils import get_crown_centroid, closest_leaf_dist_xy
from .math_utils import yaw_to_quat_wxyz, yaw_from_quat_wxyz, rotate_vec_z
from .config import (
    REST_POSE_VALUES,
    ANGLE_RANDOM_RANGE,
    TREE_INWARD_OFFSET,
    MIN_TREE_RADIUS,
    MAX_TREE_RADIUS,
    LEAF_CLEARANCE,
    PLACEMENT_MAX_ATTEMPTS,
    HOVER_OFFSET,
    HOVER_PULLBACK_M,
    DEBUG_VERBOSE,
)


class RobotController:
    """Controls robot base positioning and joint state."""
    
    def __init__(self, stage, debug_verbose: bool = False):
        """Initialize robot controller.
        
        Args:
            stage: USD stage
            debug_verbose: Enable verbose console output
        """
        self.stage = stage
        self.debug_verbose = debug_verbose
    
    def set_rest_pose(self, env, rest_pose_tensor: torch.Tensor,
                     env_ids: Optional[List[int]] = None, noise_scale: float = 0.05) -> None:
        """Set arm to rest pose with optional joint jitter for robustness.
        
        Args:
            env: Gym environment
            rest_pose_tensor: Rest pose values as tensor
            env_ids: List of env IDs to reset (None = all)
            noise_scale: Maximum noise offset in radians
        """
        robot = env.unwrapped.scene["robot"]
        
        if env_ids is None:
            num_envs = env.unwrapped.num_envs
            joint_pos = rest_pose_tensor.expand(num_envs, -1).clone()
            env_ids_t = None
        else:
            env_ids_t = torch.as_tensor(env_ids, device=rest_pose_tensor.device, dtype=torch.long)
            num_envs = env_ids_t.shape[0]
            joint_pos = rest_pose_tensor.expand(num_envs, -1).clone()
        
        # Add jitter to first 5 joints (not gripper)
        noise = (torch.rand(num_envs, 5, device=joint_pos.device) * 2.0 - 1.0) * noise_scale
        joint_pos[:, :5] += noise
        
        zero_vel = torch.zeros_like(joint_pos)
        if env_ids is None:
            robot.write_joint_state_to_sim(joint_pos, zero_vel)
        else:
            robot.write_joint_state_to_sim(joint_pos, zero_vel, env_ids=env_ids_t)
    
    def randomize_robot_root_pose(self, env, palm_root_paths: List[str],
                                 episode_rng: np.random.Generator,
                                 env_ids: Optional[List[int]] = None) -> None:
        """Place robot at random angle on circle around tree crown.
        
        Args:
            env: Gym environment
            palm_root_paths: List of palm root paths
            episode_rng: Random number generator for episode
            env_ids: List of env IDs to randomize (None = all)
        """
        robot = env.unwrapped.scene["robot"]
        device = robot.data.default_root_state.device
        
        if env_ids is None:
            env_ids = list(range(env.unwrapped.num_envs))
        env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        
        new_root = robot.data.default_root_state[env_ids_t].clone()
        current_root_pos_w = robot.data.root_pos_w[env_ids_t].cpu().numpy().astype(np.float64)
        env_origins = env.unwrapped.scene.env_origins[env_ids_t, :2].cpu().numpy().astype(np.float64)
        
        trunk_xys = []
        trunk_zs = []
        
        for i, env_id in enumerate(env_ids):
            palm_root_path = palm_root_paths[env_id]
            crown = get_crown_centroid(self.stage, palm_root_path)
            trunk_xy = np.array([float(crown[0]), float(crown[1])], dtype=np.float64)
            trunk_z = float(crown[2])
            trunk_xys.append(trunk_xy)
            trunk_zs.append(trunk_z)
            
            current_xy = current_root_pos_w[i, :2]
            rel = current_xy - trunk_xy
            radius0 = float(np.linalg.norm(rel))
            bearing0 = float(np.arctan2(rel[1], rel[0]))
            
            if self.debug_verbose:
                radius_after = max(radius0 - TREE_INWARD_OFFSET, MIN_TREE_RADIUS)
                print(f"[randomize env {env_id}] current_pos={current_xy} trunk={trunk_xy} "
                      f"radius0={radius0:.3f}m radius_after={radius_after:.3f}m", flush=True)
            
            # Try to find leaf-free placement
            new_x = new_y = None
            last_dist = None
            for attempt in range(PLACEMENT_MAX_ATTEMPTS):
                theta = float(episode_rng.uniform(-ANGLE_RANDOM_RANGE, ANGLE_RANDOM_RANGE))
                radius = max(radius0 - TREE_INWARD_OFFSET, MIN_TREE_RADIUS)
                radius = min(radius, MAX_TREE_RADIUS)
                bearing = bearing0 + theta
                cand_x = float(trunk_xy[0]) + radius * float(np.cos(bearing))
                cand_y = float(trunk_xy[1]) + radius * float(np.sin(bearing))
                d = closest_leaf_dist_xy(self.stage, palm_root_path, (cand_x, cand_y))
                last_dist = d
                
                if d is None or d >= LEAF_CLEARANCE:
                    new_x, new_y = cand_x, cand_y
                    if attempt > 0 and self.debug_verbose:
                        print(f"[reset env {env_id}] leaf-clear placement found on "
                              f"attempt {attempt + 1} (closest leaf {d:.3f} m)", flush=True)
                    break
                elif self.debug_verbose:
                    print(f"[reset env {env_id}] attempt {attempt + 1}/{PLACEMENT_MAX_ATTEMPTS}: "
                          f"leaf at {d:.3f}m (< {LEAF_CLEARANCE:.2f}), retrying", flush=True)
            
            if new_x is None:
                new_x = cand_x
                new_y = cand_y
                if self.debug_verbose:
                    print(f"[reset env {env_id}] no leaf-clear placement after "
                          f"{PLACEMENT_MAX_ATTEMPTS} attempts; closest leaf {last_dist:.3f}m",
                          flush=True)
            
            # Set base position and orientation
            face_x = float(trunk_xy[0]) - new_x
            face_y = float(trunk_xy[1]) - new_y
            yaw = float(np.arctan2(face_x, -face_y))
            quat = yaw_to_quat_wxyz(yaw)
            
            new_root[i, 0] = new_x
            new_root[i, 1] = new_y
            new_root[i, 2] = trunk_zs[i]
            new_root[i, 3:7] = torch.tensor(quat, dtype=new_root.dtype, device=device)
        
        robot.write_root_pose_to_sim(new_root[:, :7], env_ids=env_ids_t)
        zero_vel = torch.zeros((len(env_ids), 6), dtype=new_root.dtype, device=device)
        robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_t)
        
        # Reset joint state
        rest_pose_tensor = torch.tensor(REST_POSE_VALUES, dtype=new_root.dtype, device=device)
        self.set_rest_pose(env, rest_pose_tensor, env_ids=env_ids)
    
    def get_deterministic_target(self, stage, palm_root_path: str) -> np.ndarray:
        """Get deterministic hover target for a palm tree.
        
        Args:
            stage: USD stage
            palm_root_path: Path to palm root
            
        Returns:
            Target position as (x, y, z)
        """
        crown_centroid = get_crown_centroid(stage, palm_root_path)
        return crown_centroid + HOVER_OFFSET
    
    def prepare_episode_targets(self, palm_root_paths: List[str],
                               robot_xys: Optional[np.ndarray] = None,
                               env_ids: Optional[List[int]] = None) -> np.ndarray:
        """Compute hover targets for environments.
        
        Args:
            palm_root_paths: List of palm root paths
            robot_xys: Robot XY positions for pullback calculation
            env_ids: Environment IDs to prepare (None = all)
            
        Returns:
            Array of target positions
        """
        if env_ids is None:
            env_ids = list(range(len(palm_root_paths)))
        
        targets = []
        for env_id in env_ids:
            palm_root_path = palm_root_paths[env_id]
            crown_centroid = get_crown_centroid(self.stage, palm_root_path)
            target = self.get_deterministic_target(self.stage, palm_root_path)
            
            if robot_xys is not None and HOVER_PULLBACK_M > 0.0:
                rxy = np.asarray(robot_xys[env_id], dtype=np.float64)
                cxy = np.array([crown_centroid[0], crown_centroid[1]], dtype=np.float64)
                back = rxy - cxy
                n = float(np.linalg.norm(back))
                if n > 1e-6:
                    back /= n
                    target[0] += float(back[0]) * HOVER_PULLBACK_M
                    target[1] += float(back[1]) * HOVER_PULLBACK_M
            
            if self.debug_verbose:
                robot_xy = robot_xys[env_id] if robot_xys is not None else None
                print(f"[prepare_targets env {env_id}] leaf_mean={crown_centroid} "
                      f"target={target} robot_xy={robot_xy}", flush=True)
            targets.append(target)
        
        return np.asarray(targets, dtype=np.float32)
