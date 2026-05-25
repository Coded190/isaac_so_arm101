# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LeRobot dataset management for data collection."""

import os
import cv2
import numpy as np
from typing import Optional, List, Dict
from PIL import Image

from .config import TASK_DESCRIPTION


class DatasetManager:
    """Manages creation and recording of LeRobot datasets."""
    
    def __init__(self, dataset_root: str, fps: int = 30, debug_verbose: bool = False):
        """Initialize dataset manager.
        
        Args:
            dataset_root: Root directory for datasets
            fps: Frames per second for recording
            debug_verbose: Enable verbose console output
        """
        self.dataset_root = dataset_root
        self.fps = fps
        self.debug_verbose = debug_verbose
        self.datasets = None
    
    def initialize_datasets(self, num_envs: int, num_dof: int,
                           img_shape: tuple) -> None:
        """Initialize LeRobot datasets for all environments.
        
        Args:
            num_envs: Number of environments
            num_dof: Number of robot DOF
            img_shape: Image shape as (C, H, W)
        """
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            print("[ERROR]: 'lerobot' package not found. Install with 'pip install lerobot'.")
            raise
        
        IMG_H, IMG_W = img_shape[1], img_shape[2]
        
        features = {
            "env_id": {"dtype": "int64", "shape": (1,), "names": None},
            "observation.state": {
                "dtype": "float32", "shape": (num_dof,),
                "names": [f"joint_{i}" for i in range(num_dof)],
            },
            "observation.state.ee_pose": {
                "dtype": "float32", "shape": (7,),
                "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            },
            "observation.images.wrist_camera": {
                "dtype": "video", "shape": (3, IMG_H, IMG_W),
                "names": ["c", "h", "w"],
            },
            "action": {
                "dtype": "float32", "shape": (7,),
                "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
            },
        }
        
        self.datasets = []
        for env_id in range(num_envs):
            env_repo_id = f"local/vla_palm_dataset_env_{env_id:04d}"
            env_root = os.path.join(self.dataset_root, f"env_{env_id:04d}")
            env_dataset = LeRobotDataset.create(
                repo_id=env_repo_id,
                root=env_root,
                fps=self.fps,
                features=features,
            )
            # Force immediate metadata flush
            env_dataset.meta.metadata_buffer_size = 1
            self.datasets.append(env_dataset)
        
        abs_save_path = os.path.abspath(self.dataset_root)
        print(f"[INFO]: Initialized {num_envs} per-env LeRobot datasets.")
        print(f"[INFO]: >>> SAVING DATA TO DIRECTORY: {abs_save_path} <<<")
    
    def add_frame(self, env_id: int, joint_positions: np.ndarray,
                 ee_position: np.ndarray, ee_quaternion: np.ndarray,
                 camera_image: np.ndarray, action: np.ndarray) -> None:
        """Record a frame to the dataset.
        
        Args:
            env_id: Environment ID
            joint_positions: Joint positions (num_dof,)
            ee_position: End effector position (3,)
            ee_quaternion: End effector quaternion WXYZ (4,)
            camera_image: Camera image with shape (H, W, C) or (H, W, 4)
            action: Control action (7,)
        """
        if self.datasets is None or env_id >= len(self.datasets):
            return
        
        # Convert RGBA to RGB if needed
        if camera_image.shape[-1] == 4:
            img_rgb = cv2.cvtColor(camera_image, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = camera_image
        
        ee_pose = np.concatenate([ee_position, ee_quaternion])
        
        frame_dict = {
            "env_id": np.array([env_id], dtype=np.int64),
            "task": TASK_DESCRIPTION,
            "observation.state": joint_positions.astype(np.float32),
            "observation.state.ee_pose": ee_pose.astype(np.float32),
            "observation.images.wrist_camera": Image.fromarray(img_rgb),
            "action": action.astype(np.float32),
        }
        
        self.datasets[env_id].add_frame(frame_dict)
    
    def save_episode(self, env_id: int, save: bool = True) -> None:
        """Save or discard current episode buffer for an environment.
        
        Args:
            env_id: Environment ID
            save: True to save episode, False to discard
        """
        if self.datasets is None or env_id >= len(self.datasets):
            return
        
        if save:
            self.datasets[env_id].save_episode()
        else:
            self.datasets[env_id].clear_episode_buffer()
    
    def save_all_episodes(self, save_mask: np.ndarray) -> int:
        """Save episodes based on success mask.
        
        Args:
            save_mask: Boolean array indicating which episodes to save
            
        Returns:
            Total frames saved
        """
        if self.datasets is None:
            return 0
        
        total_frames = 0
        for env_id in range(len(self.datasets)):
            if save_mask[env_id]:
                self.datasets[env_id].save_episode()
            else:
                self.datasets[env_id].clear_episode_buffer()
        
        return total_frames
    
    def finalize(self) -> None:
        """Finalize and close all datasets."""
        if self.datasets is None:
            return
        
        for dataset in self.datasets:
            dataset.finalize()
        
        abs_save_path = os.path.abspath(self.dataset_root)
        print(f"[INFO]: >>> DATA SUCCESSFULLY SAVED TO: {abs_save_path} <<<")
