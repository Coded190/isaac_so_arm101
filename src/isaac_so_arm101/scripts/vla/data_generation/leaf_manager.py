# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Leaf culling and visibility management for palm trees."""

import numpy as np
from typing import List, Optional
from pxr import UsdGeom

from .geometry_utils import iter_leaf_prims, leaf_world_positions, get_crown_centroid
from .config import LEAF_CULL_Z_OFFSET, LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX


class LeafManager:
    """Manages leaf visibility and culling for palm tree randomization."""
    
    def __init__(self, stage, debug_verbose: bool = False):
        """Initialize leaf manager.
        
        Args:
            stage: USD stage
            debug_verbose: Enable verbose console output
        """
        self.stage = stage
        self.debug_verbose = debug_verbose
    
    def set_leaf_prims_active(self, palm_root_path: str, active: bool = True) -> None:
        """Activate or deactivate every leaf using Visibility.
        
        Args:
            palm_root_path: Path to palm root
            active: True to make visible, False to hide
        """
        for child in iter_leaf_prims(self.stage, palm_root_path):
            if UsdGeom.Xformable(child):
                imageable = UsdGeom.Imageable(child)
                if active:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
    
    def remove_top_leaves(self, palm_root_path: str, crown_z: float, keep_ratio: float,
                          z_threshold_offset: float = LEAF_CULL_Z_OFFSET) -> None:
        """Remove top leaves from a palm tree.
        
        Args:
            palm_root_path: Path to palm root
            crown_z: Z-coordinate of crown center
            keep_ratio: Fraction of top leaves to keep (0.5 = remove 50%)
            z_threshold_offset: Z offset above crown_z for culling threshold
        """
        cull_z = crown_z + z_threshold_offset
        leaves = leaf_world_positions(self.stage, palm_root_path)
        
        top_leaves = [(prim, pos[2]) for prim, pos in leaves if pos[2] > cull_z]
        top_leaves.sort(key=lambda x: -x[1])
        n_remove = int(len(top_leaves) * (1.0 - keep_ratio))
        
        for prim, _ in top_leaves[:n_remove]:
            UsdGeom.Imageable(prim).MakeInvisible()
    
    def cull_episode_leaves(self, palm_root_paths: List[str], episode_rng: np.random.Generator,
                            cull_prob: float, env_ids: Optional[List[int]] = None) -> None:
        """Cull top leaves for each environment based on probability.
        
        Should be called BEFORE positioning the robot so that leaf clearance checks
        and robot positioning are based on the final (culled) leaf geometry.
        
        Args:
            palm_root_paths: List of palm root paths
            episode_rng: Random number generator for this episode
            cull_prob: Probability of culling (0.0-1.0)
            env_ids: List of environment IDs to cull (None = all)
        """
        if env_ids is None:
            env_ids = list(range(len(palm_root_paths)))
        
        for env_id in env_ids:
            palm_root_path = palm_root_paths[env_id]
            # Re-activate all leaves first in case they were culled in previous episode
            self.set_leaf_prims_active(palm_root_path, active=True)
            
            if episode_rng.random() < cull_prob:
                crown_centroid = get_crown_centroid(self.stage, palm_root_path)
                keep_ratio = float(episode_rng.uniform(LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX))
                self.remove_top_leaves(
                    palm_root_path,
                    crown_z=crown_centroid[2],
                    keep_ratio=keep_ratio,
                )
