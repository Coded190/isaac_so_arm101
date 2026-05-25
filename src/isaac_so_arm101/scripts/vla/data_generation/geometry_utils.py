# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Geometric utility functions for spatial calculations."""

import numpy as np
from typing import Tuple, Optional, List
from pxr import Usd, UsdGeom


def iter_leaf_prims(stage, palm_root_path: str):
    """Iterate over all leaf primitive objects under a palm tree.
    
    Args:
        stage: USD stage
        palm_root_path: Path to the palm root primitive
        
    Yields:
        Leaf primitives
    """
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm:
        return
    crown = _get_palm_crown_prim(stage, palm_root_path)
    search_root = crown if crown else palm
    for child in Usd.PrimRange(search_root):
        if child == search_root:
            continue
        name = child.GetName().lower()
        if (name.startswith("leaf_") or name.startswith("leaf_b_")) and UsdGeom.Xformable(child):
            yield child


def _get_palm_crown_prim(stage, palm_root_path: str):
    """Get the crown primitive from a palm tree root.
    
    Args:
        stage: USD stage
        palm_root_path: Path to the palm root primitive
        
    Returns:
        Crown primitive or the palm root if crown not found
    """
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm or not palm.IsValid():
        return None
    crown = palm.GetChild("crown")
    if crown and crown.IsValid():
        return crown
    return palm


def leaf_world_positions(stage, palm_root_path: str) -> List[Tuple]:
    """Get world positions of all leaf primitives.
    
    Args:
        stage: USD stage
        palm_root_path: Path to the palm root primitive
        
    Returns:
        List of (primitive, position) tuples
    """
    out = []
    for prim in iter_leaf_prims(stage, palm_root_path):
        xf = UsdGeom.Xformable(prim)
        try:
            wp = xf.ComputeLocalToWorldTransform(0).ExtractTranslation()
            out.append((prim, np.array([wp[0], wp[1], wp[2]], dtype=np.float64)))
        except Exception:
            continue
    return out


def get_crown_centroid(stage, palm_root_path: str) -> np.ndarray:
    """Get the mean world position of all active leaves (crown centroid).
    
    Args:
        stage: USD stage
        palm_root_path: Path to the palm root primitive
        
    Returns:
        Crown centroid position as (x, y, z)
    """
    leaves = leaf_world_positions(stage, palm_root_path)
    if not leaves:
        return np.array([0.0, 0.0, 5.0])
    positions = np.stack([p for _, p in leaves], axis=0)
    return positions.mean(axis=0)


def closest_leaf_dist_xy(stage, palm_root_path: str, robot_xy: np.ndarray) -> Optional[float]:
    """Get closest XY distance from robot to any active leaf.
    
    Args:
        stage: USD stage
        palm_root_path: Path to the palm root primitive
        robot_xy: Robot XY position as (x, y)
        
    Returns:
        Closest distance or None if no leaves present
    """
    leaves = leaf_world_positions(stage, palm_root_path)
    if not leaves:
        return None
    rxy = np.array([float(robot_xy[0]), float(robot_xy[1])], dtype=np.float64)
    return float(min(
        np.hypot(pos[0] - rxy[0], pos[1] - rxy[1]) for _, pos in leaves
    ))


def closest_leaf_dist_3d(stage, palm_root_path: str, point_xyz: np.ndarray) -> Optional[float]:
    """Get closest 3D distance from point to any active leaf.
    
    Args:
        stage: USD stage
        palm_root_path: Path to the palm root primitive
        point_xyz: Point position as (x, y, z)
        
    Returns:
        Closest distance or None if no leaves present
    """
    leaves = leaf_world_positions(stage, palm_root_path)
    if not leaves:
        return None
    pt = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])],
                  dtype=np.float64)
    return float(min(np.linalg.norm(pos - pt) for _, pos in leaves))
