# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera positioning and control for viewport recording."""

import numpy as np
from typing import Optional, Tuple

from .math_utils import yaw_from_quat_wxyz
from .config import (
    CAMERA_LATERAL_OFFSET,
    CAMERA_HEIGHT_OFFSET,
    CAMERA_TARGET_LIFT,
    DEBUG_VERBOSE,
)


class CameraController:
    """Controls viewport camera positioning to follow robot and crown."""
    
    def __init__(self, debug_verbose: bool = False):
        """Initialize camera controller.
        
        Args:
            debug_verbose: Enable verbose console output
        """
        self.debug_verbose = debug_verbose
        self._set_camera_view = self._import_camera_view()
    
    @staticmethod
    def _import_camera_view():
        """Import the camera view setting function."""
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            return set_camera_view
        except Exception:
            try:
                from omni.isaac.core.utils.viewports import set_camera_view
                return set_camera_view
            except Exception:
                return None
    
    def update_recording_camera(self, robot_xy: np.ndarray, crown_xy: np.ndarray, 
                               base_z: float, episode_rng: np.random.Generator,
                               lateral_offset: float = CAMERA_LATERAL_OFFSET,
                               height_offset: float = CAMERA_HEIGHT_OFFSET,
                               target_lift: float = CAMERA_TARGET_LIFT) -> None:
        """Move viewport camera to random side of robot-crown line.
        
        Picks a random left/right side and points at the midpoint between
        robot and crown, lifted to frame both base and canopy.
        
        Args:
            robot_xy: Robot position as (x, y)
            crown_xy: Crown position as (x, y)
            base_z: Robot base Z height
            episode_rng: Random number generator
            lateral_offset: Distance to side from center line
            height_offset: Camera height above base
            target_lift: Height of look-at point above base
        """
        if self._set_camera_view is None:
            return
        
        forward = np.array([float(crown_xy[0]) - float(robot_xy[0]),
                           float(crown_xy[1]) - float(robot_xy[1])],
                          dtype=np.float64)
        n = float(np.linalg.norm(forward))
        if n < 1e-6:
            return
        
        forward /= n
        perpendicular = np.array([-forward[1], forward[0]], dtype=np.float64)
        
        side = 1.0 if episode_rng.random() < 0.5 else -1.0
        camera_pos = [
            float(robot_xy[0] + side * lateral_offset * perpendicular[0]),
            float(robot_xy[1] + side * lateral_offset * perpendicular[1]),
            float(base_z + height_offset),
        ]
        target = [
            float(0.5 * (float(robot_xy[0]) + float(crown_xy[0]))),
            float(0.5 * (float(robot_xy[1]) + float(crown_xy[1]))),
            float(base_z + target_lift),
        ]
        
        try:
            self._set_camera_view(eye=camera_pos, target=target)
        except Exception:
            return
        
        if self.debug_verbose:
            side_name = "right" if side > 0 else "left"
            print(f"[camera] follow → {side_name} side  eye={camera_pos}  "
                  f"target={target}", flush=True)
