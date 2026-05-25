# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mathematical utility functions for quaternion and vector operations."""

import numpy as np
from typing import Tuple


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in (w, x, y, z) format.
    
    Args:
        q1: First quaternion as (w, x, y, z)
        q2: Second quaternion as (w, x, y, z)
        
    Returns:
        Product quaternion as (w, x, y, z)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Get the conjugate of a quaternion in (w, x, y, z) format.
    
    Args:
        q: Quaternion as (w, x, y, z)
        
    Returns:
        Conjugate quaternion
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to axis-angle representation.
    
    Args:
        q: Quaternion as (w, x, y, z)
        
    Returns:
        Axis-angle as 3-element vector (angle * axis)
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.zeros(3)
    q = q / n
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-6:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = q[1:4] / sin_half
    return axis * angle


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    """Create quaternion (w, x, y, z) for rotation about world Z axis.
    
    Args:
        yaw: Rotation angle about Z-axis in radians
        
    Returns:
        Quaternion as (w, x, y, z)
    """
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Extract yaw (rotation about Z) from quaternion (w, x, y, z).
    
    Args:
        q: Quaternion as (w, x, y, z)
        
    Returns:
        Yaw angle in radians
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def rotate_vec_z(v: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a 3-vector by yaw radians about the +Z axis.
    
    Args:
        v: 3D vector to rotate
        yaw: Rotation angle in radians
        
    Returns:
        Rotated vector
    """
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([c * v[0] - s * v[1],
                     s * v[0] + c * v[1],
                     v[2]], dtype=np.float64)


def cap_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Clamp a vector's norm to a maximum value.
    
    Args:
        vector: The vector to clamp
        max_norm: Maximum allowed norm
        
    Returns:
        Clamped vector
    """
    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)
