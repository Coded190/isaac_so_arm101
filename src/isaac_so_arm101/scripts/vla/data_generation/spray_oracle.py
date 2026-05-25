# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Finite state machine for spray control task execution."""

import numpy as np

from .math_utils import cap_vector_norm
from .config import (
    ACTION_CLAMP,
    POSITION_GAIN,
    SPRAY_DURATION,
    FSM_POSITION_THRESHOLD,
    FSM_MAX_STATE_STEPS,
)


class SprayOracle:
    """6-state FSM for palm spray task control.
    
    States:
        0: approach waypoint (horizontal midpoint)
        1: approach hover target
        2: descend/settle and rotate wrist down
        3: spray (hold position while spraying)
        4: success hold (post-spray position hold)
        5: fail hold (fallback if timeout/failure)
    """

    def __init__(self):
        """Initialize the FSM state machine."""
        self.state = 0
        self.spray_counter = 0
        self.state_steps = 0
        self.completed = False
        self.timed_out = False
        self.approach_waypoint = None
        self.fail_pos = None
    
    def _advance(self, next_state: int) -> None:
        """Transition to next state."""
        self.state = next_state
        self.state_steps = 0
    
    def _position_command(self, error_vector: np.ndarray) -> np.ndarray:
        """Compute position command with clamped magnitude.
        
        Args:
            error_vector: Position error to correct
            
        Returns:
            Clamped position command
        """
        return cap_vector_norm(POSITION_GAIN * error_vector, ACTION_CLAMP)
    
    def compute_action(self, ee_pos: np.ndarray, ee_quat: np.ndarray,
                      hover_target: np.ndarray) -> np.ndarray:
        """Compute control action based on current FSM state.
        
        Args:
            ee_pos: End effector position in world frame
            ee_quat: End effector quaternion (w, x, y, z) - not currently used
            hover_target: Target position above crown to approach
            
        Returns:
            7-DOF action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action = np.zeros(7)
        action[6] = 0.0  # Gripper always closed
        
        # Initialize approach waypoint if needed
        if self.approach_waypoint is None:
            self.approach_waypoint = np.copy(hover_target)
            # Midpoint between current EE and target (horizontally)
            self.approach_waypoint[0] = (ee_pos[0] + hover_target[0]) / 2.0
            self.approach_waypoint[1] = (ee_pos[1] + hover_target[1]) / 2.0
        
        self.state_steps += 1
        
        # State 0: Approach horizontal waypoint
        if self.state == 0:
            err_to_waypoint = self.approach_waypoint - ee_pos
            action[0:3] = self._position_command(err_to_waypoint)
            if np.linalg.norm(err_to_waypoint) < FSM_POSITION_THRESHOLD:
                self._advance(1)
            elif self.state_steps >= FSM_MAX_STATE_STEPS:
                self.timed_out = True
                self.fail_pos = np.copy(ee_pos)
                self._advance(5)
        
        # State 1: Approach hover target
        elif self.state == 1:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)
            if np.linalg.norm(err_to_hover) < FSM_POSITION_THRESHOLD:
                self._advance(2)
            elif self.state_steps >= FSM_MAX_STATE_STEPS:
                self.timed_out = True
                self.fail_pos = np.copy(ee_pos)
                self._advance(5)
        
        # State 2: Settle (descend + rotate wrist down)
        elif self.state == 2:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)
            # Wrist rotation handled by external joint override
            if self.state_steps >= 90:  # 3 seconds @ 30Hz
                self.spray_counter = SPRAY_DURATION
                self._advance(3)
        
        # State 3: Spray
        elif self.state == 3:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)
            self.spray_counter -= 1
            if self.spray_counter <= 0:
                self.completed = True
                self._advance(4)
        
        # State 4: Success hold
        elif self.state == 4:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)
        
        # State 5: Fail hold
        elif self.state == 5:
            if self.fail_pos is not None:
                err_to_fail = self.fail_pos - ee_pos
                action[0:3] = self._position_command(err_to_fail)
        
        return action
    
    def reset(self) -> None:
        """Reset FSM state for new episode."""
        self.state = 0
        self.spray_counter = 0
        self.state_steps = 0
        self.completed = False
        self.timed_out = False
        self.approach_waypoint = None
        self.fail_pos = None
