"""Shared leader → follower steps (sim and no-sim)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from isaac_so_arm101.devices.leader_map import (
    joints6_from_named,
    leader_action_from_state,
    pingti_follower_action_from_joints,
    pingti_follower_action_from_leader,
    pingti_joint_pos_from_leader,
)
from isaac_so_arm101.teleop_constants import PINGTI_JOINTS


@dataclass
class LeaderHwStep:
    raw: dict[str, float]
    joints6: tuple[float, ...]
    so101_action: dict[str, float]
    pingti_action: dict[str, float]


@dataclass
class LeaderHwLoopResult:
    steps: int
    last: LeaderHwStep | None
    so101_sends: list[dict[str, float]] = field(default_factory=list)
    pingti_sends: list[dict[str, float]] = field(default_factory=list)


def step_leader_followers(leader, *, so101=None, pingti=None) -> LeaderHwStep:
    """SO101 leader → 6 PingTi rads, optional SO101 follower + 8-motor PingTi."""
    raw = leader.get_action()
    joints6 = pingti_joint_pos_from_leader(raw)
    so101_action = leader_action_from_state(raw)
    pingti_action = pingti_follower_action_from_leader(raw)
    if so101 is not None:
        so101.send_action(so101_action)
    if pingti is not None:
        pingti.send_action(pingti_action)
    return LeaderHwStep(raw, joints6, so101_action, pingti_action)


def pingti_action_from_sim_named(positions: dict[str, float]) -> dict[str, float]:
    """Sim joint name→rad (Isaac ``send_action`` / ``joint_pos_target``) → 8 Feetech goals."""
    return pingti_follower_action_from_joints(joints6_from_named(positions))


def named_joints_from_robot(robot: Any) -> dict[str, float]:
    """Prefer commanded ``joint_pos_target`` (sim send_action), else measured ``joint_pos``."""
    names = list(robot.joint_names)
    data = robot.data
    src = getattr(data, "joint_pos_target", None)
    if src is None:
        src = data.joint_pos
    vec = src[0].detach().cpu()
    named = {name: float(vec[i]) for i, name in enumerate(names)}
    missing = [name for name in PINGTI_JOINTS if name not in named]
    if missing:
        raise KeyError(f"robot joints missing {missing}; got {sorted(named)}")
    return named


def send_sim_joints_to_pingti(robot: Any, pingti) -> dict[str, float]:
    action = pingti_action_from_sim_named(named_joints_from_robot(robot))
    pingti.send_action(action)
    return action


def run_leader_hw_loop(
    leader,
    *,
    so101=None,
    pingti=None,
    steps: int,
    hz: float = 0.0,
) -> LeaderHwLoopResult:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    period = (1.0 / hz) if hz > 0.0 else 0.0
    result = LeaderHwLoopResult(steps=0, last=None)
    for i in range(steps):
        t0 = time.perf_counter()
        step = step_leader_followers(leader, so101=so101, pingti=pingti)
        result.steps = i + 1
        result.last = step
        if so101 is not None:
            result.so101_sends.append(dict(step.so101_action))
        if pingti is not None:
            result.pingti_sends.append(dict(step.pingti_action))
        if period > 0.0:
            remain = period - (time.perf_counter() - t0)
            if remain > 0.0:
                time.sleep(remain)
    return result
