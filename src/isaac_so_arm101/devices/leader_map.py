"""Map SO-ARM101 leader motor readings onto PingTi joint targets.

Ported from origin/tele-op's 6-D JointPosition path without merging that
branch. Leader motors use LeRobot's default Feetech ranges
(RANGE_M100_100 for arm, RANGE_0_100 for gripper). PingTi targets are
radians in ``PINGTI_JOINTS`` order (arm then gripper).
"""

from __future__ import annotations

from isaac_so_arm101.teleop_constants import (
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_GRIPPER_JOINT,
    PINGTI_JOINT_LIMITS_RAD,
    PINGTI_JOINT_TO_FOLLOWER_MOTORS,
    PINGTI_JOINTS,
    SO101_LEADER_ARM_RANGE,
    SO101_LEADER_GRIPPER_RANGE,
    SO101_LEADER_MOTORS,
    SO101_TO_PINGTI,
)


def strip_leader_keys(state: dict[str, float]) -> dict[str, float]:
    """Accept ``shoulder_pan`` or LeRobot ``shoulder_pan.pos`` keys."""
    out: dict[str, float] = {}
    for key, value in state.items():
        name = key[:-4] if key.endswith(".pos") else key
        out[name] = float(value)
    return out


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def map_signed_m100(value: float, lo: float, hi: float) -> float:
    """Zero-preserving map: leader 0 → joint 0, ±100 → URDF limits."""
    src_lo, src_hi = SO101_LEADER_ARM_RANGE
    x = _clip(float(value), src_lo, src_hi)
    if x >= 0.0:
        return (x / src_hi) * hi if src_hi else 0.0
    return (x / src_lo) * lo if src_lo else 0.0


def map_gripper_0_100(value: float, lo: float, hi: float) -> float:
    src_lo, src_hi = SO101_LEADER_GRIPPER_RANGE
    x = _clip(float(value), src_lo, src_hi)
    span = src_hi - src_lo
    if span == 0.0:
        return lo
    t = (x - src_lo) / span
    return lo + t * (hi - lo)


def rad_to_signed_m100(rad: float, lo: float, hi: float) -> float:
    """Inverse of ``map_signed_m100`` (joint rad → leader/follower ±100)."""
    x = _clip(float(rad), lo, hi)
    if x >= 0.0:
        return (x / hi) * SO101_LEADER_ARM_RANGE[1] if hi else 0.0
    return (x / lo) * SO101_LEADER_ARM_RANGE[0] if lo else 0.0


def rad_to_gripper_0_100(rad: float, lo: float, hi: float) -> float:
    x = _clip(float(rad), lo, hi)
    span = hi - lo
    if span == 0.0:
        return SO101_LEADER_GRIPPER_RANGE[0]
    t = (x - lo) / span
    src_lo, src_hi = SO101_LEADER_GRIPPER_RANGE
    return src_lo + t * (src_hi - src_lo)


def leader_state_hold(values: dict[str, float] | None = None) -> dict[str, float]:
    """LeRobot-style ``{motor}.pos`` dict; unspecified motors stay at 0."""
    state = {f"{name}.pos": 0.0 for name in SO101_LEADER_MOTORS}
    if values:
        for key, value in values.items():
            name = key[:-4] if key.endswith(".pos") else key
            if name not in SO101_TO_PINGTI:
                raise KeyError(f"unknown leader motor {name!r}")
            state[f"{name}.pos"] = float(value)
    return state


def pingti_joint_pos_from_leader(state: dict[str, float]) -> tuple[float, ...]:
    """Return 6 PingTi joint targets (rad) in ``PINGTI_JOINTS`` order."""
    motors = strip_leader_keys(state)
    missing = [name for name in SO101_LEADER_MOTORS if name not in motors]
    if missing:
        raise KeyError(f"leader state missing motors {missing}; got {sorted(motors)}")
    targets: list[float] = []
    for motor in SO101_LEADER_MOTORS:
        pingti = SO101_TO_PINGTI[motor]
        lo, hi = PINGTI_JOINT_LIMITS_RAD[pingti]
        raw = motors[motor]
        if pingti == PINGTI_GRIPPER_JOINT:
            mapped = map_gripper_0_100(raw, lo, hi)
        else:
            mapped = map_signed_m100(raw, lo, hi)
        targets.append(_clip(mapped, lo, hi))
    if tuple(SO101_TO_PINGTI[m] for m in SO101_LEADER_MOTORS) != PINGTI_JOINTS:
        raise RuntimeError("SO101_TO_PINGTI order drifted from PINGTI_JOINTS")
    return tuple(targets)


def leader_action_from_state(state: dict[str, float]) -> dict[str, float]:
    """LeRobot SO101 follower ``send_action`` dict (``{motor}.pos``)."""
    motors = strip_leader_keys(state)
    missing = [name for name in SO101_LEADER_MOTORS if name not in motors]
    if missing:
        raise KeyError(f"leader state missing motors {missing}; got {sorted(motors)}")
    return {f"{name}.pos": float(motors[name]) for name in SO101_LEADER_MOTORS}


def pingti_follower_action_from_joints(joints: tuple[float, ...] | list[float]) -> dict[str, float]:
    """Expand 6 PingTi URDF radians into 8 Feetech ``{motor}.pos`` goals.

    Dual-drive joints (shoulder_pitch, elbow_pitch) get the **same** command on
    both motors. Units match LeRobot RANGE_M100_100 (arm) / RANGE_0_100 (gripper).
    """
    if len(joints) != len(PINGTI_JOINTS):
        raise ValueError(f"expected {len(PINGTI_JOINTS)} PingTi joints, got {len(joints)}")
    action: dict[str, float] = {}
    for joint, rad in zip(PINGTI_JOINTS, joints, strict=True):
        lo, hi = PINGTI_JOINT_LIMITS_RAD[joint]
        if joint == PINGTI_GRIPPER_JOINT:
            norm = rad_to_gripper_0_100(rad, lo, hi)
        else:
            norm = rad_to_signed_m100(rad, lo, hi)
        motors = PINGTI_JOINT_TO_FOLLOWER_MOTORS[joint]
        for motor in motors:
            action[f"{motor}.pos"] = float(norm)
    missing = [name for name in PINGTI_FOLLOWER_MOTORS if f"{name}.pos" not in action]
    if missing:
        raise RuntimeError(f"PingTi follower action missing {missing}")
    return action


def joints6_from_named(positions: dict[str, float]) -> tuple[float, ...]:
    missing = [name for name in PINGTI_JOINTS if name not in positions]
    if missing:
        raise KeyError(f"sim joints missing {missing}; got {sorted(positions)}")
    return tuple(float(positions[name]) for name in PINGTI_JOINTS)
