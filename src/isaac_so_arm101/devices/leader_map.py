"""Map SO-ARM101 leader motor readings onto PingTi joint targets.

Ported from origin/tele-op's 6-D JointPosition path without merging that
branch. Leader motors use LeRobot's default Feetech ranges
(RANGE_M100_100 for arm, RANGE_0_100 for gripper). PingTi targets are
radians in ``PINGTI_JOINTS`` order (arm then gripper).
"""

from __future__ import annotations

import math

from isaac_so_arm101.teleop_constants import (
    GRIPPER_FEETECH_CLOSED_FLOOR,
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_GRIPPER_JOINT,
    PINGTI_JOINT_LIMITS_RAD,
    PINGTI_JOINTS,
    PINGTI_MIRROR_PRIMARY,
    SO101_LEADER_ARM_RANGE,
    SO101_LEADER_GRIPPER_RANGE,
    SO101_LEADER_MOTORS,
    SO101_PINGTI_SIGN,
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
    """Return 6 PingTi joint targets (rad) in ``PINGTI_JOINTS`` order.

    Arm joints apply ``SO101_PINGTI_SIGN`` then clip to URDF limits so a
    reversed leader axis cannot command past the PingTi stop.
    """
    motors = strip_leader_keys(state)
    missing = [name for name in SO101_LEADER_MOTORS if name not in motors]
    if missing:
        raise KeyError(f"leader state missing motors {missing}; got {sorted(motors)}")
    targets: list[float] = []
    for motor in SO101_LEADER_MOTORS:
        pingti = SO101_TO_PINGTI[motor]
        lo, hi = PINGTI_JOINT_LIMITS_RAD[pingti]
        raw = motors[motor]
        sign = SO101_PINGTI_SIGN[motor]
        if pingti == PINGTI_GRIPPER_JOINT:
            mapped = map_gripper_0_100(raw, lo, hi)
        else:
            mapped = sign * map_signed_m100(raw, lo, hi)
        targets.append(_clip(mapped, lo, hi))
    if tuple(SO101_TO_PINGTI[m] for m in SO101_LEADER_MOTORS) != PINGTI_JOINTS:
        raise RuntimeError("SO101_TO_PINGTI order drifted from PINGTI_JOINTS")
    return tuple(targets)


def pingti_named_joints_from_leader(state: dict[str, float]) -> dict[str, float]:
    """Leader dict → ``{pingti_joint: rad}`` (already clipped to URDF limits)."""
    return {name: value for name, value in zip(PINGTI_JOINTS, pingti_joint_pos_from_leader(state), strict=True)}


def _unclipped_mapped_rad(motor: str, raw: float) -> float:
    pingti = SO101_TO_PINGTI[motor]
    lo, hi = PINGTI_JOINT_LIMITS_RAD[pingti]
    sign = SO101_PINGTI_SIGN[motor]
    if pingti == PINGTI_GRIPPER_JOINT:
        return map_gripper_0_100(raw, lo, hi)
    return sign * map_signed_m100(raw, lo, hi)


def clipped_joint_report(state: dict[str, float], atol: float = 1e-5) -> list[str]:
    """PingTi joints whose signed leader map is outside the URDF stop."""
    motors = strip_leader_keys(state)
    over: list[str] = []
    for motor in SO101_LEADER_MOTORS:
        if motor not in motors:
            continue
        pingti = SO101_TO_PINGTI[motor]
        lo, hi = PINGTI_JOINT_LIMITS_RAD[pingti]
        unclipped = _unclipped_mapped_rad(motor, motors[motor])
        if unclipped < lo - atol or unclipped > hi + atol:
            over.append(pingti)
    return over


def leader_action_from_state(state: dict[str, float]) -> dict[str, float]:
    """LeRobot SO101 follower ``send_action`` dict (``{motor}.pos``)."""
    motors = strip_leader_keys(state)
    missing = [name for name in SO101_LEADER_MOTORS if name not in motors]
    if missing:
        raise KeyError(f"leader state missing motors {missing}; got {sorted(motors)}")
    return {f"{name}.pos": float(motors[name]) for name in SO101_LEADER_MOTORS}


def clamp_gripper_feetech(value: float) -> float:
    """Keep PingTi gripper Goal off the 0–4095 calibration extreme (closed stop)."""
    return _clip(float(value), GRIPPER_FEETECH_CLOSED_FLOOR, SO101_LEADER_GRIPPER_RANGE[1])


def pingti_follower_action_from_leader(state: dict[str, float]) -> dict[str, float]:
    """6 SO101-named primaries → 8 PingTi Feetech goals (bridge dual-drive table).

    ``shoulder_lift`` / ``elbow_flex`` secondaries are mechanically opposite, so
    they get ``-val`` (RANGE_M100_100), matching pingti_lerobot_bridge.
    """
    primary = leader_action_from_state(state)
    motors = strip_leader_keys(primary)
    action: dict[str, float] = {}
    for name in PINGTI_FOLLOWER_MOTORS:
        if name.endswith("_secondary"):
            parent = name.removesuffix("_secondary")
            if parent not in PINGTI_MIRROR_PRIMARY:
                raise RuntimeError(f"unexpected secondary motor {name}")
            action[f"{name}.pos"] = -float(motors[parent])
        else:
            action[f"{name}.pos"] = float(motors[name])
    return action


def urdf_rad_to_feetech_m100(rad: float) -> float:
    """Sim/URDF radians → Feetech RANGE_M100_100.

    Seeded PingTi calibration is range 0–4095, so LeRobot ±100 is a half turn
    (±π rad), not the URDF stop. Mapping through URDF limits sent ~2× the pan
    command (limit ±π/2 vs motor ±π) and the real pan/roll overshot sim and the
    SO-101.
    """
    x = float(rad) / math.pi * SO101_LEADER_ARM_RANGE[1]
    return _clip(x, SO101_LEADER_ARM_RANGE[0], SO101_LEADER_ARM_RANGE[1])


def pingti_follower_action_from_joints(joints: tuple[float, ...] | list[float]) -> dict[str, float]:
    """Sim/URDF 6 rad → 8 Feetech goals.

    Scale is ``rad/π×100`` (calibration 0–4095). Sign undoes ``SO101_PINGTI_SIGN``
    so the real PingTi matches the SO-101: pan/lift/elbow/wrist_flex are inverted
    in sim to look right, but Feetech polarity matches the leader. Wrist_roll and
    gripper are already ``+1`` and stay unchanged.
    """
    if len(joints) != len(PINGTI_JOINTS):
        raise ValueError(f"expected {len(PINGTI_JOINTS)} PingTi joints, got {len(joints)}")
    state: dict[str, float] = {}
    for motor, joint, rad in zip(SO101_LEADER_MOTORS, PINGTI_JOINTS, joints, strict=True):
        lo, hi = PINGTI_JOINT_LIMITS_RAD[joint]
        if joint == PINGTI_GRIPPER_JOINT:
            state[motor] = clamp_gripper_feetech(rad_to_gripper_0_100(rad, lo, hi))
        else:
            # Undo SO101_PINGTI_SIGN: sim rad is already flipped for those axes.
            state[motor] = SO101_PINGTI_SIGN[motor] * urdf_rad_to_feetech_m100(rad)
    return pingti_follower_action_from_leader(state)


def follow_error_report(
    joints: tuple[float, ...] | list[float],
    present: dict[str, float],
    goal: dict[str, float] | None = None,
) -> list[str]:
    """Compare sim-mapped Feetech targets to real Present/Goal (normalized ±100 / 0–100)."""
    desired = pingti_follower_action_from_joints(joints)
    lines: list[str] = []
    for motor in PINGTI_FOLLOWER_MOTORS:
        des = desired.get(f"{motor}.pos")
        pre = present.get(motor)
        gol = (goal or {}).get(motor)
        err = None
        if des is not None and pre is not None:
            err = float(pre) - float(des)
        des_s = f"{float(des):.2f}" if des is not None else "na"
        pre_s = f"{float(pre):.2f}" if pre is not None else "na"
        gol_s = f"{float(gol):.2f}" if gol is not None else "na"
        err_s = f"{err:+.2f}" if err is not None else "na"
        lines.append(
            f"[teleop_hw] compare motor={motor} desired={des_s} goal={gol_s} "
            f"present={pre_s} present_minus_desired={err_s}"
        )
    return lines


def joints6_from_named(positions: dict[str, float]) -> tuple[float, ...]:
    missing = [name for name in PINGTI_JOINTS if name not in positions]
    if missing:
        raise KeyError(f"sim joints missing {missing}; got {sorted(positions)}")
    return tuple(float(positions[name]) for name in PINGTI_JOINTS)
