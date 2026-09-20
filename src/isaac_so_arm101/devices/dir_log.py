"""Grep-friendly leader vs sim joint direction logs.

Lines start with ``[teleop_dir]`` so you can ``grep teleop_dir logs/teleop_joint_dir.log``.

``numeric`` compares raw leader Δ vs sim Δ (same sign or opposite).
``visual`` applies ``SO101_PINGTI_SIGN``: ``same`` means the mapped sim axis
moved the way the sign table intends (opposite numeric is expected when sign=-1).
"""

from __future__ import annotations

from pathlib import Path

from isaac_so_arm101.devices.leader_map import strip_leader_keys
from isaac_so_arm101.teleop_constants import (
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_JOINTS,
    SO101_LEADER_MOTORS,
    SO101_PINGTI_SIGN,
    SO101_TO_PINGTI,
)

LEADER_MOVE_EPS = 0.5
SIM_MOVE_EPS = 0.01


def delta_dir(delta: float, eps: float) -> str:
    if abs(delta) < eps:
        return "hold"
    return "+" if delta > 0 else "-"


def pair_status(leader_d: float, sim_d: float, sign: float) -> tuple[str, str]:
    """Return (numeric, visual) in {hold, same, opposite}."""
    if abs(leader_d) < LEADER_MOVE_EPS or abs(sim_d) < SIM_MOVE_EPS:
        return "hold", "hold"
    numeric = "same" if leader_d * sim_d > 0.0 else "opposite"
    visual = "same" if (sign * leader_d) * sim_d > 0.0 else "opposite"
    return numeric, visual


def meas_tracks_cmd(cmd_d: float, meas_d: float) -> str:
    if abs(cmd_d) < SIM_MOVE_EPS or abs(meas_d) < SIM_MOVE_EPS:
        return "hold"
    return "yes" if cmd_d * meas_d > 0.0 else "no"


class JointDirLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_leader: dict[str, float] | None = None
        self._prev_cmd: dict[str, float] | None = None
        self._prev_meas: dict[str, float] | None = None
        self._fh = self.path.open("a", encoding="utf-8")
        signs = " ".join(f"{m}={SO101_PINGTI_SIGN[m]:.0f}" for m in SO101_LEADER_MOTORS)
        self.emit(f"[teleop_dir] open path={self.path} signs={signs}")

    def close(self) -> None:
        if self._fh is None:
            return
        self.emit("[teleop_dir] close")
        self._fh.close()
        self._fh = None

    def emit(self, line: str) -> None:
        print(line, flush=True)
        if self._fh is not None:
            self._fh.write(line + "\n")
            self._fh.flush()

    def step(
        self,
        *,
        step: int,
        raw_leader: dict[str, float],
        sim_cmd: tuple[float, ...] | list[float],
        sim_meas: dict[str, float],
        pingti_action: dict[str, float] | None = None,
        force: bool = False,
    ) -> bool:
        leader = strip_leader_keys(raw_leader)
        cmd = {name: float(value) for name, value in zip(PINGTI_JOINTS, sim_cmd, strict=True)}
        meas = {name: float(sim_meas[name]) for name in PINGTI_JOINTS}
        if self._prev_leader is None:
            self._prev_leader = leader
            self._prev_cmd = cmd
            self._prev_meas = meas
            parts = [
                f"[teleop_dir] step={step} kind=init",
            ]
            for motor in SO101_LEADER_MOTORS:
                joint = SO101_TO_PINGTI[motor]
                parts.append(
                    f"{motor}={leader[motor]:.2f}->{joint} cmd={cmd[joint]:.4f} meas={meas[joint]:.4f}"
                )
            self.emit(" ".join(parts))
            return True

        moved = False
        lines: list[str] = []
        for motor in SO101_LEADER_MOTORS:
            joint = SO101_TO_PINGTI[motor]
            d_lead = leader[motor] - self._prev_leader[motor]
            d_cmd = cmd[joint] - self._prev_cmd[joint]
            d_meas = meas[joint] - self._prev_meas[joint]
            if abs(d_lead) >= LEADER_MOVE_EPS or abs(d_cmd) >= SIM_MOVE_EPS or abs(d_meas) >= SIM_MOVE_EPS:
                moved = True
            sign = SO101_PINGTI_SIGN[motor]
            numeric, visual = pair_status(d_lead, d_cmd, sign)
            track = meas_tracks_cmd(d_cmd, d_meas)
            hw = ""
            if pingti_action is not None:
                key = f"{motor}.pos"
                if key in pingti_action:
                    hw_val = float(pingti_action[key])
                    if abs(leader[motor]) < LEADER_MOVE_EPS or abs(hw_val) < LEADER_MOVE_EPS:
                        hw_vs_leader = "hold"
                    else:
                        hw_vs_leader = "same" if leader[motor] * hw_val > 0.0 else "opposite"
                    hw = f" pingti_{motor}={hw_val:.2f} hw_vs_leader={hw_vs_leader}"
            lines.append(
                f"[teleop_dir] step={step} motor={motor} pingti={joint} sign={sign:.0f} "
                f"leader={leader[motor]:.2f} dL={d_lead:+.2f}({delta_dir(d_lead, LEADER_MOVE_EPS)}) "
                f"sim_cmd={cmd[joint]:.4f} dC={d_cmd:+.4f}({delta_dir(d_cmd, SIM_MOVE_EPS)}) "
                f"sim_meas={meas[joint]:.4f} dM={d_meas:+.4f}({delta_dir(d_meas, SIM_MOVE_EPS)}) "
                f"numeric={numeric} visual={visual} meas_tracks={track}{hw}"
            )

        self._prev_leader = leader
        self._prev_cmd = cmd
        self._prev_meas = meas
        if not (moved or force):
            return False
        for line in lines:
            self.emit(line)
        return True


GOAL_HIT_EPS = 5.0


def format_hw_lines(
    *,
    step: int,
    pingti_action: dict[str, float] | None,
    snap: dict | None,
) -> list[str]:
    """One grep line per PingTi motor: sent vs Goal vs Present vs Torque_Enable."""
    if snap is None:
        return [f"[teleop_hw] step={step} kind=no_bus"]
    sent = pingti_action or {}
    present = snap.get("present") or {}
    goal = snap.get("goal") or {}
    present_raw = snap.get("present_raw") or {}
    goal_raw = snap.get("goal_raw") or {}
    torque = snap.get("torque") or {}
    mode = snap.get("mode") or {}
    moving = snap.get("moving") or {}
    lines: list[str] = []
    for name in PINGTI_FOLLOWER_MOTORS:
        key = f"{name}.pos"
        sent_v = sent.get(key)
        goal_v = goal.get(name)
        present_v = present.get(name)
        sent_s = f"{sent_v:.2f}" if sent_v is not None else "na"
        goal_s = f"{goal_v:.2f}" if goal_v is not None else "na"
        present_s = f"{present_v:.2f}" if present_v is not None else "na"
        if goal_v is None or present_v is None:
            hit = "na"
        else:
            hit = "yes" if abs(goal_v - present_v) < GOAL_HIT_EPS else "no"
        wrote = "na"
        if sent_v is not None and goal_v is not None:
            wrote = "yes" if abs(sent_v - goal_v) < GOAL_HIT_EPS else "no"
        lines.append(
            f"[teleop_hw] step={step} motor={name} sent={sent_s} goal={goal_s} present={present_s} "
            f"goal_raw={goal_raw.get(name, 'na')} present_raw={present_raw.get(name, 'na')} "
            f"torque={torque.get(name, 'na')} lock={snap.get('lock', {}).get(name, 'na')} "
            f"mode={mode.get(name, 'na')} moving={moving.get(name, 'na')} "
            f"goal_wrote={wrote} goal_hit={hit}"
        )
    return lines


def emit_hw_lines(
    *,
    step: int,
    pingti_action: dict[str, float] | None,
    snap: dict | None,
    logger: JointDirLogger | None = None,
) -> None:
    for line in format_hw_lines(step=step, pingti_action=pingti_action, snap=snap):
        if logger is not None:
            logger.emit(line)
        else:
            print(line, flush=True)
