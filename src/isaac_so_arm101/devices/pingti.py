"""PingTi follower: LeRobot SOFollower with an 8-motor Feetech bus.

PingTi URDF/sim has 6 joints. Hardware has 8 motors matching pingti_lerobot_bridge
(SO101 names, ids 1–8). Dual-drive secondaries are mechanically opposite, so
``shoulder_lift`` / ``elbow_flex`` get ``-val`` on ``*_secondary``. Connect /
calibrate / disconnect stay on LeRobot ``SOFollower``; this module replaces the
6-motor SO101 bus and the dual-drive configure/PID table.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import Any

from isaac_so_arm101.devices.leader_map import (
    clamp_gripper_feetech,
    pingti_follower_action_from_joints,
    pingti_follower_action_from_leader,
)
from isaac_so_arm101.devices.so101 import (
    FollowerLike,
    LEROBOT_INSTALL_HINT,
    _first_import,
    _make_config,
    _require_lerobot,
    connect_lerobot_device,
)
from isaac_so_arm101.teleop_constants import (
    GRIPPER_FEETECH_CLOSED_FLOOR,
    GRIPPER_FEETECH_CLOSED_HOLD,
    PINGTI_FOLLOWER_MOTOR_IDS,
    PINGTI_FOLLOWER_MOTOR_MODELS,
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_MIRROR_PRIMARY,
)

# From Present, not last Goal. 0.35 was below the STS3250 / loaded-joint deadband
# (lift/elbow/wrist_flex Goal sat 0.35 from Present and never moved). 3.5 from
# Present is ~6° and cannot slam: the motor never sees more than 3.5 units of error.
GOAL_SLEW_MAX = 3.5
NUDGE_DELTA = 2.0
NUDGE_MOTOR = "wrist_roll"
PINGTI_SEND_EVERY_STEPS = 4
GRIPPER_CLOSED_FLOOR = GRIPPER_FEETECH_CLOSED_FLOOR


def gripper_goal_from_present(target: float, present: float | None) -> float:
    """Closed leader/sim must not pull id 8 past the jaws stop.

    Floor Goal at ``GRIPPER_FEETECH_CLOSED_FLOOR``. If Present is already in the
    closed band, keep Present (or the floor if Present is still 0 from cal).
    """
    goal = clamp_gripper_feetech(target)
    if present is None:
        return goal
    pre = float(present)
    if pre <= GRIPPER_FEETECH_CLOSED_HOLD and goal <= GRIPPER_FEETECH_CLOSED_HOLD:
        return clamp_gripper_feetech(pre)
    return goal


def slew_goal(current: float, target: float, max_step: float = GOAL_SLEW_MAX) -> float:
    """Limit one Goal_Position step. Large jumps trip Feetech VIN on the 12 V rail."""
    delta = float(target) - float(current)
    if abs(delta) <= max_step:
        return float(target)
    return float(current) + max_step * (1.0 if delta > 0.0 else -1.0)


def slew_goals(
    last: dict[str, float],
    target: dict[str, float],
    max_step: float = GOAL_SLEW_MAX,
) -> dict[str, float]:
    """Per-motor slew from ``last`` toward ``target`` (no motor jumps more than ``max_step``)."""
    out: dict[str, float] = {}
    for name, value in target.items():
        out[name] = slew_goal(float(last.get(name, value)), float(value), max_step)
    return out


def iter_slew_path(
    start: dict[str, float],
    target: dict[str, float],
    max_step: float = GOAL_SLEW_MAX,
    max_iters: int = 10_000,
):
    """Yield successive slewed goal dicts until ``start`` reaches ``target``."""
    current = {name: float(value) for name, value in start.items()}
    goal = {name: float(target.get(name, value)) for name, value in current.items()}
    for _ in range(max_iters):
        nxt = slew_goals(current, goal, max_step)
        yield dict(nxt)
        if all(abs(nxt[name] - goal[name]) < 1e-9 for name in nxt):
            return
        current = nxt
    raise RuntimeError("slew path did not converge")


def plan_nudge_and_return(
    present: dict[str, float],
    motor: str = NUDGE_MOTOR,
    delta: float = NUDGE_DELTA,
    max_step: float = GOAL_SLEW_MAX,
) -> list[dict[str, float]]:
    """Tiny one-motor nudge then back. Every consecutive step is ≤ ``max_step``."""
    if motor not in present:
        raise KeyError(f"nudge motor {motor!r} not in present {sorted(present)}")
    if abs(float(delta)) > 5.0:
        raise ValueError(f"nudge delta {delta} is too large; keep it under 5 normalized units")
    origin = {name: float(value) for name, value in present.items()}
    bumped = dict(origin)
    bumped[motor] = origin[motor] + float(delta)
    out = list(iter_slew_path(origin, bumped, max_step))
    last = out[-1] if out else origin
    out.extend(iter_slew_path(last, origin, max_step))
    return out


class MockPingTiFollower:
    """Record 8-motor ``send_action`` dicts (no serial)."""

    def __init__(self):
        self.sent: list[dict[str, float]] = []

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        payload = pingti_follower_action_from_leader(action)
        self.sent.append(payload)
        return dict(payload)

    def send_joints(self, joints: tuple[float, ...] | list[float]) -> dict[str, float]:
        return self.send_action(pingti_follower_action_from_joints(joints))

    def disconnect(self) -> None:
        return None


class PingTiFollowerSession:
    def __init__(self, inner: FollowerLike, *, port: str | None = None):
        self._inner = inner
        self._port = port or getattr(getattr(inner, "config", None), "port", None)
        self._shutdown_done = False
        self.last_snapshot: dict[str, Any] | None = None

    def send_action(self, action: dict[str, float]) -> Any:
        return self._inner.send_action(action)

    def send_joints(self, joints: tuple[float, ...] | list[float]) -> Any:
        send_joints = getattr(self._inner, "send_joints", None)
        if callable(send_joints):
            return send_joints(joints)
        return self.send_action(pingti_follower_action_from_joints(joints))

    def hw_snapshot(self) -> dict[str, Any] | None:
        """Read Goal_Position, Present_Position, Torque_Enable (normalized where LeRobot does)."""
        bus = getattr(self._inner, "bus", None)
        if bus is None:
            return None
        present = _seq_read(bus, "Present_Position", normalize=True)
        goal = _seq_read(bus, "Goal_Position", normalize=True)
        torque = _seq_read(bus, "Torque_Enable", normalize=False)
        lock = _seq_read(bus, "Lock", normalize=False)
        self.last_snapshot = {
            "present": {k: float(v) for k, v in present.items()},
            "goal": {k: float(v) for k, v in goal.items()},
            "present_raw": {},
            "goal_raw": {},
            "torque": {k: int(v) for k, v in torque.items()},
            "mode": {},
            "lock": {k: int(v) for k, v in lock.items()},
            "moving": {k: -1 for k in bus.motors},
        }
        return self.last_snapshot

    def ensure_torque_on(self) -> dict[str, int] | None:
        bus = getattr(self._inner, "bus", None)
        if bus is None:
            return None
        torque, present = hold_present_enable_torque(bus)
        require_full_torque(bus, torque, label="PingTi re-enable")
        setattr(self._inner, "_last_goal", present)
        return torque

    def apply_nudge_and_return(
        self,
        motor: str = NUDGE_MOTOR,
        delta: float = NUDGE_DELTA,
        pause_s: float = 0.05,
    ) -> list[dict[str, float]]:
        """Slew one motor by ``delta`` then back. Never jumps more than ``GOAL_SLEW_MAX``."""
        last = getattr(self._inner, "_last_goal", None)
        if not isinstance(last, dict) or motor not in last:
            snap = self.hw_snapshot()
            if snap is None or not snap.get("present"):
                raise RuntimeError("cannot nudge: no _last_goal and no Present_Position")
            last = {name: float(value) for name, value in snap["present"].items()}
            setattr(self._inner, "_last_goal", dict(last))
        plan = plan_nudge_and_return(last, motor=motor, delta=delta)
        sent: list[dict[str, float]] = []
        prev = {name: float(value) for name, value in last.items()}
        for goals in plan:
            for name, value in goals.items():
                if abs(float(value) - float(prev.get(name, value))) > GOAL_SLEW_MAX + 1e-6:
                    raise RuntimeError(
                        f"nudge slam blocked motor={name} from={prev.get(name)} to={value} "
                        f"max={GOAL_SLEW_MAX}"
                    )
            payload = {f"{name}.pos": float(value) for name, value in goals.items()}
            out = self.send_action(payload)
            sent.append(dict(out) if isinstance(out, dict) else dict(payload))
            prev = dict(goals)
            time.sleep(pause_s)
        print(
            f"[teleop_hw] pingti_nudge motor={motor} delta={delta:.2f} steps={len(plan)} "
            f"slew_max={GOAL_SLEW_MAX}",
            flush=True,
        )
        return sent

    def close(self) -> None:
        """Drop torque on every PingTi motor, then close the bus. Idempotent.

        LeRobot ``disconnect(disable_torque=True)`` aborts when gripper id 8 is in
        Overload, which used to leave ids 1–7 holding. Per-motor Torque_Enable=0
        plus a raw scservo pass still works with id 8 latched.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        print("[teleop] PingTi shutdown: disabling torque", flush=True)
        if isinstance(self._inner, MockPingTiFollower):
            print("[teleop_hw] pingti_shutdown mock", flush=True)
            return
        bus = getattr(self._inner, "bus", None)
        if bus is not None:
            for motor in list(bus.motors):
                try:
                    bus.write("Torque_Enable", motor, 0, num_retry=3)
                    print(f"[teleop_hw] pingti_shutdown motor={motor} Torque_Enable=0", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[teleop_hw] pingti_shutdown motor={motor} err={exc!r}", flush=True)
        cfg = getattr(self._inner, "config", None)
        if cfg is not None and hasattr(cfg, "disable_torque_on_disconnect"):
            cfg.disable_torque_on_disconnect = False
        disconnect = getattr(self._inner, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"[teleop_hw] pingti_disconnect_failed err={exc!r}", flush=True)
                handler = getattr(bus, "port_handler", None) if bus is not None else None
                if handler is not None:
                    try:
                        handler.closePort()
                    except Exception as cexc:  # noqa: BLE001
                        print(f"[teleop_hw] pingti_port_close_failed err={cexc!r}", flush=True)
        if self._port:
            torque = disable_pingti_torque_raw(self._port, fatal=False)
            off = [mid for mid, val in torque.items() if int(val) == 0]
            print(f"[teleop] PingTi shutdown complete torque_off_ids={off}", flush=True)


def _seq_read(bus, data_name: str, *, normalize: bool = True, num_retry: int = 4) -> dict[str, Any]:
    """Feetech 8-motor sync_read often returns no status packet; read one id at a time.

    One overloaded motor (gripper id 8) must not abort the other seven reads.
    """
    out: dict[str, Any] = {}
    for motor in bus.motors:
        try:
            out[motor] = bus.read(data_name, motor, normalize=normalize, num_retry=num_retry)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[teleop_hw] read_failed name={data_name} motor={motor} err={exc!r}",
                flush=True,
            )
    return out


def disable_pingti_torque_raw(
    port: str,
    motor_ids=range(1, 9),
    *,
    fatal: bool = True,
) -> dict[int, int]:
    """Write Torque_Enable=0 without a LeRobot handshake (works if id 8 is overloaded)."""
    from scservo_sdk import PacketHandler, PortHandler

    handler = PortHandler(port)
    if not handler.openPort():
        msg = f"[teleop] could not open {port} to disable PingTi torque"
        if fatal:
            raise SystemExit(msg)
        print(msg, flush=True)
        return {}
    try:
        if not handler.setBaudRate(1_000_000):
            msg = f"[teleop] could not set 1 Mbps on {port}"
            if fatal:
                raise SystemExit(msg)
            print(msg, flush=True)
            return {}
        packet = PacketHandler(0)
        torque: dict[int, int] = {}
        for mid in motor_ids:
            comm, err = packet.write1ByteTxRx(handler, int(mid), 40, 0)
            raw, comm_r, _err_r = packet.read1ByteTxRx(handler, int(mid), 40)
            torque[int(mid)] = int(raw) if comm_r == 0 else -1
            print(
                f"[teleop_hw] torque_off id={mid} comm={comm} err={err} torque={torque[int(mid)]}",
                flush=True,
            )
        return torque
    finally:
        handler.closePort()


def hold_present_enable_torque(bus) -> tuple[dict[str, int], dict[str, float]]:
    """STS motors often refuse torque unless Goal_Position already equals Present.

    A leftover Kit session had torque=1, mode=position, Lock=1, Goal_Position=0 on
    every motor while Present was hundreds of ticks away: Goal writes were ignored.
    Unlock first, hold present with acknowledged writes, then torque on. Never leave
    a partial bus torqued.
    """
    bus.disable_torque(num_retry=5)
    time.sleep(0.05)
    present = _seq_read(bus, "Present_Position", normalize=True)
    present_raw = _seq_read(bus, "Present_Position", normalize=False)
    for motor, value in present.items():
        bus.write("Goal_Position", motor, value, num_retry=4)
    time.sleep(0.05)
    # Skip LeRobot enable_torque() — it also writes Lock=1, which blocked Goal
    # writes on this bus. Torque on, EEPROM unlocked.
    for motor in bus.motors:
        bus.write("Torque_Enable", motor, 1, num_retry=5)
        bus.write("Lock", motor, 0, num_retry=5)
    torque = {name: int(v) for name, v in _seq_read(bus, "Torque_Enable", normalize=False).items()}
    for name in bus.motors:
        print(
            f"[teleop_hw] pingti_hold motor={name} present={float(present[name]):.2f} "
            f"present_raw={int(present_raw[name])} torque_en={torque[name]}",
            flush=True,
        )
    return torque, {name: float(present[name]) for name in present}


def require_full_torque(bus, torque: dict[str, int], *, label: str) -> None:
    off = [name for name, val in torque.items() if int(val) != 1]
    if not off:
        return
    bus.disable_torque()
    raise SystemExit(
        f"[teleop] {label} REFUSING: Torque_Enable is not 1 on {off}. "
        "Not enabling a partial PingTi bus."
    )


def _pingti_motors(Motor, MotorNormMode, *, body_norm=None) -> dict:
    if body_norm is None:
        body_norm = MotorNormMode.RANGE_M100_100
    motors = {}
    for name, motor_id in PINGTI_FOLLOWER_MOTOR_IDS.items():
        model = PINGTI_FOLLOWER_MOTOR_MODELS[name]
        if name == "gripper":
            norm = MotorNormMode.RANGE_0_100
        else:
            norm = body_norm
        motors[name] = Motor(motor_id, model, norm)
    return motors


def _feetech_types():
    try:
        motors_mod = importlib.import_module("lerobot.motors")
        feetech_mod = importlib.import_module("lerobot.motors.feetech")
    except ImportError as exc:
        raise SystemExit(f"[teleop] {LEROBOT_INSTALL_HINT}") from exc
    Motor = getattr(motors_mod, "Motor", None)
    MotorNormMode = getattr(motors_mod, "MotorNormMode", None)
    FeetechMotorsBus = getattr(feetech_mod, "FeetechMotorsBus", None)
    OperatingMode = getattr(feetech_mod, "OperatingMode", None)
    missing = [
        name
        for name, obj in (
            ("Motor", Motor),
            ("MotorNormMode", MotorNormMode),
            ("FeetechMotorsBus", FeetechMotorsBus),
            ("OperatingMode", OperatingMode),
        )
        if obj is None
    ]
    if missing:
        raise SystemExit(f"[teleop] lerobot Feetech missing {missing}")
    return Motor, MotorNormMode, FeetechMotorsBus, OperatingMode


def pingti_motor_table() -> dict[str, tuple[int, str]]:
    """Name → (id, LeRobot protocol model). Import-safe for tests without serial."""
    return {
        name: (PINGTI_FOLLOWER_MOTOR_IDS[name], PINGTI_FOLLOWER_MOTOR_MODELS[name])
        for name in PINGTI_FOLLOWER_MOTORS
    }


def make_pingti_follower(
    *,
    port: str,
    robot_id: str = "pingti_follower",
    calibration_dir: str | Path | None = None,
):
    """Construct LeRobot SOFollower with PingTi's 8-motor bus (does not connect)."""
    _require_lerobot()
    so_cls, cfg_cls = _first_import(
        (
            ("lerobot.robots.so_follower", ("SOFollower", "SOFollowerRobotConfig")),
            ("lerobot.robots.so_follower", ("SO101Follower", "SO101FollowerConfig")),
        )
    )
    Motor, MotorNormMode, FeetechMotorsBus, OperatingMode = _feetech_types()

    class PingTiFollower(so_cls):
        """SOFollower whose Feetech bus matches pingti_lerobot_bridge (8 motors)."""

        name = "pingti_follower"

        def __init__(self, config):
            super().__init__(config)
            body_norm = MotorNormMode.DEGREES if getattr(config, "use_degrees", False) else MotorNormMode.RANGE_M100_100
            self.bus = FeetechMotorsBus(
                port=config.port,
                motors=_pingti_motors(Motor, MotorNormMode, body_norm=body_norm),
                calibration=self.calibration,
            )

        def configure(self) -> None:
            # Do not use torque_disabled(): its finally enable_torque() writes
            # Lock=1 with no retries and has dropped status packets on this 8-motor bus.
            self.bus.disable_torque(num_retry=5)
            self.bus.configure_motors()
            for motor in self.bus.motors:
                if self.bus.motors[motor].model == "sts3250":
                    self.bus.write("Maximum_Acceleration", motor, 100, num_retry=4)
                    self.bus.write("Acceleration", motor, 100, num_retry=4)
                    self.bus.write("P_Coefficient", motor, 8, num_retry=4)
                    self.bus.write("I_Coefficient", motor, 0, num_retry=4)
                    self.bus.write("D_Coefficient", motor, 5, num_retry=4)
                else:
                    self.bus.write("Maximum_Acceleration", motor, 254, num_retry=4)
                    self.bus.write("Acceleration", motor, 254, num_retry=4)
                    self.bus.write("P_Coefficient", motor, 16, num_retry=4)
                    self.bus.write("I_Coefficient", motor, 0, num_retry=4)
                    self.bus.write("D_Coefficient", motor, 8, num_retry=4)
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=4)

        def connect(self, calibrate: bool = True) -> None:
            """Handshake without LeRobot's interactive motion calibration.

            Kit teleop has no TTY. Seed RANGE_M100_100 from motor Min/Max/Homing
            registers (or write an existing JSON onto the bus).
            """
            self.bus.connect()
            if not self.calibration:
                cal = self.bus.read_calibration()
                self.calibration = cal
                self.bus.calibration = cal
                save = getattr(self, "_save_calibration", None)
                if callable(save):
                    save()
                print(
                    f"[teleop] PingTi seeded calibration from motor registers "
                    f"path={getattr(self, 'calibration_fpath', None)} motors={sorted(cal)}",
                    flush=True,
                )
            elif not self.bus.is_calibrated:
                print(
                    f"[teleop] PingTi writing existing calibration JSON onto motors "
                    f"id={getattr(self, 'id', None)}",
                    flush=True,
                )
                self.bus.write_calibration(self.calibration)
            for cam in self.cameras.values():
                cam.connect()
            self.configure()
            torque, present = hold_present_enable_torque(self.bus)
            self._last_goal = present
            for motor in self.bus.motors:
                mode = int(self.bus.read("Operating_Mode", motor))
                print(
                    f"[teleop_hw] pingti_connect motor={motor} id={self.bus.motors[motor].id} "
                    f"torque_en={torque[motor]} mode={mode}",
                    flush=True,
                )
            require_full_torque(self.bus, torque, label="PingTi connect")
            print(f"[teleop] PingTi follower connected port={self.config.port}", flush=True)

        def send_action(self, action):
            """Slew then ACK each Goal_Position. Do not use sync_write on this bus.

            LeRobot ``sync_write`` is fire-and-forget. On this mixed STS3215/STS3250
            8-motor chain those packets are dropped: Kit logged follow=ON and
            desired lift=100 while Goal/Present stayed at the hold-present value.
            ``write`` waits for a status packet, which is how the arm moved before.
            """
            wrapped = pingti_follower_action_from_leader(action)
            goal_pos = {
                key.removesuffix(".pos"): val for key, val in wrapped.items() if key.endswith(".pos")
            }
            last = getattr(self, "_last_goal", None)
            if not isinstance(last, dict):
                last = dict(goal_pos)
                self._last_goal = last
            # Slew from Present, not last commanded Goal. Walking Goal away from a
            # stalled Present left lift Goal≈+55 while Present stayed ≈-61.
            try:
                present = _seq_read(self.bus, "Present_Position", normalize=True, num_retry=2)
            except Exception as exc:  # noqa: BLE001
                print(f"[teleop_hw] present_read_failed err={exc!r}; slew from last Goal", flush=True)
                present = {}
            base = {name: float(present.get(name, last.get(name, value))) for name, value in goal_pos.items()}
            slewed = slew_goals(base, goal_pos)
            # Duals must stay exact opposites of the slewed primary. Independent
            # slew on the secondary leaves a few units of mismatch and the pair fights
            # (lift/elbow Present never left the hold pose).
            for primary in PINGTI_MIRROR_PRIMARY:
                sec = f"{primary}_secondary"
                if primary in slewed:
                    slewed[sec] = -float(slewed[primary])
            sent: dict[str, float] = {}
            wrote = 0
            skip_gripper = bool(getattr(self, "_gripper_overload", False))
            for motor, clipped in slewed.items():
                if motor == "gripper":
                    clipped = gripper_goal_from_present(clipped, present.get("gripper"))
                    if skip_gripper:
                        sent[f"{motor}.pos"] = last.get(motor, clipped)
                        continue
                try:
                    self.bus.write("Goal_Position", motor, clipped, num_retry=2)
                    last[motor] = clipped
                    wrote += 1
                except Exception as wexc:  # noqa: BLE001
                    print(
                        f"[teleop_hw] write_failed motor={motor} target={goal_pos.get(motor, clipped):.2f} "
                        f"clipped={clipped:.2f} err={wexc!r}",
                        flush=True,
                    )
                    if motor == "gripper" and "overload" in str(wexc).lower():
                        self._gripper_overload = True
                        print(
                            "[teleop_hw] gripper_overload id=8; skipping further gripper Goal writes. "
                            "Feetech latches Overload until torque-off or power-cycle.",
                            flush=True,
                        )
                        try:
                            self.bus.write("Torque_Enable", "gripper", 0, num_retry=2)
                        except Exception as texc:  # noqa: BLE001
                            print(f"[teleop_hw] gripper_torque_off_failed err={texc!r}", flush=True)
                sent[f"{motor}.pos"] = last.get(motor, clipped)
            n = int(getattr(self, "_send_count", 0)) + 1
            self._send_count = n
            if n <= 8 or n % 20 == 0:
                print(
                    f"[teleop_hw] send_seq n={n} wrote={wrote}/{len(slewed)} "
                    f"lift={sent.get('shoulder_lift.pos')} "
                    f"pan={sent.get('shoulder_pan.pos')} "
                    f"roll={sent.get('wrist_roll.pos')}",
                    flush=True,
                )
            if wrote == 0:
                raise RuntimeError("PingTi Goal_Position write failed on every motor")
            return sent

    cfg = _make_config(cfg_cls, port=port, robot_id=robot_id, calibration_dir=calibration_dir)
    return PingTiFollower(cfg)


def open_pingti_follower(
    *,
    port: str,
    robot_id: str = "pingti_follower",
    recalibrate: bool = False,
    mock: bool = False,
) -> PingTiFollowerSession:
    if mock:
        print("[teleop] using MockPingTiFollower (no serial)", flush=True)
        return PingTiFollowerSession(MockPingTiFollower(), port=port)
    device = make_pingti_follower(port=port, robot_id=robot_id)
    connect_lerobot_device(
        device,
        port=port,
        label="PingTi follower",
        recalibrate=recalibrate,
        motor_ids=range(1, 9),
    )
    return PingTiFollowerSession(device, port=port)
