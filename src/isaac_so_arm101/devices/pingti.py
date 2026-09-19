"""PingTi follower: LeRobot SOFollower with an 8-motor Feetech bus.

PingTi URDF/sim has 6 joints. Hardware has 8 motors: ``shoulder_pitch`` and
``elbow_pitch`` are dual-drive, so both motors of a pair get the same
``Goal_Position``. Connect / calibrate / ``send_action`` / disconnect are
LeRobot's ``SOFollower`` methods. This module only replaces the 6-motor SO101
bus with PingTi's 8 motors (STS3250 shoulder duals, STS3215 elsewhere).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from isaac_so_arm101.devices.leader_map import pingti_follower_action_from_joints
from isaac_so_arm101.devices.so101 import (
    FollowerLike,
    LEROBOT_INSTALL_HINT,
    _first_import,
    _make_config,
    _require_lerobot,
    connect_lerobot_device,
)
from isaac_so_arm101.teleop_constants import (
    PINGTI_FOLLOWER_MOTOR_IDS,
    PINGTI_FOLLOWER_MOTOR_MODELS,
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_GRIPPER_JOINT,
)


class MockPingTiFollower:
    """Record 8-motor ``send_action`` dicts (no serial)."""

    def __init__(self):
        self.sent: list[dict[str, float]] = []

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        motors = {key.removesuffix(".pos") if key.endswith(".pos") else key: float(val) for key, val in action.items()}
        missing = [name for name in PINGTI_FOLLOWER_MOTORS if name not in motors]
        extra = [name for name in motors if name not in PINGTI_FOLLOWER_MOTORS]
        if missing or extra:
            raise KeyError(f"PingTi send_action expected {list(PINGTI_FOLLOWER_MOTORS)}; missing={missing} extra={extra}")
        payload = {f"{name}.pos": motors[name] for name in PINGTI_FOLLOWER_MOTORS}
        self.sent.append(payload)
        return dict(payload)

    def send_joints(self, joints: tuple[float, ...] | list[float]) -> dict[str, float]:
        return self.send_action(pingti_follower_action_from_joints(joints))

    def disconnect(self) -> None:
        return None


class PingTiFollowerSession:
    def __init__(self, inner: FollowerLike):
        self._inner = inner

    def send_action(self, action: dict[str, float]) -> Any:
        return self._inner.send_action(action)

    def send_joints(self, joints: tuple[float, ...] | list[float]) -> Any:
        send_joints = getattr(self._inner, "send_joints", None)
        if callable(send_joints):
            return send_joints(joints)
        return self.send_action(pingti_follower_action_from_joints(joints))

    def close(self) -> None:
        disconnect = getattr(self._inner, "disconnect", None)
        if callable(disconnect):
            disconnect()


def _pingti_motors(Motor, MotorNormMode, *, body_norm=None) -> dict:
    if body_norm is None:
        body_norm = MotorNormMode.RANGE_M100_100
    motors = {}
    for name, motor_id in PINGTI_FOLLOWER_MOTOR_IDS.items():
        model = PINGTI_FOLLOWER_MOTOR_MODELS[name]
        if name == PINGTI_GRIPPER_JOINT:
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
    Motor, MotorNormMode, FeetechMotorsBus, _ = _feetech_types()

    class PingTiFollower(so_cls):
        """SOFollower whose Feetech bus is PingTi (8 motors), not SO-101 (6)."""

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
            super().configure()
            gripper = PINGTI_GRIPPER_JOINT
            if gripper not in self.bus.motors:
                return
            # SOFollower.configure only special-cases motor name "gripper".
            with self.bus.torque_disabled():
                self.bus.write("Max_Torque_Limit", gripper, 500)
                self.bus.write("Protection_Current", gripper, 250)
                self.bus.write("Overload_Torque", gripper, 25)

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
        return PingTiFollowerSession(MockPingTiFollower())
    device = make_pingti_follower(port=port, robot_id=robot_id)
    connect_lerobot_device(device, port=port, label="PingTi follower", recalibrate=recalibrate)
    return PingTiFollowerSession(device)
