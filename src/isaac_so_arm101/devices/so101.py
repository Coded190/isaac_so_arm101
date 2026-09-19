"""LeRobot SO-ARM101 leader / follower sessions (optional hardware extra).

Do not vendor origin/tele-op Feetech code. Keyboard teleop does not import this
at module load; hardware is opened only for ``--teleop_device so101leader``.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

from isaac_so_arm101.teleop_constants import SO101_LEADER_MOTORS

LEROBOT_INSTALL_HINT = (
    "LeRobot is required for --teleop_device so101leader (not in the default uv lock).\n"
    "  UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv pip install 'lerobot[feetech]'\n"
    "If that command changes torch, restore the Lab 3 pin:\n"
    "  UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv pip install -U "
    "torch==2.11.0 torchvision==0.26.0 "
    "--index-url https://download.pytorch.org/whl/cu128\n"
    "Calibrate once with `lerobot-calibrate` (see README). "
    "Add this user to the `dialout` group for /dev/ttyACM*."
)


class LeaderLike(Protocol):
    def get_action(self) -> dict[str, float]: ...

    def disconnect(self) -> None: ...


class FollowerLike(Protocol):
    def send_action(self, action: dict[str, float]) -> Any: ...

    def disconnect(self) -> None: ...


class MockSO101Leader:
    """Zero pose leader for Kit smoke without a serial adapter."""

    def __init__(self, hold: dict[str, float] | None = None):
        self.hold = hold or {f"{name}.pos": 0.0 for name in SO101_LEADER_MOTORS}

    def get_action(self) -> dict[str, float]:
        return dict(self.hold)

    def disconnect(self) -> None:
        return None


class ScriptedSO101Leader:
    """Replay canned leader frames (same dict shape as LeRobot ``get_action``)."""

    def __init__(self, frames: list[dict[str, float]]):
        if not frames:
            raise ValueError("ScriptedSO101Leader needs at least one frame")
        self.frames = [dict(frame) for frame in frames]
        self.index = 0

    def get_action(self) -> dict[str, float]:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return dict(frame)

    def disconnect(self) -> None:
        return None


class SO101LeaderSession:
    def __init__(self, inner: LeaderLike):
        self._inner = inner

    def get_action(self) -> dict[str, float]:
        return self._inner.get_action()

    def close(self) -> None:
        disconnect = getattr(self._inner, "disconnect", None)
        if callable(disconnect):
            disconnect()


class SO101FollowerSession:
    def __init__(self, inner: FollowerLike):
        self._inner = inner

    def send_action(self, action: dict[str, float]) -> None:
        self._inner.send_action(action)

    def close(self) -> None:
        disconnect = getattr(self._inner, "disconnect", None)
        if callable(disconnect):
            disconnect()


def _first_import(paths: tuple[tuple[str, tuple[str, ...]], ...]):
    errors: list[str] = []
    for module_name, attr_names in paths:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        missing = [name for name in attr_names if not hasattr(module, name)]
        if missing:
            errors.append(f"{module_name} missing {missing}")
            continue
        return tuple(getattr(module, name) for name in attr_names)
    raise ImportError(" ; ".join(errors))


def _require_lerobot() -> None:
    try:
        importlib.import_module("lerobot")
    except ImportError as exc:
        raise SystemExit(f"[teleop] {LEROBOT_INSTALL_HINT}") from exc


def _make_config(cfg_cls, *, port: str, robot_id: str) -> Any:
    # LeRobot 0.6 defaults use_degrees=True. Keep RANGE_M100_100 so the PingTi
    # map (leader 0 → joint 0, ±100 → URDF limits) matches origin/tele-op.
    kwargs_tries = (
        {"port": port, "id": robot_id, "use_degrees": False},
        {"port": port, "id": robot_id},
        {"port": port, "id": robot_id, "cameras": {}},
        {"port": port, "id": robot_id, "cameras": {}, "use_degrees": False},
    )
    last_exc: Exception | None = None
    for kwargs in kwargs_tries:
        try:
            return cfg_cls(**kwargs)
        except TypeError as exc:
            last_exc = exc
    raise TypeError(f"could not construct {cfg_cls}: {last_exc}") from last_exc


def open_so101_leader(
    *,
    port: str,
    robot_id: str = "so101_leader",
    recalibrate: bool = False,
    mock: bool = False,
) -> SO101LeaderSession:
    if mock:
        print("[teleop] using MockSO101Leader (no serial)", flush=True)
        return SO101LeaderSession(MockSO101Leader())
    _require_lerobot()
    cls, cfg_cls = _first_import(
        (
            ("lerobot.teleoperators.so_leader", ("SO101Leader", "SO101LeaderConfig")),
            ("lerobot.teleoperators.so101_leader", ("SO101Leader", "SO101LeaderConfig")),
            ("lerobot.teleoperators.so_leader", ("SOLeader", "SOLeaderTeleopConfig")),
        )
    )
    cfg = _make_config(cfg_cls, port=port, robot_id=robot_id)
    device = cls(cfg)
    connect = getattr(device, "connect", None)
    if not callable(connect):
        raise SystemExit("[teleop] LeRobot leader has no connect()")
    print(f"[teleop] connecting SO101 leader port={port} id={robot_id}", flush=True)
    connect(calibrate=recalibrate)
    if recalibrate and hasattr(device, "calibrate"):
        device.calibrate()
    return SO101LeaderSession(device)


def open_so101_follower(
    *,
    port: str,
    robot_id: str = "so101_follower",
    recalibrate: bool = False,
) -> SO101FollowerSession:
    _require_lerobot()
    cls, cfg_cls = _first_import(
        (
            ("lerobot.robots.so_follower", ("SO101Follower", "SO101FollowerConfig")),
            ("lerobot.robots.so101_follower", ("SO101Follower", "SO101FollowerConfig")),
            ("lerobot.robots.so_follower", ("SOFollower", "SOFollowerRobotConfig")),
        )
    )
    cfg = _make_config(cfg_cls, port=port, robot_id=robot_id)
    device = cls(cfg)
    connect = getattr(device, "connect", None)
    if not callable(connect):
        raise SystemExit("[teleop] LeRobot follower has no connect()")
    print(f"[teleop] connecting SO101 follower port={port} id={robot_id}", flush=True)
    connect(calibrate=recalibrate)
    if recalibrate and hasattr(device, "calibrate"):
        device.calibrate()
    return SO101FollowerSession(device)
