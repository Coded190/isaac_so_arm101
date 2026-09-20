"""LeRobot SO-ARM101 leader / follower sessions (optional hardware extra).

Do not vendor origin/tele-op Feetech code. Keyboard teleop does not import this
at module load; hardware is opened only for ``--teleop_device so101leader``.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from isaac_so_arm101.teleop_constants import SO101_LEADER_MOTORS

# STS3215 operating window (0.1 V register units → volts). Leader supplies
# are typically 5–7.4 V. Above ~14 V sets Feetech VIN and can cook motors.
# Do NOT ignore VIN status — refuse the bus instead.
FEETECH_VIN_ERROR_BIT = 1
FEETECH_VOLTAGE_MIN_V = 4.5
FEETECH_VOLTAGE_MAX_V = 14.0
_FEETECH_PRESENT_VOLTAGE_ADDR = 62


@contextmanager
def _noninteractive_calibration_input() -> Iterator[None]:
    """If stdin is not a TTY, accept a loaded calibration JSON (do not type ``c``)."""
    if sys.stdin.isatty():
        yield
        return
    original = builtins.input

    def _input(prompt: object = "") -> str:
        print(prompt, end="", flush=True)
        print(
            "[teleop] stdin is not a TTY; using loaded calibration file (not recalibrating)",
            flush=True,
        )
        return ""

    builtins.input = _input
    try:
        yield
    finally:
        builtins.input = original

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


class MockSO101Follower:
    """Record SO101 ``send_action`` dicts (no serial)."""

    def __init__(self):
        self.sent: list[dict[str, float]] = []

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        motors = {key.removesuffix(".pos") if key.endswith(".pos") else key: float(val) for key, val in action.items()}
        missing = [name for name in SO101_LEADER_MOTORS if name not in motors]
        extra = [name for name in motors if name not in SO101_LEADER_MOTORS]
        if missing or extra:
            raise KeyError(
                f"SO101 send_action expected {list(SO101_LEADER_MOTORS)}; missing={missing} extra={extra}"
            )
        payload = {f"{name}.pos": motors[name] for name in SO101_LEADER_MOTORS}
        self.sent.append(payload)
        return dict(payload)

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


def _make_config(
    cfg_cls,
    *,
    port: str,
    robot_id: str,
    calibration_dir: str | Path | None = None,
) -> Any:
    # LeRobot 0.6 defaults use_degrees=True. Keep RANGE_M100_100 so the PingTi
    # map (leader 0 → joint 0, ±100 → URDF limits) matches origin/tele-op.
    base: dict[str, Any] = {"port": port, "id": robot_id}
    if calibration_dir is not None:
        base["calibration_dir"] = Path(calibration_dir)
    kwargs_tries = (
        {**base, "use_degrees": False},
        dict(base),
        {**base, "cameras": {}},
        {**base, "cameras": {}, "use_degrees": False},
    )
    last_exc: Exception | None = None
    for kwargs in kwargs_tries:
        try:
            return cfg_cls(**kwargs)
        except TypeError as exc:
            last_exc = exc
    raise TypeError(f"could not construct {cfg_cls}: {last_exc}") from last_exc


def read_feetech_bus_voltages(
    port: str, motor_ids: range | list[int] | None = None
) -> list[tuple[int, float, int]]:
    """Return ``(id, volts, error_status)`` for each pingable motor. Closes the port."""
    if motor_ids is None:
        motor_ids = range(1, 7)
    try:
        from scservo_sdk import PortHandler, PacketHandler
    except ImportError as exc:
        raise SystemExit(
            f"[teleop] scservo_sdk is required to check Feetech bus voltage on {port}. "
            "Install with `uv pip install 'lerobot[feetech]'`."
        ) from exc

    handler = PortHandler(port)
    if not handler.openPort():
        raise SystemExit(f"[teleop] could not open {port} to read motor voltage")
    try:
        if not handler.setBaudRate(1_000_000):
            raise SystemExit(f"[teleop] could not set 1 Mbps on {port}")
        packet = PacketHandler(0)
        rows: list[tuple[int, float, int]] = []
        for motor_id in motor_ids:
            _model, comm, err = packet.ping(handler, motor_id)
            if comm != 0:
                continue
            raw, comm_v, _err_v = packet.read1ByteTxRx(handler, motor_id, _FEETECH_PRESENT_VOLTAGE_ADDR)
            if comm_v != 0:
                continue
            rows.append((int(motor_id), float(raw) / 10.0, int(err)))
        return rows
    finally:
        handler.closePort()


def assert_feetech_bus_voltage_ok(
    port: str, label: str, motor_ids: range | list[int] | None = None
) -> None:
    """Refuse connect if VIN is set or voltage is outside the STS3215 window.

    This is a safety interlock. A 15 V adapter previously set VIN on every
    motor; ignoring that bit would keep teleop running on an overloaded bus.
    """
    rows = read_feetech_bus_voltages(port, motor_ids=motor_ids)
    if not rows:
        raise SystemExit(
            f"[teleop] {label} voltage preflight found no STS motors on {port}. "
            "Check USB, 1 Mbps bus, and motor power."
        )
    vin_ids = [mid for mid, _volt, err in rows if err & FEETECH_VIN_ERROR_BIT]
    oos = [(mid, volt) for mid, volt, _err in rows if volt < FEETECH_VOLTAGE_MIN_V or volt > FEETECH_VOLTAGE_MAX_V]
    summary = " ".join(f"id{mid}={volt:.1f}V" for mid, volt, _err in rows)
    print(f"[teleop] {label} bus voltage {summary}", flush=True)
    if vin_ids or oos:
        raise SystemExit(
            f"[teleop] {label} REFUSING connect: Feetech VIN/over-voltage on {port}. "
            f"vin_ids={vin_ids} out_of_spec={oos} "
            f"allowed=[{FEETECH_VOLTAGE_MIN_V:.1f}, {FEETECH_VOLTAGE_MAX_V:.1f}] V. "
            "Supply must be inside the STS window (not an unregulated 15 V adapter)."
        )


def connect_lerobot_device(
    device: Any,
    *,
    port: str,
    label: str,
    recalibrate: bool = False,
    motor_ids: range | list[int] | None = None,
) -> None:
    """Use LeRobot's connect/calibrate path. Do not skip calibration.

    ``connect(calibrate=True)`` is what SOLeader/SOFollower expect: handshake the
    Feetech bus, load ``~/.cache/huggingface/lerobot/.../{id}.json``, and prompt
    if the file is missing or does not match the motors. ``get_action`` /
    ``send_action`` then ``sync_read`` / ``sync_write`` ``Present_Position`` /
    ``Goal_Position`` with LeRobot's RANGE_M100_100 unnormalize.
    """
    connect = getattr(device, "connect", None)
    if not callable(connect):
        raise SystemExit(f"[teleop] LeRobot {label} has no connect()")
    log_calibration_file(device, label)
    print(f"[teleop] connecting {label} port={port} id={getattr(device, 'id', None)}", flush=True)
    assert_feetech_bus_voltage_ok(port, label, motor_ids=motor_ids)
    with _noninteractive_calibration_input():
        connect(calibrate=True)
    if recalibrate:
        calibrate = getattr(device, "calibrate", None)
        if not callable(calibrate):
            raise SystemExit(f"[teleop] LeRobot {label} has no calibrate()")
        calibrate()


def calibration_file_status(device: Any) -> tuple[Path | None, bool, bool]:
    """Return (path, exists_on_disk, loaded_into_device.calibration).

    LeRobot ``Robot.__init__`` already loads the JSON when ``calibration_fpath``
    exists. This is the explicit check we log before connect.
    """
    raw = getattr(device, "calibration_fpath", None)
    path = Path(raw) if raw else None
    exists = bool(path is not None and path.is_file())
    loaded = bool(getattr(device, "calibration", None))
    return path, exists, loaded


def log_calibration_file(device: Any, label: str) -> None:
    path, exists, loaded = calibration_file_status(device)
    print(
        f"[teleop] {label} calibration_file={path} exists={exists} loaded={loaded}",
        flush=True,
    )
    if path is not None and not exists:
        print(
            f"[teleop] {label} no calibration JSON at that path. "
            "LeRobot connect(calibrate=True) will run interactive calibrate() "
            "if the bus does not already match a file.",
            flush=True,
        )


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
    connect_lerobot_device(device, port=port, label="SO101 leader", recalibrate=recalibrate)
    return SO101LeaderSession(device)


def open_so101_follower(
    *,
    port: str,
    robot_id: str = "so101_follower",
    recalibrate: bool = False,
    mock: bool = False,
) -> SO101FollowerSession:
    if mock:
        print("[teleop] using MockSO101Follower (no serial)", flush=True)
        return SO101FollowerSession(MockSO101Follower())
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
    connect_lerobot_device(device, port=port, label="SO101 follower", recalibrate=recalibrate)
    return SO101FollowerSession(device)


def require_distinct_serial_ports(*ports: str | None) -> None:
    used = [port for port in ports if port]
    if len(used) != len(set(used)):
        raise SystemExit(f"[teleop] leader / follower / PingTi serial ports must be distinct, got {used}")
