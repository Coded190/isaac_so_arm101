"""SO-ARM101 leader → real PingTi follower (no Isaac Sim).

Bypasses the simulator. Leader ±100 / gripper 0–100 expands to 8 Feetech
``{motor}.pos`` goals (SO101 names; dual-drive secondaries get ``-val``).
Optional real SO101 follower still gets the original leader dict.

    UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact teleop_hw --mock --steps 12
    UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact teleop_hw \
      --port /dev/ttyACM0 --pingti_port /dev/ttyACM1
    # optional SO101 follower on a third adapter:
    #   --follower_port /dev/ttyACM2
"""

from __future__ import annotations

import argparse
import sys

from isaac_so_arm101.devices.leader_map import leader_state_hold
from isaac_so_arm101.devices.pipeline import run_leader_hw_loop
from isaac_so_arm101.devices.pingti import open_pingti_follower
from isaac_so_arm101.devices.so101 import (
    ScriptedSO101Leader,
    open_so101_follower,
    open_so101_leader,
    require_distinct_serial_ports,
)
from isaac_so_arm101.teleop_constants import (
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_JOINT_TO_FOLLOWER_MOTORS,
    PINGTI_PHYSICAL_MOTOR_COUNT,
    SO101_LEADER_MOTORS,
    SO101_TO_PINGTI,
)


def _scripted_isolation_frames() -> list[dict[str, float]]:
    frames = [leader_state_hold()]
    for motor in SO101_LEADER_MOTORS:
        cmd = 80.0 if motor == "gripper" else 40.0
        frames.append(leader_state_hold({motor: cmd}))
        frames.append(leader_state_hold())
    return frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO101 leader → PingTi follower (no sim).")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="SO101 leader serial port.")
    parser.add_argument("--pingti_port", type=str, default=None, help="Real PingTi follower serial port.")
    parser.add_argument(
        "--follower_port",
        type=str,
        default=None,
        help="Optional real SO101 follower (leader motor space, not PingTi radians).",
    )
    parser.add_argument("--leader_id", type=str, default="so101_leader")
    parser.add_argument("--follower_id", type=str, default="so101_follower")
    parser.add_argument("--pingti_id", type=str, default="pingti_follower")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Scripted leader + recording followers (no USB).")
    parser.add_argument("--steps", type=int, default=0, help="Finite steps. 0 = run until Ctrl+C (hardware).")
    parser.add_argument("--hz", type=float, default=30.0, help="Loop rate. 0 disables sleep.")
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> int:
    mock = bool(args.mock)
    steps = int(args.steps)
    if mock and steps < 1:
        steps = len(_scripted_isolation_frames())
    if not mock and steps < 1:
        steps = 10**9
    if not mock and not args.pingti_port:
        print("[teleop_hw] --pingti_port is required unless --mock", file=sys.stderr)
        return 2

    leader = None
    so101 = None
    pingti = None
    try:
        if mock:
            print(
                f"[teleop_hw] mock SO101 leader → PingTi {PINGTI_PHYSICAL_MOTOR_COUNT}-motor follower "
                f"(no serial) steps={steps}",
                flush=True,
            )
            leader = ScriptedSO101Leader(_scripted_isolation_frames())
            pingti = open_pingti_follower(port="mock", mock=True)
            so101 = open_so101_follower(port="mock", mock=True)
        else:
            require_distinct_serial_ports(args.port, args.follower_port, args.pingti_port)
            leader = open_so101_leader(
                port=args.port,
                robot_id=args.leader_id,
                recalibrate=args.recalibrate,
                mock=False,
            )
            pingti = open_pingti_follower(
                port=args.pingti_port,
                robot_id=args.pingti_id,
                recalibrate=args.recalibrate,
                mock=False,
            )
            if args.follower_port:
                so101 = open_so101_follower(
                    port=args.follower_port,
                    robot_id=args.follower_id,
                    recalibrate=args.recalibrate,
                    mock=False,
                )

        result = run_leader_hw_loop(leader, so101=so101, pingti=pingti, steps=steps, hz=args.hz)
        if args.log_every > 0 and result.last is not None:
            last = result.last
            print(
                f"[teleop_hw] steps={result.steps} joints={[round(x, 4) for x in last.joints6]} "
                f"so101_keys={sorted(last.so101_action)} pingti_keys={sorted(last.pingti_action)}",
                flush=True,
            )
        if mock:
            _assert_mock_isolation(result)
            print("[teleop_hw] PASS mock SO101 leader → PingTi 8-motor + SO101 follower", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[teleop_hw] interrupted", flush=True)
        return 0
    finally:
        for session in (pingti, so101, leader):
            if session is None:
                continue
            close = getattr(session, "close", None) or getattr(session, "disconnect", None)
            if callable(close):
                close()


def _assert_mock_isolation(result) -> None:
    if result.steps < 3:
        raise SystemExit("[teleop_hw] FAIL: expected isolation frames")
    if not result.pingti_sends:
        raise SystemExit("[teleop_hw] FAIL: PingTi follower recorded no sends")
    if not result.so101_sends:
        raise SystemExit("[teleop_hw] FAIL: SO101 follower recorded no sends")
    pingti_zero = result.pingti_sends[0]
    so101_zero = result.so101_sends[0]
    if sorted(k.removesuffix(".pos") for k in pingti_zero) != sorted(PINGTI_FOLLOWER_MOTORS):
        raise SystemExit(f"[teleop_hw] FAIL: PingTi keys {sorted(pingti_zero)}")
    if sorted(k.removesuffix(".pos") for k in so101_zero) != sorted(SO101_LEADER_MOTORS):
        raise SystemExit(f"[teleop_hw] FAIL: SO101 keys {sorted(so101_zero)}")

    # Frame 0 hold, then pairs (cmd, hold) per leader motor.
    for i, motor in enumerate(SO101_LEADER_MOTORS):
        frame = 1 + 2 * i
        if frame >= len(result.pingti_sends):
            raise SystemExit(f"[teleop_hw] FAIL: missing isolation frame for {motor}")
        pingti = result.pingti_sends[frame]
        so101 = result.so101_sends[frame]
        joint = SO101_TO_PINGTI[motor]
        duals = PINGTI_JOINT_TO_FOLLOWER_MOTORS[joint]
        cmd = pingti[f"{duals[0]}.pos"]
        if abs(cmd) < 1.0 and motor != "gripper":
            raise SystemExit(f"[teleop_hw] FAIL: {motor} did not drive {joint} (pos={cmd})")
        if motor == "gripper" and abs(cmd) < 1.0:
            raise SystemExit(f"[teleop_hw] FAIL: gripper command was {cmd}")
        for dual in duals[1:]:
            if pingti[f"{dual}.pos"] != -cmd:
                raise SystemExit(
                    f"[teleop_hw] FAIL: dual {duals} expected {cmd}/{-cmd}, "
                    f"got {cmd}/{pingti[f'{dual}.pos']}"
                )
        for other_joint, other_motors in PINGTI_JOINT_TO_FOLLOWER_MOTORS.items():
            if other_joint == joint:
                continue
            for other in other_motors:
                val = pingti[f"{other}.pos"]
                if abs(val) > 1e-6:
                    raise SystemExit(f"[teleop_hw] FAIL: {motor} leaked to {other}={val}")
        for so_motor in SO101_LEADER_MOTORS:
            val = so101[f"{so_motor}.pos"]
            if so_motor == motor:
                if abs(val) < 1.0:
                    raise SystemExit(f"[teleop_hw] FAIL: SO101 follower {motor} stay {val}")
            elif abs(val) > 1e-6:
                raise SystemExit(f"[teleop_hw] FAIL: SO101 follower {motor} leaked to {so_motor}={val}")


def main(argv: list[str] | None = None) -> int:
    return run_from_args(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
