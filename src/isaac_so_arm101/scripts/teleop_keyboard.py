# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard SE(3) teleoperation of PingTi in Isaac Lab.

Isaac Sim / Kit viewport must have focus for keys to register. Lab 3.0
requires ``--viz kit`` (the Omniverse window). Newton GL / Viser / Rerun
do not feed Se3Keyboard.

    UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact teleop --scene palm --num_envs 1 --viz kit

SO-ARM101 leader (6-D joint position, optional real followers):

    UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact teleop --scene palm --viz kit \
      --teleop_device so101leader --port /dev/ttyACM0
    # optional real SO101 follower (leader motor space, not PingTi radians):
    #   --follower_port /dev/ttyACM1
    # optional real PingTi follower (sim 6 joints expanded to 8 motors):
    #   --pingti_port /dev/ttyACM2
    # Kit smoke without serial:
    #   --teleop_device so101leader --mock_leader --mock_pingti --mock_follower
"""

from __future__ import annotations

import argparse
import atexit
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard SE(3) teleop for PingTi in Isaac Lab.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (use 1 for teleop).")
parser.add_argument("--task", type=str, default=None, help="Gym task name. Overrides --scene.")
parser.add_argument(
    "--scene",
    choices=["procedural", "palm"],
    default="palm",
    help="palm garden (default, needs fetch_assets) or procedural ground plane.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Scale Se3Keyboard pos/rot sensitivity.")
parser.add_argument("--log_every", type=int, default=60, help="Print [teleop] telemetry every N steps.")
parser.add_argument(
    "--dir_log",
    type=str,
    default="logs/teleop_joint_dir.log",
    help="Append [teleop_dir] leader vs sim joint deltas (also printed). grep teleop_dir.",
)
parser.add_argument(
    "--teleop_device",
    choices=["keyboard", "so101leader"],
    default="keyboard",
    help="keyboard=7-D SE3; so101leader=6-D PingTi joint pos. Lab --device is still cuda/cpu.",
)
parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="SO101 leader serial port.")
parser.add_argument(
    "--follower_port",
    type=str,
    default=None,
    help="Optional real SO101 follower serial port. Sends leader motor space, not PingTi radians.",
)
parser.add_argument(
    "--pingti_port",
    type=str,
    default=None,
    help="Optional real PingTi follower. Holds present until N (or J) in the Kit viewport, then slowly follows sim.",
)
parser.add_argument(
    "--pingti_follow",
    action="store_true",
    default=False,
    help="Start slewing the real PingTi toward sim after connect (still capped). Default is hold present until N.",
)
parser.add_argument("--leader_id", type=str, default="so101_leader", help="LeRobot calibration id for the leader.")
parser.add_argument("--follower_id", type=str, default="so101_follower", help="LeRobot calibration id for the follower.")
parser.add_argument("--pingti_id", type=str, default="pingti_follower", help="LeRobot calibration id for the PingTi follower.")
parser.add_argument("--recalibrate", action="store_true", help="Run LeRobot calibrate() on connect.")
parser.add_argument(
    "--mock_leader",
    action="store_true",
    help="so101leader without serial: hold leader zeros (Kit / action-space smoke).",
)
parser.add_argument(
    "--mock_follower",
    action="store_true",
    help="Record SO101 follower send_action without serial (requires so101leader).",
)
parser.add_argument(
    "--mock_pingti",
    action="store_true",
    help="Record PingTi 8-motor send_action without serial (keyboard or so101leader).",
)
parser.add_argument(
    "--disable_pingti_torque",
    action="store_true",
    help="Write Torque_Enable=0 on PingTi ids 1–8 and exit (no Kit). Stop any teleop that holds the port first.",
)
AppLauncher.add_app_launcher_args(parser)
# Lab 3.0: omit --viz and AppLauncher goes headless. Se3Keyboard needs Kit.
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

if args_cli.disable_pingti_torque:
    from isaac_so_arm101.devices.pingti import disable_pingti_torque_raw

    port = args_cli.pingti_port or "/dev/ttyACM1"
    torque = disable_pingti_torque_raw(port)
    bad = [mid for mid, val in torque.items() if int(val) != 0]
    if bad:
        print(f"[teleop] PingTi torque still on ids={bad} port={port}", flush=True)
        sys.exit(1)
    print(f"[teleop] PingTi torque off port={port} ids={sorted(torque)}", flush=True)
    sys.exit(0)

_viz_raw = getattr(args_cli, "visualizer", None)
if isinstance(_viz_raw, str):
    _viz_list = [item.strip().lower() for item in _viz_raw.split(",") if item.strip()]
elif _viz_raw is None:
    _viz_list = ["kit"]
    args_cli.visualizer = ["kit"]
else:
    _viz_list = [str(item).strip().lower() for item in _viz_raw]

_headless = bool(getattr(args_cli, "headless", False))
if _headless or _viz_list == ["none"] or (not _viz_list):
    print(
        "[teleop] keyboard teleop requires the Isaac Sim Kit viewport. "
        "Do not pass --headless or --viz none. Use --viz kit.",
        file=sys.stderr,
    )
    sys.exit(2)
if "kit" not in _viz_list:
    print(
        f"[teleop] Se3Keyboard listens on the Kit viewport, not '{','.join(_viz_list)}'. "
        "Pass --viz kit (or kit plus another visualizer).",
        file=sys.stderr,
    )
    sys.exit(2)

if args_cli.task is None:
    args_cli.task = (
        "Isaac-PING-TI-Teleop-Palm-v0" if args_cli.scene == "palm" else "Isaac-PING-TI-Teleop-v0"
    )
if args_cli.mock_leader and args_cli.teleop_device != "so101leader":
    print("[teleop] --mock_leader requires --teleop_device so101leader", file=sys.stderr)
    sys.exit(2)
if args_cli.follower_port and args_cli.teleop_device != "so101leader":
    print("[teleop] --follower_port requires --teleop_device so101leader", file=sys.stderr)
    sys.exit(2)
if args_cli.mock_follower and args_cli.teleop_device != "so101leader":
    print("[teleop] --mock_follower requires --teleop_device so101leader", file=sys.stderr)
    sys.exit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: E402, F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaac_so_arm101.devices.leader_map import (  # noqa: E402
    clipped_joint_report,
    follow_error_report,
    pingti_named_joints_from_leader,
)
from isaac_so_arm101.devices.pingti import GOAL_SLEW_MAX, PINGTI_SEND_EVERY_STEPS  # noqa: E402

PINGTI_FOLLOW_ENABLE_KEYS = ("N", "J")
PINGTI_FOLLOW_HOLD_KEYS = ("M",)
PINGTI_FOLLOW_KEYS = PINGTI_FOLLOW_ENABLE_KEYS + PINGTI_FOLLOW_HOLD_KEYS + ("H",)


def poll_kit_key_rising(session: dict, keys: tuple[str, ...] = PINGTI_FOLLOW_KEYS) -> tuple[str, ...]:
    """Rising-edge Kit keys. H is Isaac Sim Hide and often never reaches Se3Keyboard."""
    try:
        import carb
        import omni

        iface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        keyboard_input = carb.input.KeyboardInput
    except Exception:
        return ()
    down: list[str] = []
    for name in keys:
        key = getattr(keyboard_input, name, None)
        if key is None:
            continue
        try:
            if iface.get_keyboard_value(keyboard, key):
                down.append(name)
        except Exception:
            continue
    prev = session.get("follow_keys_down")
    if not isinstance(prev, set):
        prev = set()
    now = set(down)
    session["follow_keys_down"] = now
    return tuple(sorted(now - prev))
from isaac_so_arm101.devices.dir_log import JointDirLogger, emit_hw_lines  # noqa: E402
from isaac_so_arm101.teleop_constants import (  # noqa: E402
    JOINT_POS_ACTION_DIM,
    PINGTI_EE_BODY,
    PINGTI_JOINTS,
    SE3_ACTION_DIM,
    SO101_PINGTI_SIGN,
)
from isaac_so_arm101.tasks.teleop.teleop_env_cfg import apply_teleop_device  # noqa: E402
from isaac_so_arm101.teleop_root import (  # noqa: E402
    KitRootSync,
    format_root_diag,
    usd_authored_world_pose_xyzw,
    usd_xform_op_names,
)

POS_SENS = 0.05
ROT_SENS = 0.05
LARGE_POS_DELTA = 0.2


def _print_bake_pose(prim) -> None:
    pose = usd_authored_world_pose_xyzw(prim)
    if pose is None:
        print("[teleop] no Robot prim to bake", flush=True)
        return
    x, y, z, qx, qy, qz, qw = pose
    print(
        f"[teleop] bake PINGTI_PALM_POS=({x:.5f}, {y:.5f}, {z:.5f}) "
        f"PINGTI_PALM_ROT=({qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f})  # Lab3 xyzw",
        flush=True,
    )


def _frame_kit_camera(env, env_cfg) -> None:
    """Point the Kit viewport at the teleop viewer eye/lookat (not origin)."""
    eye = tuple(float(v) for v in env_cfg.viewer.eye)
    lookat = tuple(float(v) for v in env_cfg.viewer.lookat)
    sim = getattr(env.unwrapped, "sim", None)
    setter = getattr(sim, "set_camera_view", None) if sim is not None else None
    if callable(setter):
        setter(eye, lookat)
        print(f"[teleop] kit camera eye={eye} lookat={lookat}", flush=True)
        return
    print("[teleop] env.sim.set_camera_view unavailable; Kit camera may stay at origin", flush=True)


def _log_dome_light(stage, light_prim) -> None:
    """Log garden HDRI bind. Keep authored intensity (1000); do not boost."""
    if light_prim is None or not light_prim.IsValid():
        print("[teleop] garden DomeLight missing; relying on /World/light fallback", flush=True)
        return
    try:
        from pxr import UsdLux
    except ImportError:
        return
    if not light_prim.IsA(UsdLux.DomeLight):
        print(f"[teleop] {light_prim.GetPath()} is not a DomeLight", flush=True)
        return
    dome = UsdLux.DomeLight(light_prim)
    tex = dome.GetTextureFileAttr().Get()
    tex_path = getattr(tex, "resolvedPath", None) or getattr(tex, "path", None) or tex
    intensity = dome.GetIntensityAttr().Get()
    if intensity is not None and float(intensity) > 1000.0:
        dome.GetIntensityAttr().Set(1000.0)
        print(f"[teleop] dome_light_tex={tex_path} intensity={intensity} -> 1000", flush=True)
        return
    print(f"[teleop] dome_light_tex={tex_path} intensity={intensity}", flush=True)


def _ee_pose_str(env) -> str:
    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies(PINGTI_EE_BODY)
    pos = robot.data.body_pos_w[0, body_ids[0]].detach()
    return f"ee_pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"


def _snap_sim_to_leader(env, leader) -> None:
    """Write the live leader pose into sim, clipped to PingTi URDF limits."""
    if leader is None:
        return
    try:
        raw = leader.get_action()
    except Exception as exc:  # noqa: BLE001
        print(f"[teleop] snap_to_leader_failed err={exc!r}", flush=True)
        return
    if not isinstance(raw, dict):
        print("[teleop] snap_to_leader_skipped reason=no_leader_action", flush=True)
        return
    try:
        named = pingti_named_joints_from_leader(raw)
    except Exception as exc:  # noqa: BLE001
        print(f"[teleop] snap_to_leader_failed err={exc!r}", flush=True)
        return
    robot = env.unwrapped.scene["robot"]
    q = robot.data.joint_pos.clone()
    names = list(robot.joint_names)
    missing = [name for name in PINGTI_JOINTS if name not in names]
    if missing:
        print(f"[teleop] snap_to_leader_failed missing_joints={missing}", flush=True)
        return
    for name, value in named.items():
        q[:, names.index(name)] = float(value)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    setter = getattr(robot, "set_joint_position_target", None)
    if callable(setter):
        setter(q)
    clipped = clipped_joint_report(raw)
    print(
        f"[teleop] snap_to_leader q={[round(named[n], 4) for n in PINGTI_JOINTS]} "
        f"clipped={clipped}",
        flush=True,
    )


def main():
    use_leader = args_cli.teleop_device == "so101leader"
    from isaac_so_arm101.devices.so101 import require_distinct_serial_ports

    require_distinct_serial_ports(
        args_cli.port if use_leader and not args_cli.mock_leader else None,
        args_cli.follower_port if not args_cli.mock_follower else None,
        args_cli.pingti_port if not args_cli.mock_pingti else None,
    )
    if args_cli.pingti_port and not args_cli.mock_pingti:
        from isaac_so_arm101.devices.so101 import read_feetech_bus_voltages

        rows = read_feetech_bus_voltages(args_cli.pingti_port, motor_ids=range(1, 9))
        found = [mid for mid, _volt, _err in rows]
        print(
            f"[teleop] PingTi preflight port={args_cli.pingti_port} "
            f"found_ids={found} "
            + " ".join(f"id{mid}={volt:.1f}V" for mid, volt, _err in rows),
            flush=True,
        )
        missing = [mid for mid in range(1, 9) if mid not in found]
        if missing:
            raise SystemExit(
                f"[teleop] PingTi missing motor ids {missing} on {args_cli.pingti_port} "
                f"(need 1-8: pan, dual lift, dual elbow, wrist_flex, wrist_roll, gripper). "
                "Check 12 V supply to the lower arm / daisy-chain. Refusing partial bus."
            )
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    apply_teleop_device(env_cfg, args_cli.teleop_device)
    env = gym.make(args_cli.task, cfg=env_cfg)

    expected_dim = JOINT_POS_ACTION_DIM if use_leader else SE3_ACTION_DIM
    print(f"[teleop] task={args_cli.task} teleop_device={args_cli.teleop_device} action_dim={expected_dim}")
    print(f"[teleop] Gym observation space: {env.observation_space}")
    print(f"[teleop] Gym action space: {env.action_space}")
    print("[teleop] Click the Isaac Sim viewport so keyboard events go to the sim, not the terminal.")
    if use_leader:
        print(
            "[teleop] SO101 leader drives PingTi joints. Keys: R reset+snap sim to leader, "
            "N/J enable real PingTi follow, M hold (H is Isaac Sim Hide), P print bake pose. "
            f"leader_port={args_cli.port} follower_port={args_cli.follower_port} "
            f"pingti_port={args_cli.pingti_port} mock_leader={args_cli.mock_leader} "
            f"mock_follower={args_cli.mock_follower} mock_pingti={args_cli.mock_pingti} "
            f"signs={ {m: SO101_PINGTI_SIGN[m] for m in SO101_PINGTI_SIGN} } "
            f"dir_log={args_cli.dir_log}"
        )
    else:
        print(
            "[teleop] Keys: W/S x, A/D y, Q/E z, Z/X roll, T/G pitch, C/V yaw, "
            "K gripper, R reset env, L clear keyboard deltas, P print bake pose, "
            "N/J real PingTi follow/hold. "
            f"pingti_port={args_cli.pingti_port} mock_pingti={args_cli.mock_pingti}"
        )
    print(
        "[teleop] Select /World/envs/env_0/Robot and edit Translate / Orient / Scale. "
        "Panel pose is held every physics step so the free root stays at crown height "
        "(gravity cannot pull it down). Grep reason=usd_attr / reason=hold.",
        flush=True,
    )

    sim_device = str(env.unwrapped.device)
    teleop = Se3Keyboard(
        Se3KeyboardCfg(
            pos_sensitivity=POS_SENS * args_cli.sensitivity,
            rot_sensitivity=ROT_SENS * args_cli.sensitivity,
            sim_device=sim_device,
        )
    )
    print(teleop)

    session = {
        "leader": None,
        "pingti_follow": False,
        "follow_keys_down": set(),
        "follow_toggle_at": 0.0,
        "pingti_send_now": False,
    }

    def _reset():
        print("[teleop] reset")
        env.reset()
        teleop.reset()
        _snap_sim_to_leader(env, session["leader"])
        if session["pingti_follow"]:
            print(
                "[teleop] snap_to_leader sim only; real PingTi will slew toward that pose "
                f"(follow=ON, not a jump)",
                flush=True,
            )
        else:
            print(
                "[teleop] snap_to_leader sim only; real PingTi still holding present "
                "(press N to follow slowly)",
                flush=True,
            )

    teleop.add_callback("R", _reset)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    robot_prim = None
    root_sync = KitRootSync()
    try:
        import omni.usd
        from isaaclab.sim.utils import find_matching_prims
        from isaac_so_arm101.scene_prims import (
            find_dome_light_prim,
            find_palm_root_prim,
        )

        stage = omni.usd.get_context().get_stage()
        matches = find_matching_prims(robot.cfg.prim_path)
        robot_prim = matches[0] if matches else stage.GetPrimAtPath("/World/envs/env_0/Robot")
        palm, palm_path = find_palm_root_prim(stage, 0)
        light, light_path = find_dome_light_prim(stage, 0)
        crown = palm.GetChild("crown") if palm is not None else None
        print(
            f"[teleop] palm_root={palm_path} has_crown={bool(crown and crown.IsValid())} "
            f"dome_light={light_path}",
            flush=True,
        )
        root = robot.data.root_pos_w[0].detach()
        print(
            f"[teleop] robot_root=({root[0]:.3f}, {root[1]:.3f}, {root[2]:.3f}) "
            f"prim={robot_prim.GetPath() if robot_prim else None} "
            f"xformOps={usd_xform_op_names(robot_prim)}",
            flush=True,
        )
        try:
            print(format_root_diag(root_sync.apply(robot, robot_prim)), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[teleop] USD root sync failed at startup: {exc}", flush=True)
        _frame_kit_camera(env, env_cfg)
        _log_dome_light(stage, light)
    except Exception as exc:  # noqa: BLE001 — dump is diagnostic only
        print(f"[teleop] prim dump skipped: {exc}", flush=True)

    def _bake():
        _print_bake_pose(robot_prim)

    teleop.add_callback("P", _bake)

    leader = None
    follower = None
    pingti = None
    dir_log = None
    step = 0
    try:
        if use_leader:
            from isaac_so_arm101.devices.leader_map import (
                leader_action_from_state,
                pingti_joint_pos_from_leader,
            )
            from isaac_so_arm101.devices.so101 import open_so101_follower, open_so101_leader

            leader = open_so101_leader(
                port=args_cli.port,
                robot_id=args_cli.leader_id,
                recalibrate=args_cli.recalibrate,
                mock=args_cli.mock_leader,
            )
            session["leader"] = leader
            _snap_sim_to_leader(env, leader)
            dir_log = JointDirLogger(args_cli.dir_log)
            if args_cli.follower_port or args_cli.mock_follower:
                follower = open_so101_follower(
                    port=args_cli.follower_port or "mock",
                    robot_id=args_cli.follower_id,
                    recalibrate=args_cli.recalibrate,
                    mock=args_cli.mock_follower or not args_cli.follower_port,
                )
        if args_cli.pingti_port or args_cli.mock_pingti:
            from isaac_so_arm101.devices.pipeline import send_sim_joints_to_pingti
            from isaac_so_arm101.devices.pingti import open_pingti_follower

            pingti = open_pingti_follower(
                port=args_cli.pingti_port or "mock",
                robot_id=args_cli.pingti_id,
                recalibrate=args_cli.recalibrate,
                mock=args_cli.mock_pingti or not args_cli.pingti_port,
            )
            atexit.register(pingti.close)
            print(
                f"[teleop] PingTi holding present. Press N (or J) in the Isaac Sim viewport to slowly "
                f"follow sim. H is Kit Hide and will not arm follow. Press N again to hold. "
                f"slew_max={GOAL_SLEW_MAX} send_every={PINGTI_SEND_EVERY_STEPS}. "
                "Do not expect the real arm to move until N. "
                "Ctrl+C or closing Kit disables PingTi torque so the arm goes limp. "
                "Grep pingti_follow=ON / sent= / PingTi shutdown in the terminal.",
                flush=True,
            )
            if args_cli.pingti_follow:
                session["pingti_follow"] = True
                session["pingti_send_now"] = True
                print(
                    f"[teleop] pingti_follow=ON at start (--pingti_follow) slew_max={GOAL_SLEW_MAX}",
                    flush=True,
                )

        def _set_pingti_follow(on: bool):
            if pingti is None:
                print("[teleop] pingti_follow ignored (no --pingti_port / --mock_pingti)", flush=True)
                return
            now = time.monotonic()
            if now - float(session.get("follow_toggle_at", 0.0)) < 0.25 and session["pingti_follow"] == on:
                return
            session["follow_toggle_at"] = now
            session["pingti_follow"] = bool(on)
            if session["pingti_follow"]:
                session["pingti_send_now"] = True
            state = "ON" if session["pingti_follow"] else "HOLD"
            print(
                f"[teleop] pingti_follow={state} slew_max={GOAL_SLEW_MAX} "
                "(N/J enable follow, M hold; slew from Present so Goal cannot walk away)",
                flush=True,
            )

        for _key in PINGTI_FOLLOW_ENABLE_KEYS:
            teleop.add_callback(_key, lambda: _set_pingti_follow(True))
        for _key in PINGTI_FOLLOW_HOLD_KEYS:
            teleop.add_callback(_key, lambda: _set_pingti_follow(False))

        while simulation_app.is_running():
            with torch.inference_mode():
                rising = poll_kit_key_rising(session)
                if any(k in PINGTI_FOLLOW_HOLD_KEYS for k in rising):
                    _set_pingti_follow(False)
                elif any(k in PINGTI_FOLLOW_ENABLE_KEYS for k in rising):
                    _set_pingti_follow(True)
                raw = None
                if use_leader:
                    teleop.advance()
                    raw = leader.get_action()
                    joints6 = pingti_joint_pos_from_leader(raw)
                    cmd = torch.tensor(joints6, dtype=torch.float32, device=env.unwrapped.device)
                    if follower is not None:
                        follower.send_action(leader_action_from_state(raw))
                else:
                    cmd = teleop.advance()
                if cmd.numel() != expected_dim:
                    raise RuntimeError(
                        f"[teleop] expected {expected_dim}-D device command, got shape {tuple(cmd.shape)}"
                    )
                if cmd.device != env.unwrapped.device:
                    cmd = cmd.to(env.unwrapped.device)
                actions = cmd.unsqueeze(0).expand(env.unwrapped.num_envs, -1).contiguous()
                if not use_leader:
                    pos_norm = float(torch.linalg.norm(cmd[:3]))
                    if pos_norm > LARGE_POS_DELTA:
                        print(f"[teleop] large pos delta={pos_norm:.3f} cmd={cmd.tolist()}")
                sync_info = None
                try:
                    sync_info = root_sync.apply(robot, robot_prim)
                    if sync_info.get("applied") and sync_info.get("reason") == "usd_attr":
                        print(format_root_diag(sync_info, step=step), flush=True)
                except Exception as exc:  # noqa: BLE001
                    if step < 5 or (args_cli.log_every > 0 and step % args_cli.log_every == 0):
                        print(f"[teleop] USD root sync failed step={step}: {exc}", flush=True)
                try:
                    env.step(actions)
                except Exception as exc:  # noqa: BLE001
                    name = type(exc).__name__
                    if name in {"LinAlgError", "_LinAlgError"} or "singular" in str(exc).lower():
                        print(
                            f"[teleop] step={step} IK singular ({exc}); skip step. Press R to reset.",
                            flush=True,
                        )
                        continue
                    raise
                pingti_action = None
                send_now = bool(session.pop("pingti_send_now", False))
                if pingti is not None and session["pingti_follow"] and (
                    send_now or step % PINGTI_SEND_EVERY_STEPS == 0
                ):
                    try:
                        pingti_action = send_sim_joints_to_pingti(robot, pingti)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[teleop_hw] send_failed step={step} err={exc!r}", flush=True)
                try:
                    root_sync.sync_visual_scale(robot_prim, robot)
                except Exception as exc:  # noqa: BLE001
                    if step < 5:
                        print(f"[teleop] fabric scale sync failed step={step}: {exc}", flush=True)
                step += 1
                if dir_log is not None and raw is not None:
                    robot = env.unwrapped.scene["robot"]
                    names = list(robot.joint_names)
                    pos = robot.data.joint_pos[0].detach().cpu()
                    meas = {name: float(pos[i]) for i, name in enumerate(names)}
                    force = args_cli.log_every > 0 and step % args_cli.log_every == 0
                    dir_log.step(
                        step=step,
                        raw_leader=raw,
                        sim_cmd=tuple(float(x) for x in cmd.detach().cpu().tolist()),
                        sim_meas=meas,
                        pingti_action=pingti_action,
                        force=force,
                    )
                want_hw = pingti is not None and (
                    args_cli.log_every > 0 and step % args_cli.log_every == 0
                )
                if want_hw:
                    snap = None
                    try:
                        snap = pingti.hw_snapshot()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[teleop_hw] snapshot_failed step={step} err={exc!r}", flush=True)
                    else:
                        if snap is not None and any(int(v) != 1 for v in snap.get("torque", {}).values()):
                            off = [n for n, v in snap["torque"].items() if int(v) != 1]
                            print(
                                f"[teleop_hw] step={step} torque_off motors={off} "
                                "holding_present_then_enable",
                                flush=True,
                            )
                            pingti.ensure_torque_on()
                            snap = pingti.hw_snapshot()
                    emit_hw_lines(
                        step=step,
                        pingti_action=pingti_action,
                        snap=snap,
                        logger=dir_log,
                    )
                    if snap is not None:
                        try:
                            robot = env.unwrapped.scene["robot"]
                            named = {n: float(robot.data.joint_pos[0][i].detach()) for i, n in enumerate(robot.joint_names)}
                            from isaac_so_arm101.devices.leader_map import joints6_from_named

                            for line in follow_error_report(
                                joints6_from_named(named),
                                snap.get("present") or {},
                                snap.get("goal") or {},
                            ):
                                if dir_log is not None:
                                    dir_log.emit(line)
                                else:
                                    print(line, flush=True)
                        except Exception as exc:  # noqa: BLE001
                            print(f"[teleop_hw] compare_failed step={step} err={exc!r}", flush=True)
                if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                    robot = env.unwrapped.scene["robot"]
                    joints = robot.data.joint_pos[0].detach()
                    print(
                        f"[teleop] step={step} pingti_follow={'ON' if session['pingti_follow'] else 'HOLD'} "
                        f"pingti_sent={'yes' if pingti_action else 'no'} "
                        f"action={ [round(x, 4) for x in cmd.tolist()] } "
                        f"{_ee_pose_str(env)} joints={ [round(x, 3) for x in joints.tolist()] }"
                    )
                    if sync_info is not None:
                        print(format_root_diag(sync_info, step=step), flush=True)
    except KeyboardInterrupt:
        print("[teleop] interrupted (Ctrl+C); disabling PingTi torque", flush=True)
    finally:
        print("[teleop] stopping; PingTi torque off", flush=True)
        if dir_log is not None:
            dir_log.close()
        if pingti is not None:
            pingti.close()
        if follower is not None:
            follower.close()
        if leader is not None:
            leader.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
