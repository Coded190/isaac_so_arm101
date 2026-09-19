# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard SE(3) teleoperation of PingTi in Isaac Lab.

Isaac Sim / Kit viewport must have focus for keys to register. Lab 3.0
requires ``--viz kit`` (the Omniverse window). Newton GL / Viser / Rerun
do not feed Se3Keyboard.

    UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact teleop --scene palm --num_envs 1 --viz kit
"""

from __future__ import annotations

import argparse
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
AppLauncher.add_app_launcher_args(parser)
# Lab 3.0: omit --viz and AppLauncher goes headless. Se3Keyboard needs Kit.
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: E402, F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaac_so_arm101.teleop_constants import PINGTI_EE_BODY, SE3_ACTION_DIM  # noqa: E402
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


def _ee_pose_str(env) -> str:
    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies(PINGTI_EE_BODY)
    pos = robot.data.body_pos_w[0, body_ids[0]].detach()
    return f"ee_pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[teleop] task={args_cli.task}")
    print(f"[teleop] Gym observation space: {env.observation_space}")
    print(f"[teleop] Gym action space: {env.action_space}")
    print("[teleop] Click the Isaac Sim viewport so keyboard events go to the sim, not the terminal.")
    print(
        "[teleop] Keys: W/S x, A/D y, Q/E z, Z/X roll, T/G pitch, C/V yaw, "
        "K gripper, R reset env, L clear keyboard deltas, P print bake pose."
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

    def _reset():
        print("[teleop] reset")
        env.reset()
        teleop.reset()

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
    except Exception as exc:  # noqa: BLE001 — dump is diagnostic only
        print(f"[teleop] prim dump skipped: {exc}", flush=True)

    def _bake():
        _print_bake_pose(robot_prim)

    teleop.add_callback("P", _bake)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            cmd = teleop.advance()
            if cmd.numel() != SE3_ACTION_DIM:
                raise RuntimeError(
                    f"[teleop] expected {SE3_ACTION_DIM}-D device command, got shape {tuple(cmd.shape)}"
                )
            if cmd.device != env.unwrapped.device:
                cmd = cmd.to(env.unwrapped.device)
            actions = cmd.unsqueeze(0).expand(env.unwrapped.num_envs, -1).contiguous()
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
            try:
                root_sync.sync_visual_scale(robot_prim, robot)
            except Exception as exc:  # noqa: BLE001
                if step < 5:
                    print(f"[teleop] fabric scale sync failed step={step}: {exc}", flush=True)
            step += 1
            if args_cli.log_every > 0 and step % args_cli.log_every == 0:
                robot = env.unwrapped.scene["robot"]
                joints = robot.data.joint_pos[0].detach()
                print(
                    f"[teleop] step={step} action={ [round(x, 4) for x in cmd.tolist()] } "
                    f"{_ee_pose_str(env)} joints={ [round(x, 3) for x in joints.tolist()] }"
                )
                if sync_info is not None:
                    print(format_root_diag(sync_info, step=step), flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
