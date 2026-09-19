"""Headless smoke: keyboard SE3 and scripted SO101-leader joints move PingTi.

Does not open a Kit window. Measures PhysX joint / EE motion.

    UV_PROJECT_ENVIRONMENT=.venv-isaacsim-6.1 uv run --inexact teleop_sim_smoke --viz none
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PingTi keyboard + scripted-leader sim smoke.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=90, help="Physics steps per case.")
parser.add_argument(
    "--case",
    choices=["keyboard", "leader", "both"],
    default="both",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["none"], headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaac_so_arm101.devices.leader_map import (  # noqa: E402
    leader_state_hold,
    pingti_joint_pos_from_leader,
)
from isaac_so_arm101.tasks.teleop.teleop_env_cfg import apply_teleop_device  # noqa: E402
from isaac_so_arm101.teleop_constants import (  # noqa: E402
    JOINT_POS_ACTION_DIM,
    PINGTI_EE_BODY,
    PINGTI_JOINTS,
    SE3_ACTION_DIM,
    SO101_TO_PINGTI,
)

TASK = "Isaac-PING-TI-Teleop-v0"
KEYBOARD_DZ = 0.04
LEADER_HOLD = 40.0
EE_MIN_DELTA = 0.02
JOINT_MIN_DELTA = 0.08


def _named_joints(env) -> dict[str, float]:
    robot = env.unwrapped.scene["robot"]
    names = list(robot.joint_names)
    pos = robot.data.joint_pos[0].detach().cpu()
    return {name: float(pos[i]) for i, name in enumerate(names)}


def _ee_xyz(env) -> tuple[float, float, float]:
    robot = env.unwrapped.scene["robot"]
    body_ids, _ = robot.find_bodies(PINGTI_EE_BODY)
    pos = robot.data.body_pos_w[0, body_ids[0]].detach().cpu()
    return float(pos[0]), float(pos[1]), float(pos[2])


def _make_env(teleop_device: str):
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    apply_teleop_device(env_cfg, teleop_device)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    return env


def _step(env, action_row: list[float], steps: int) -> None:
    action = torch.tensor([action_row], dtype=torch.float32, device=env.unwrapped.device)
    action = action.expand(env.unwrapped.num_envs, -1).contiguous()
    for _ in range(steps):
        with torch.inference_mode():
            env.step(action)


def run_keyboard() -> None:
    print("[smoke] case=keyboard task=Isaac-PING-TI-Teleop-v0 se3_dim=7", flush=True)
    env = _make_env("keyboard")
    try:
        print(f"[smoke] keyboard action_space={env.action_space}", flush=True)
        before_ee = _ee_xyz(env)
        before_j = _named_joints(env)
        action = [0.0, 0.0, KEYBOARD_DZ, 0.0, 0.0, 0.0, 1.0]
        if len(action) != SE3_ACTION_DIM:
            raise RuntimeError("keyboard smoke action is not 7-D")
        _step(env, action, args_cli.num_steps)
        after_ee = _ee_xyz(env)
        after_j = _named_joints(env)
        dz = after_ee[2] - before_ee[2]
        print(
            f"[smoke] keyboard ee_before=({before_ee[0]:.4f},{before_ee[1]:.4f},{before_ee[2]:.4f}) "
            f"ee_after=({after_ee[0]:.4f},{after_ee[1]:.4f},{after_ee[2]:.4f}) dz={dz:.4f}",
            flush=True,
        )
        for name in PINGTI_JOINTS:
            print(
                f"[smoke] keyboard joint {name} {before_j.get(name, float('nan')):.4f} -> "
                f"{after_j.get(name, float('nan')):.4f}",
                flush=True,
            )
        if dz < EE_MIN_DELTA:
            raise SystemExit(
                f"[smoke] FAIL keyboard: expected EE +z > {EE_MIN_DELTA} m, got dz={dz:.4f}"
            )
        print("[smoke] PASS keyboard PingTi EE moved", flush=True)
    finally:
        env.close()


def run_leader() -> None:
    print("[smoke] case=leader scripted SO101 get_action dict -> PingTi 6-D", flush=True)
    env = _make_env("so101leader")
    try:
        print(f"[smoke] leader action_space={env.action_space}", flush=True)
        failures: list[str] = []
        for motor, joint in SO101_TO_PINGTI.items():
            if motor == "gripper":
                continue
            before = _named_joints(env)
            raw = leader_state_hold({motor: LEADER_HOLD})
            target = pingti_joint_pos_from_leader(raw)
            if len(target) != JOINT_POS_ACTION_DIM:
                raise RuntimeError("leader smoke action is not 6-D")
            print(f"[smoke] leader commanding {motor} -> {joint} target={target[PINGTI_JOINTS.index(joint)]:.4f}", flush=True)
            _step(env, list(target), args_cli.num_steps)
            after = _named_joints(env)
            delta = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in PINGTI_JOINTS}
            commanded = abs(after[joint])
            print(
                f"[smoke] leader motor={motor} -> {joint} "
                f"before={before[joint]:.4f} after={after[joint]:.4f} "
                f"phase_delta={delta[joint]:.4f} target={target[PINGTI_JOINTS.index(joint)]:.4f}",
                flush=True,
            )
            if commanded < JOINT_MIN_DELTA:
                failures.append(f"{motor}->{joint} did not move (after={after[joint]:.4f})")
        before_g = _named_joints(env)["gripper_moving"]
        grip_target = pingti_joint_pos_from_leader(leader_state_hold({"gripper": 80.0}))
        print(f"[smoke] leader commanding gripper -> gripper_moving target={grip_target[-1]:.4f}", flush=True)
        _step(env, list(grip_target), args_cli.num_steps)
        after_g = _named_joints(env)["gripper_moving"]
        print(
            f"[smoke] leader motor=gripper -> gripper_moving "
            f"before={before_g:.4f} after={after_g:.4f} target={grip_target[-1]:.4f}",
            flush=True,
        )
        if after_g - before_g < JOINT_MIN_DELTA and after_g < JOINT_MIN_DELTA:
            failures.append(f"gripper_moving did not open (after={after_g:.4f})")
        if failures:
            raise SystemExit("[smoke] FAIL leader: " + "; ".join(failures))
        print("[smoke] PASS leader scripted SO101 moved the mapped PingTi joints", flush=True)
    finally:
        env.close()


def main() -> int:
    print(f"[smoke] task={TASK} steps={args_cli.num_steps} case={args_cli.case}", flush=True)
    if args_cli.case in {"keyboard", "both"}:
        run_keyboard()
    if args_cli.case in {"leader", "both"}:
        run_leader()
    print("[smoke] PASS all requested cases", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
