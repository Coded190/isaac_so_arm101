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
    pingti_follower_action_from_joints,
    pingti_joint_pos_from_leader,
)
from isaac_so_arm101.devices.pingti import MockPingTiFollower  # noqa: E402
from isaac_so_arm101.tasks.teleop.teleop_env_cfg import apply_teleop_device  # noqa: E402
from isaac_so_arm101.teleop_constants import (  # noqa: E402
    JOINT_POS_ACTION_DIM,
    PINGTI_DUAL_JOINTS,
    PINGTI_EE_BODY,
    PINGTI_FOLLOWER_MOTORS,
    PINGTI_JOINT_TO_FOLLOWER_MOTORS,
    PINGTI_JOINTS,
    SE3_ACTION_DIM,
    SO101_TO_PINGTI,
)

TASK = "Isaac-PING-TI-Teleop-v0"
KEYBOARD_DZ = 0.04
KEYBOARD_DY = 0.04
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


def _step(env, action_row: list[float], steps: int, pingti: MockPingTiFollower | None = None) -> None:
    action = torch.tensor([action_row], dtype=torch.float32, device=env.unwrapped.device)
    action = action.expand(env.unwrapped.num_envs, -1).contiguous()
    for _ in range(steps):
        with torch.inference_mode():
            env.step(action)
            if pingti is not None:
                pingti.send_action(pingti_follower_action_from_joints(tuple(_named_joints(env)[n] for n in PINGTI_JOINTS)))


def _assert_pingti_duals(action: dict[str, float], label: str) -> None:
    missing = [name for name in PINGTI_FOLLOWER_MOTORS if f"{name}.pos" not in action]
    if missing:
        raise SystemExit(f"[smoke] FAIL {label}: PingTi action missing {missing}")
    if len(action) != len(PINGTI_FOLLOWER_MOTORS):
        raise SystemExit(f"[smoke] FAIL {label}: expected 8 motors, got {sorted(action)}")
    for joint in PINGTI_DUAL_JOINTS:
        motors = PINGTI_JOINT_TO_FOLLOWER_MOTORS[joint]
        a = action[f"{motors[0]}.pos"]
        b = action[f"{motors[1]}.pos"]
        if a != -b:
            raise SystemExit(f"[smoke] FAIL {label}: {joint} dual {a} vs {b} (expected secondary=-primary)")


def run_keyboard() -> None:
    print("[smoke] case=keyboard task=Isaac-PING-TI-Teleop-v0 se3_dim=7", flush=True)
    env = _make_env("keyboard")
    pingti = MockPingTiFollower()
    try:
        print(f"[smoke] keyboard action_space={env.action_space}", flush=True)
        before_ee = _ee_xyz(env)
        before_j = _named_joints(env)
        action_z = [0.0, 0.0, KEYBOARD_DZ, 0.0, 0.0, 0.0, 1.0]
        if len(action_z) != SE3_ACTION_DIM:
            raise RuntimeError("keyboard smoke action is not 7-D")
        _step(env, action_z, args_cli.num_steps, pingti=pingti)
        after_z = _ee_xyz(env)
        dz = after_z[2] - before_ee[2]
        print(
            f"[smoke] keyboard ee_before=({before_ee[0]:.4f},{before_ee[1]:.4f},{before_ee[2]:.4f}) "
            f"ee_after_z=({after_z[0]:.4f},{after_z[1]:.4f},{after_z[2]:.4f}) dz={dz:.4f}",
            flush=True,
        )
        if dz < EE_MIN_DELTA:
            raise SystemExit(
                f"[smoke] FAIL keyboard: expected EE +z > {EE_MIN_DELTA} m, got dz={dz:.4f}"
            )
        mid_g = _named_joints(env)["gripper_moving"]
        action_close = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        _step(env, action_close, args_cli.num_steps, pingti=pingti)
        after_j = _named_joints(env)
        after_g = after_j["gripper_moving"]
        after_ee = _ee_xyz(env)
        print(
            f"[smoke] keyboard gripper {mid_g:.4f} -> {after_g:.4f} "
            f"ee=({after_ee[0]:.4f},{after_ee[1]:.4f},{after_ee[2]:.4f})",
            flush=True,
        )
        if after_g > mid_g - JOINT_MIN_DELTA:
            raise SystemExit(
                f"[smoke] FAIL keyboard: expected gripper close, {mid_g:.4f} -> {after_g:.4f}"
            )
        for name in PINGTI_JOINTS:
            print(
                f"[smoke] keyboard joint {name} {before_j.get(name, float('nan')):.4f} -> "
                f"{after_j.get(name, float('nan')):.4f}",
                flush=True,
            )
        hw = pingti_follower_action_from_joints(tuple(after_j[n] for n in PINGTI_JOINTS))
        _assert_pingti_duals(hw, "keyboard mock pingti")
        expected_sends = 2 * args_cli.num_steps
        if len(pingti.sent) != expected_sends:
            raise SystemExit(
                f"[smoke] FAIL keyboard: mock pingti sent {len(pingti.sent)} != {expected_sends}"
            )
        print(
            f"[smoke] keyboard mock pingti sends={len(pingti.sent)} last_dual_shoulder="
            f"{hw['shoulder_lift.pos']:.3f}/{hw['shoulder_lift_secondary.pos']:.3f}",
            flush=True,
        )
        print("[smoke] PASS keyboard PingTi EE +z and gripper close", flush=True)
    finally:
        env.close()

    print("[smoke] case=keyboard-y from home pose", flush=True)
    env = _make_env("keyboard")
    pingti = MockPingTiFollower()
    try:
        before_ee = _ee_xyz(env)
        action_y = [0.0, KEYBOARD_DY, 0.0, 0.0, 0.0, 0.0, 1.0]
        _step(env, action_y, args_cli.num_steps, pingti=pingti)
        after_ee = _ee_xyz(env)
        dy = after_ee[1] - before_ee[1]
        print(
            f"[smoke] keyboard-y ee_before=({before_ee[0]:.4f},{before_ee[1]:.4f},{before_ee[2]:.4f}) "
            f"ee_after=({after_ee[0]:.4f},{after_ee[1]:.4f},{after_ee[2]:.4f}) dy={dy:.4f}",
            flush=True,
        )
        if abs(dy) < EE_MIN_DELTA:
            raise SystemExit(
                f"[smoke] FAIL keyboard-y: expected |EE dy| > {EE_MIN_DELTA} m, got dy={dy:.4f}"
            )
        hw = pingti_follower_action_from_joints(tuple(_named_joints(env)[n] for n in PINGTI_JOINTS))
        _assert_pingti_duals(hw, "keyboard-y mock pingti")
        print("[smoke] PASS keyboard PingTi EE moved in y", flush=True)
    finally:
        env.close()


def run_leader() -> None:
    print("[smoke] case=leader scripted SO101 get_action dict -> PingTi 6-D", flush=True)
    env = _make_env("so101leader")
    pingti = MockPingTiFollower()
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
            _step(env, list(target), args_cli.num_steps, pingti=pingti)
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
            hw = pingti_follower_action_from_joints(tuple(after[n] for n in PINGTI_JOINTS))
            _assert_pingti_duals(hw, f"leader {motor}")
            duals = PINGTI_JOINT_TO_FOLLOWER_MOTORS[joint]
            if abs(hw[f"{duals[0]}.pos"]) < 1.0:
                failures.append(f"{motor} sim joints did not expand to PingTi {duals[0]}")
        before_g = _named_joints(env)["gripper_moving"]
        grip_target = pingti_joint_pos_from_leader(leader_state_hold({"gripper": 80.0}))
        print(f"[smoke] leader commanding gripper -> gripper_moving target={grip_target[-1]:.4f}", flush=True)
        _step(env, list(grip_target), args_cli.num_steps, pingti=pingti)
        after_g = _named_joints(env)["gripper_moving"]
        print(
            f"[smoke] leader motor=gripper -> gripper_moving "
            f"before={before_g:.4f} after={after_g:.4f} target={grip_target[-1]:.4f}",
            flush=True,
        )
        if after_g - before_g < JOINT_MIN_DELTA and after_g < JOINT_MIN_DELTA:
            failures.append(f"gripper_moving did not open (after={after_g:.4f})")
        if not pingti.sent:
            failures.append("mock pingti recorded no leader sends")
        print(f"[smoke] leader mock pingti sends={len(pingti.sent)}", flush=True)
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
