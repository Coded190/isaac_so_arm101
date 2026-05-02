# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OBS recording variant of scripts/vla/vla_data_gen.py.

Runs the same SprayOracle FSM and palm-spray simulation so OBS can capture the
Isaac Sim viewport. All progress prints, telemetry heartbeats, and LeRobot
dataset writing are removed — the loop only drives the simulation. Visual
markers (blue hover sphere) and the brief rest-pose hold are kept so the
recorded footage matches what the data-gen run looks like.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA palm-spray recording for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate, default is 1.")
parser.add_argument("--task", type=str, default="None", help="Name of the task.")
parser.add_argument("--top_leaf_cull_prob", type=float, default=0.5, help="Probability of culling the top leaves for a given episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

import carb
carb.settings.get_settings().set_string("/log/level", "error")
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym

import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg

# ─── Kinematic / FSM constants (mirror scripts/vla/vla_data_gen.py) ────────
ACTION_CLAMP = 0.5
POSITION_GAIN = 0.75
SPRAY_DURATION = 60

REST_POSE_VALUES = [
    0.0,
    float(np.deg2rad(48.5)),
    float(np.deg2rad(-58.6)),
    1.2,
    0.0,
    0.0,
]

LEAF_CULL_Z_OFFSET = 0.03
# When an episode culls leaves, the keep-ratio is sampled uniformly from
# [LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX]. Lower keep-ratio = more leaves
# removed: 0.5 culls 50% of the top leaves, 0.8 culls 20%.
LEAF_KEEP_RATIO_MIN = 0.5
LEAF_KEEP_RATIO_MAX = 0.8
HOVER_OFFSET = np.array([0.0, 0.0, 0.35])

DOWN_QUAT = np.array([0.2588, 0.0, -0.9659, 0.0])
ORIENTATION_CLAMP = 0.2

# Per-episode root randomization. The robot is placed at a random angle on
# a circle centered on the tree, keeping its distance from the tree (and
# its world-Z height) constant. Yaw is set so the robot faces the tree.
ANGLE_RANDOM_RANGE = float(np.pi)  # ±180° → full 360° around the tree.
# Positive value moves the robot closer to the tree by this many meters
# (subtracted from the default radius). 0.0 keeps the original distance.
TREE_INWARD_OFFSET = 0.0
# Lower bound on the radius — never go closer than this to the trunk.
MIN_TREE_RADIUS = 0.08
# Reject any placement whose XY position lands within this many meters of
# any palm leaf — the base would otherwise spawn inside / through a leaf.
LEAF_CLEARANCE = 0.10
PLACEMENT_MAX_ATTEMPTS = 15


def get_palm_root_path(env_id):
    return f"/World/envs/env_{env_id}/Scene/Palm"


def _iter_leaf_prims(stage, palm_root_path):
    from pxr import UsdGeom
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm:
        return
    for child in palm.GetChildren():
        name = child.GetName()
        if name.startswith("leaf_") and UsdGeom.Xformable(child):
            yield child


def _leaf_world_positions(stage, palm_root_path):
    from pxr import UsdGeom
    out = []
    for prim in _iter_leaf_prims(stage, palm_root_path):
        xf = UsdGeom.Xformable(prim)
        try:
            wp = xf.ComputeLocalToWorldTransform(0).ExtractTranslation()
            out.append((prim, np.array([wp[0], wp[1], wp[2]], dtype=np.float64)))
        except Exception:
            continue
    return out


def set_leaf_prims_active(stage, palm_root_path, active=True):
    for prim in _iter_leaf_prims(stage, palm_root_path):
        prim.SetActive(active)


def get_crown_centroid(stage, palm_root_path):
    leaves = _leaf_world_positions(stage, palm_root_path)
    if not leaves:
        return np.array([0.0, 0.0, 5.0])
    positions = np.stack([p for _, p in leaves], axis=0)
    return positions.mean(axis=0)


def remove_top_leaves(stage, palm_root_path, crown_z, keep_ratio,
                      z_threshold_offset=LEAF_CULL_Z_OFFSET):
    cull_z = crown_z + z_threshold_offset
    leaves = _leaf_world_positions(stage, palm_root_path)
    top_leaves = [(prim, pos[2]) for prim, pos in leaves if pos[2] > cull_z]
    top_leaves.sort(key=lambda x: -x[1])
    n_remove = int(len(top_leaves) * (1.0 - keep_ratio))
    for prim, _ in top_leaves[:n_remove]:
        prim.SetActive(False)


def spawn_target_marker(stage, position_world, marker_path, radius=0.04, color=(1.0, 0.0, 0.0)):
    from pxr import UsdGeom, Gf, Sdf
    if stage.GetPrimAtPath(marker_path):
        stage.RemovePrim(marker_path)
    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(marker_path))
    sphere.GetRadiusAttr().Set(radius)
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(position_world[0]),
                                         float(position_world[1]),
                                         float(position_world[2])))
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


def spawn_target_markers(stage, target_positions, env_ids=None, marker_type="spray", color=(1.0, 0.0, 0.0)):
    if env_ids is None:
        env_ids = range(target_positions.shape[0])
    for env_id in env_ids:
        marker_path = f"/World/debug_target_markers/{marker_type}/env_{env_id}"
        spawn_target_marker(stage, target_positions[env_id], marker_path=marker_path, color=color)


def set_rest_pose(env, rest_pose_tensor, env_ids=None):
    robot = env.unwrapped.scene["robot"]
    if env_ids is None:
        joint_pos = rest_pose_tensor.expand(env.unwrapped.num_envs, -1)
        zero_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, zero_vel)
        return
    env_ids_t = torch.as_tensor(env_ids, device=rest_pose_tensor.device, dtype=torch.long)
    joint_pos = rest_pose_tensor.expand(env_ids_t.shape[0], -1)
    zero_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, zero_vel, env_ids=env_ids_t)


def get_deterministic_target(stage, palm_root_path, offset=HOVER_OFFSET):
    return get_crown_centroid(stage, palm_root_path) + offset


def prepare_episode_targets(stage, palm_root_paths, episode_rng, cull_prob, env_ids=None):
    if env_ids is None:
        env_ids = list(range(len(palm_root_paths)))
    targets = []
    for env_id in env_ids:
        palm_root_path = palm_root_paths[env_id]
        set_leaf_prims_active(stage, palm_root_path, active=True)
        crown_centroid = get_crown_centroid(stage, palm_root_path)
        if episode_rng.random() < cull_prob:
            keep_ratio = float(episode_rng.uniform(
                LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX,
            ))
            remove_top_leaves(
                stage, palm_root_path,
                crown_z=crown_centroid[2],
                keep_ratio=keep_ratio,
            )
        targets.append(get_deterministic_target(stage, palm_root_path))
    return np.asarray(targets, dtype=np.float32)


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _closest_leaf_dist_xy(stage, palm_root_path, robot_xy):
    """Closest XY distance from ``robot_xy`` to any active palm leaf, or
    ``None`` if there are no leaves to test against."""
    leaves = _leaf_world_positions(stage, palm_root_path)
    if not leaves:
        return None
    rxy = np.array([float(robot_xy[0]), float(robot_xy[1])], dtype=np.float64)
    return float(min(
        np.hypot(pos[0] - rxy[0], pos[1] - rxy[1]) for _, pos in leaves
    ))


def _yaw_to_quat_wxyz(yaw):
    """Quaternion (w, x, y, z) for a rotation about the world Z axis."""
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _yaw_from_quat_wxyz(q):
    """Extract yaw (rotation about Z) from a (w, x, y, z) quaternion."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def _rotate_vec_z(v, yaw):
    """Rotate a 3-vector by ``yaw`` radians about the +Z axis."""
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([c * v[0] - s * v[1],
                     s * v[0] + c * v[1],
                     v[2]], dtype=np.float64)


def randomize_robot_root_pose(env, stage, palm_root_paths, episode_rng,
                              env_ids=None,
                              angle_range=ANGLE_RANDOM_RANGE,
                              inward_offset=TREE_INWARD_OFFSET):
    """Place the robot at a random angle on a circle around the tree.

    With ``angle_range = π`` the robot can spawn anywhere on the full 360°
    circle around the trunk. Uses the crown-centroid XY (same XY as the
    blue hover marker) as the horizontal trunk target, but at base height
    for the face vector. The XY distance to the trunk is preserved
    (optionally shrunk by ``inward_offset``) and world Z is unchanged.
    Yaw is set so the robot faces the tree from its new position. A
    yellow marker is dropped at the target so you can see what the base
    is aiming at.
    """
    robot = env.unwrapped.scene["robot"]
    device = robot.data.default_root_state.device
    if env_ids is None:
        env_ids = list(range(env.unwrapped.num_envs))
    env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long)

    new_root = robot.data.default_root_state[env_ids_t].clone()
    trunk_xys = []
    for i, env_id in enumerate(env_ids):
        palm_root_path = palm_root_paths[env_id]
        crown = get_crown_centroid(stage, palm_root_path)
        trunk_xy = np.array([float(crown[0]), float(crown[1])], dtype=np.float64)
        trunk_xys.append(trunk_xy)
        default_xy = new_root[i, :2].cpu().numpy().astype(np.float64)
        rel = default_xy - trunk_xy
        radius0 = float(np.linalg.norm(rel))
        bearing0 = float(np.arctan2(rel[1], rel[0]))

        new_x = new_y = None
        last_dist = None
        for attempt in range(PLACEMENT_MAX_ATTEMPTS):
            theta = float(episode_rng.uniform(-angle_range, angle_range))
            radius = max(radius0 - float(inward_offset), MIN_TREE_RADIUS)
            bearing = bearing0 + theta
            cand_x = float(trunk_xy[0]) + radius * float(np.cos(bearing))
            cand_y = float(trunk_xy[1]) + radius * float(np.sin(bearing))
            d = _closest_leaf_dist_xy(stage, palm_root_path, (cand_x, cand_y))
            last_dist = d
            if d is None or d >= LEAF_CLEARANCE:
                new_x, new_y = cand_x, cand_y
                if attempt > 0:
                    print(f"[reset env {env_id}] leaf-clear placement found on "
                          f"attempt {attempt + 1} (closest leaf {d:.3f} m)",
                          flush=True)
                break
            print(f"[reset env {env_id}] attempt {attempt + 1}/"
                  f"{PLACEMENT_MAX_ATTEMPTS}: leaf at {d:.3f} m "
                  f"(< {LEAF_CLEARANCE:.2f}), retrying",
                  flush=True)
        if new_x is None:
            new_x, new_y = cand_x, cand_y
            print(f"[reset env {env_id}] no leaf-clear placement after "
                  f"{PLACEMENT_MAX_ATTEMPTS} attempts; using last sample "
                  f"(closest leaf {last_dist:.3f} m)",
                  flush=True)

        face_x = float(trunk_xy[0]) - new_x
        face_y = float(trunk_xy[1]) - new_y
        # Forward axis is -Y at yaw=0 for this robot (verified empirically).
        yaw = float(np.arctan2(face_x, -face_y))
        quat = _yaw_to_quat_wxyz(yaw)

        new_root[i, 0] = new_x
        new_root[i, 1] = new_y
        new_root[i, 3:7] = torch.tensor(quat, dtype=new_root.dtype, device=device)

    robot.write_root_pose_to_sim(new_root[:, :7], env_ids=env_ids_t)
    # Zero out root linear + angular velocity so a leftover spin from the
    # previous physics step doesn't rotate the base after we teleport it.
    zero_vel = torch.zeros((len(env_ids), 6), dtype=new_root.dtype, device=device)
    robot.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_t)
    # Re-apply the rest-pose joint state so the arm/column don't carry
    # over a non-zero base_yaw / shoulder pose from the previous episode.
    rest_pose_tensor = torch.tensor(REST_POSE_VALUES, dtype=new_root.dtype,
                                    device=device)
    set_rest_pose(env, rest_pose_tensor, env_ids=env_ids)

    # Drop a yellow marker at (trunk_xy, base_z) so the user can verify the
    # base is being aimed at the right horizontal target.
    for i, env_id in enumerate(env_ids):
        trunk_xy = trunk_xys[i]
        base_z = float(new_root[i, 2].cpu())
        marker_xyz = (float(trunk_xy[0]), float(trunk_xy[1]), base_z)
        spawn_target_marker(
            stage, marker_xyz,
            marker_path=f"/World/debug_target_markers/face_target/env_{env_id}",
            radius=0.10, color=(1.0, 1.0, 0.0),
        )
        robot_x = float(new_root[i, 0].cpu())
        robot_y = float(new_root[i, 1].cpu())
        yaw_deg = float(np.rad2deg(_yaw_from_quat_wxyz(
            new_root[i, 3:7].cpu().numpy()
        )))
        print(f"[reset env {env_id}] base=({robot_x:+.3f}, {robot_y:+.3f}, "
              f"{base_z:+.3f}) yaw={yaw_deg:+7.2f}deg "
              f"yellow_marker={marker_xyz}", flush=True)


def _quat_to_axis_angle(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.zeros(3)
    q = q / n
    if q[0] < 0.0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-6:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    axis = q[1:4] / sin_half
    return axis * angle


class SprayOracle:
    """6-state FSM: 0 approach waypoint, 1 approach hover, 2 descend/settle,
    3 spray, 4 success hold, 5 fail hold."""

    POSITION_THRESHOLD = 0.50
    MAX_STATE_STEPS = 200

    def __init__(self):
        self.state = 0
        self.spray_counter = 0
        self.state_steps = 0
        self.completed = False
        self.timed_out = False
        self.approach_waypoint = None
        self.fail_pos = None

    def _advance(self, next_state):
        self.state = next_state
        self.state_steps = 0

    @staticmethod
    def _cap_vector_norm(vector, max_norm):
        norm = np.linalg.norm(vector)
        if norm <= max_norm or norm == 0.0:
            return vector
        return vector * (max_norm / norm)

    def _position_command(self, error_vector):
        return self._cap_vector_norm(POSITION_GAIN * error_vector, ACTION_CLAMP)

    def compute_action(self, ee_pos, ee_quat, hover_target):
        action = np.zeros(7)
        action[6] = 0.0

        if self.approach_waypoint is None:
            self.approach_waypoint = np.copy(hover_target)
            self.approach_waypoint[0] = (ee_pos[0] + hover_target[0]) / 2.0
            self.approach_waypoint[1] = (ee_pos[1] + hover_target[1]) / 2.0

        err_quat = _quat_multiply(DOWN_QUAT, _quat_conjugate(ee_quat))
        err_axis_angle = _quat_to_axis_angle(err_quat)
        action[3:6] = self._cap_vector_norm(err_axis_angle, ORIENTATION_CLAMP)

        self.state_steps += 1

        if self.state == 0:
            err_to_waypoint = self.approach_waypoint - ee_pos
            action[0:3] = self._position_command(err_to_waypoint)
            if np.linalg.norm(err_to_waypoint) < self.POSITION_THRESHOLD:
                self._advance(1)
            elif self.state_steps >= self.MAX_STATE_STEPS:
                self.timed_out = True
                self.fail_pos = np.copy(ee_pos)
                self._advance(5)

        elif self.state == 1:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)
            if np.linalg.norm(err_to_hover) < self.POSITION_THRESHOLD:
                self._advance(2)
            elif self.state_steps >= self.MAX_STATE_STEPS:
                self.timed_out = True
                self.fail_pos = np.copy(ee_pos)
                self._advance(5)

        elif self.state == 2:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)
            if self.state_steps >= 30:
                self.spray_counter = SPRAY_DURATION
                self._advance(3)

        elif self.state == 3:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)
            self.spray_counter -= 1
            if self.spray_counter <= 0:
                self.completed = True
                self._advance(4)

        elif self.state == 4:
            err_to_hover = hover_target - ee_pos
            action[0:3] = self._position_command(err_to_hover)

        elif self.state == 5:
            err_to_fail = self.fail_pos - ee_pos
            action[0:3] = self._position_command(err_to_fail)

        return action


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    default_pos = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.robot.init_state.pos = (
        default_pos[0] + 0.05,
        default_pos[1] - 0.179,
        default_pos[2] + 0.1524,
    )
    env_cfg.scene.robot.init_state.joint_pos = {
        "base_yaw": REST_POSE_VALUES[0],
        "shoulder_pitch": REST_POSE_VALUES[1],
        "elbow_pitch": REST_POSE_VALUES[2],
        "wrist_pitch": REST_POSE_VALUES[3],
        "wrist_roll": REST_POSE_VALUES[4],
        "gripper_moving": REST_POSE_VALUES[5],
    }
    env_cfg.episode_length_s = 7.5

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    sim_device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    palm_root_paths = [get_palm_root_path(env_id) for env_id in range(num_envs)]
    episode_rng = np.random.default_rng(getattr(args_cli, "seed", None))

    randomize_robot_root_pose(
        env=env, stage=stage, palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
    )

    current_hover_targets = np.zeros((num_envs, 3), dtype=np.float32)
    current_hover_targets[:] = prepare_episode_targets(
        stage=stage,
        palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
        cull_prob=args_cli.top_leaf_cull_prob,
    )

    oracles = [SprayOracle() for _ in range(num_envs)]
    spawn_target_markers(stage, current_hover_targets, marker_type="hover", color=(0.0, 0.0, 1.0))

    moving_gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper")
    moving_gripper_idx = moving_gripper_indices[0]

    while simulation_app.is_running():
        ee_pos_all = env.unwrapped.scene["robot"].data.body_pos_w[:, moving_gripper_idx].cpu().numpy()
        ee_quat_all = env.unwrapped.scene["robot"].data.body_quat_w[:, moving_gripper_idx].cpu().numpy()
        root_quat_all = env.unwrapped.scene["robot"].data.root_quat_w.cpu().numpy()

        action_batch = np.zeros((num_envs, 7), dtype=np.float32)
        for env_id in range(num_envs):
            env_action = oracles[env_id].compute_action(
                ee_pos_all[env_id], ee_quat_all[env_id],
                hover_target=current_hover_targets[env_id],
            )
            # IK action term interprets the position delta in body (root)
            # frame; FSM computes it in world frame. Rotate position only
            # — the orientation transform makes things worse in this
            # config, so leave the axis-angle delta in world frame.
            base_yaw = _yaw_from_quat_wxyz(root_quat_all[env_id])
            env_action[0:3] = _rotate_vec_z(env_action[0:3], -base_yaw)
            env_action[:3] = np.clip(env_action[:3], -ACTION_CLAMP, ACTION_CLAMP)
            action_batch[env_id] = env_action

        action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=sim_device)

        _, _, terminated, truncated, _ = env.step(action_tensor)

        terminated_t = torch.as_tensor(terminated, device=sim_device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=sim_device, dtype=torch.bool)
        done_mask = torch.logical_or(terminated_t, truncated_t).reshape(-1)
        done_env_ids_t = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        done_env_ids = done_env_ids_t.cpu().tolist() if done_env_ids_t.numel() > 0 else []

        if done_env_ids:
            randomize_robot_root_pose(
                env=env, stage=stage, palm_root_paths=palm_root_paths,
                episode_rng=episode_rng, env_ids=done_env_ids,
            )
            refreshed_targets = prepare_episode_targets(
                stage=stage,
                palm_root_paths=palm_root_paths,
                episode_rng=episode_rng,
                cull_prob=args_cli.top_leaf_cull_prob,
                env_ids=done_env_ids,
            )
            for i, env_id in enumerate(done_env_ids):
                current_hover_targets[env_id] = refreshed_targets[i]
                oracles[env_id] = SprayOracle()
            spawn_target_markers(
                stage, current_hover_targets,
                env_ids=done_env_ids, marker_type="hover", color=(0.0, 0.0, 1.0),
            )

    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()
