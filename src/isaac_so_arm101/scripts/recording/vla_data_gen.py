# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OBS recording variant of scripts/vla/vla_data_gen.py.

Runs the same SprayOracle FSM and palm-spray simulation so OBS can capture the
Isaac Sim viewport. All progress prints, telemetry heartbeats, and LeRobot
dataset writing are removed — the loop only drives the simulation. Visual
markers (blue hover sphere) and the brief rest-pose hold are kept so the
recorded footage matches what the data-gen run looks like.

Windows launch:
    .\\python.bat kantas\\isaac_so_arm101\\src\\isaac_so_arm101\\scripts\\recording\\vla_data_gen.py ^
        --task Isaac-PING-TI-VLA-v0 --num_envs 1 --enable_cameras 2>$null
"""

import argparse
import re
import sys


class _NoiseFilter:
    """Wraps a text stream and drops lines whose content matches any of
    a fixed set of regex patterns. Used to scrub the cosmetic PhysX
    joint warnings (and similar noise) from stderr/stdout so they don't
    show up in screen recordings.
    """

    _DROP_PATTERNS = [
        re.compile(r"PxJoint::setActors"),
        re.compile(r"CreateJoint - cannot create"),
        re.compile(r"FabricManager::initializePointInstancer"),
        re.compile(r"primvars:displayColor:indices not found"),
        re.compile(r"omni\.kit\.notification_manager.*PhysX"),
        re.compile(r"omni\.kit\.notification_manager.*Physics USD Load"),
        re.compile(r"omni\.hydra.*update topology"),
        re.compile(r"gpu::unstable::IMemoryBudgetManagerFactory"),
    ]

    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._buf = ""

    def write(self, s):
        if not s:
            return
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(p.search(line) for p in self._DROP_PATTERNS):
                self._wrapped.write(line + "\n")

    def flush(self):
        if self._buf:
            if not any(p.search(self._buf) for p in self._DROP_PATTERNS):
                self._wrapped.write(self._buf)
            self._buf = ""
        try:
            self._wrapped.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# Install stderr/stdout filters BEFORE any Kit/Carb imports so the very
# first physics-init warnings get scrubbed too.
sys.stderr = _NoiseFilter(sys.stderr)
sys.stdout = _NoiseFilter(sys.stdout)


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLA palm-spray recording for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate, default is 1.")
parser.add_argument("--task", type=str, default="None", help="Name of the task.")
parser.add_argument("--top_leaf_cull_prob", type=float, default=1.0, help="Probability of culling the top leaves for a given episode (1.0 = always cull).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

import carb
# Just the global level. Per-channel hammering was triggering
# "getStringRawInternal: item ... is not a string" errors and timing
# out the viewport. Console noise is filtered by the Python stderr
# wrapper at the top of this file plus 2>$null in the launch command.
carb.settings.get_settings().set_string("/log/level", "error")
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym

import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg

# ─── Kinematic / FSM constants (mirror scripts/vla/vla_data_gen.py) ────────
ACTION_CLAMP = 0.25      # was 0.5 — caps per-step Cartesian delta (smaller = slower motion)
POSITION_GAIN = 0.35     # was 0.75 — proportional gain on position error (smaller = slower approach)
SPRAY_DURATION = 60

REST_POSE_VALUES = [
    0.0,                       # base_yaw
    float(np.deg2rad(-40.0)),  # shoulder_pitch  — tilted back further (was -20°; less forward reach)
    float(np.deg2rad(-73.0)),  # elbow_pitch     — near soft lower limit -74.7° (max fold-back)
    -0.8,                      # wrist_pitch     — tilted down ~46° at rest (less sky)
    0.0,                       # wrist_roll
    0.0,                       # gripper_moving
]

LEAF_CULL_Z_OFFSET = 0.03
# When an episode culls leaves, the keep-ratio is sampled uniformly from
# [LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX]. Lower keep-ratio = more leaves
# removed: 0.5 culls 50% of the top leaves, 0.8 culls 20%.
LEAF_KEEP_RATIO_MIN = 0.5
LEAF_KEEP_RATIO_MAX = 0.8
HOVER_OFFSET = np.array([0.0, 0.0, 0.35])
# Meters to pull the hover target from the crown XY back toward the robot XY.
# 0.0 = directly above crown (used to overshoot past). Increase to land short of crown.
HOVER_PULLBACK_M = 0.18

DOWN_QUAT = np.array([0.2588, 0.0, -0.9659, 0.0])
# Bumped 0.2 → 0.5 so the wrist rotates toward DOWN_QUAT ~2.5× faster
# per step. With ANGLE_RANDOM_RANGE shrunk to ±45° the IK has easy
# reach, so the orientation command no longer needs to be conservative.
ORIENTATION_CLAMP = 0.5

# Force a re-randomization only when the FSM has already given up
# (state 5 = fail_hold) AND the EE is sitting close to a leaf. This
# guards against the auto-reset firing during a normal approach where
# the EE passes through the canopy on its way to the hover target.
LEAF_STUCK_DISTANCE = 0.05
LEAF_STUCK_STEPS = 30
LEAF_STUCK_FAIL_STATE = 5

# Verbose console output: FSM state transitions and stuck-reset events.
DEBUG_VERBOSE = True

# Recording camera follows the robot. Each reset picks a random left/right
# side of the robot's "robot → tree" axis and points the camera at the
# midpoint, lifted to frame both the base and the canopy.
CAMERA_FOLLOWS_ROBOT = True
CAMERA_LATERAL_OFFSET = 1.10  # meters out to the side of the robot
CAMERA_HEIGHT_OFFSET = 0.75   # meters above the base height
# Look-at point sits at the midpoint XY between robot and crown, lifted
# this much above the base height to favor the canopy in the framing.
CAMERA_TARGET_LIFT = 0.20

# Per-episode root randomization. The robot is placed at a random angle on
# a circle centered on the tree, keeping its distance from the tree (and
# its world-Z height) constant. Yaw is set so the robot faces the tree.
ANGLE_RANDOM_RANGE = float(np.deg2rad(45.0))  # ±45° forward arc.
# Positive value moves the robot closer to the tree (subtracted from the
# default radius). NEGATIVE pushes it further away. 0.0 keeps default.
TREE_INWARD_OFFSET = 0.0
# Lower bound on the radius — never go closer than this to the trunk.
MIN_TREE_RADIUS = 0.08
# Reject any placement whose XY position lands within this many meters of
# any palm leaf — the base would otherwise spawn inside / through a leaf.
LEAF_CLEARANCE = 0.10
PLACEMENT_MAX_ATTEMPTS = 15


def get_palm_root_path(env_id):
    return f"/World/envs/env_{env_id}/Scene/palm_tree_crown"


def _get_palm_crown_prim(stage, palm_root_path):
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm or not palm.IsValid():
        return None
    crown = palm.GetChild("crown")
    if crown and crown.IsValid():
        return crown
    return palm


def disable_palm_physics(stage, palm_root_path):
    """Disable PhysX colliders and rigid-body simulation on every prim in
    the palm subtree. Leaves stay visible but become inert — the robot
    arm passes through them without disturbance, which is what we want
    for clean recording captures.
    """
    try:
        from pxr import Usd, UsdGeom
    except Exception:
        return
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm:
        return
    crown = _get_palm_crown_prim(stage, palm_root_path)
    crown_paths = set()
    if crown:
        for prim in Usd.PrimRange(crown):
            crown_paths.add(prim.GetPath())

    for prim in Usd.PrimRange(palm):
        if prim == palm:
            continue
        prim_name = prim.GetName().lower()
        if crown and prim.GetPath() in crown_paths and prim.IsA(UsdGeom.Mesh) and (
            prim_name.startswith("leaf_") or prim_name.startswith("leaf_b_")
        ):
            col_attr = prim.GetAttribute("physics:collisionEnabled")
            if col_attr:
                col_attr.Set(False)
            rb_attr = prim.GetAttribute("physics:rigidBodyEnabled")
            if rb_attr:
                rb_attr.Set(False)
            continue

        col_attr = prim.GetAttribute("physics:collisionEnabled")
        if col_attr:
            col_attr.Set(False)
        rb_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if rb_attr:
            rb_attr.Set(False)
        # Also turn off the joint prims that linked these leaves to
        # the trunk — without live rigid bodies on each end, PhysX
        # otherwise fills the log with "cannot create a joint between
        # static bodies" errors. Cosmetic; behavior is unchanged.
        if prim.GetName().lower() in {"trunk", "trunk_top"}:
            continue
        if "Joint" in prim.GetName():
            prim.SetActive(False)


def _iter_leaf_prims(stage, palm_root_path):
    from pxr import Usd, UsdGeom
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm:
        return
    crown = _get_palm_crown_prim(stage, palm_root_path)
    search_root = crown if crown else palm
    for child in Usd.PrimRange(search_root):
        if child == search_root:
            continue
        name = child.GetName().lower()
        if (name.startswith("leaf_") or name.startswith("leaf_b_")) and UsdGeom.Xformable(child):
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
    """Activate or deactivate every leaf under palm_root_path.

    When activating, iterate via GetAllChildren() so we also pick up
    leaves previously SetActive(False)'d — plain GetChildren() filters
    out inactive prims and would leave them dead forever, causing the
    tree to get progressively balder across episodes.
    """
    from pxr import UsdGeom
    for child in _iter_leaf_prims(stage, palm_root_path):
        if not UsdGeom.Xformable(child):
            continue
        if active:
            child.SetActive(True)
        else:
            child.SetActive(False)
    else:
        for prim in _iter_leaf_prims(stage, palm_root_path):
            prim.SetActive(False)


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
    UsdGeom.Imageable(sphere).MakeInvisible()


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


def prepare_episode_targets(stage, palm_root_paths, episode_rng, cull_prob,
                             robot_xys=None, env_ids=None):
    """Compute hover targets per env. If ``robot_xys`` is given, the target is
    pulled HOVER_PULLBACK_M meters along the crown→robot XY direction so the
    EE lands short of the trunk instead of past it."""
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
        target = get_deterministic_target(stage, palm_root_path)
        if robot_xys is not None and HOVER_PULLBACK_M > 0.0:
            rxy = np.asarray(robot_xys[env_id], dtype=np.float64)
            cxy = np.array([crown_centroid[0], crown_centroid[1]], dtype=np.float64)
            back = rxy - cxy
            n = float(np.linalg.norm(back))
            if n > 1e-6:
                back /= n
                target[0] += float(back[0]) * HOVER_PULLBACK_M
                target[1] += float(back[1]) * HOVER_PULLBACK_M
        targets.append(target)
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


def _update_recording_camera(robot_xy, crown_xy, base_z, episode_rng,
                             lateral_offset=CAMERA_LATERAL_OFFSET,
                             height_offset=CAMERA_HEIGHT_OFFSET,
                             target_lift=CAMERA_TARGET_LIFT):
    """Move the active viewport camera to a random left/right side of the
    line from the robot to the crown, slightly above the base, looking at
    the midpoint between them. Silently does nothing if the Isaac Sim
    viewport utility isn't importable in this build.
    """
    set_camera_view = None
    try:
        from isaacsim.core.utils.viewports import set_camera_view  # type: ignore
    except Exception:
        try:
            from omni.isaac.core.utils.viewports import set_camera_view  # type: ignore
        except Exception:
            return

    forward = np.array([float(crown_xy[0]) - float(robot_xy[0]),
                        float(crown_xy[1]) - float(robot_xy[1])],
                       dtype=np.float64)
    n = float(np.linalg.norm(forward))
    if n < 1e-6:
        return
    forward /= n
    perpendicular = np.array([-forward[1], forward[0]], dtype=np.float64)

    side = 1.0 if episode_rng.random() < 0.5 else -1.0
    camera_pos = [
        float(robot_xy[0] + side * lateral_offset * perpendicular[0]),
        float(robot_xy[1] + side * lateral_offset * perpendicular[1]),
        float(base_z + height_offset),
    ]
    target = [
        float(0.5 * (float(robot_xy[0]) + float(crown_xy[0]))),
        float(0.5 * (float(robot_xy[1]) + float(crown_xy[1]))),
        float(base_z + target_lift),
    ]
    try:
        set_camera_view(eye=camera_pos, target=target)
    except Exception:
        return
    if DEBUG_VERBOSE:
        side_name = "right" if side > 0 else "left"
        print(f"[camera] follow → {side_name} side  eye={camera_pos}  "
              f"target={target}", flush=True)


def _closest_leaf_dist_3d(stage, palm_root_path, point_xyz):
    """Closest 3D distance from ``point_xyz`` to any active palm leaf,
    or ``None`` if no leaves are present."""
    leaves = _leaf_world_positions(stage, palm_root_path)
    if not leaves:
        return None
    pt = np.array([float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2])],
                  dtype=np.float64)
    return float(min(np.linalg.norm(pos - pt) for _, pos in leaves))


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
    # base is being aimed at the right horizontal target. Also re-aim the
    # active viewport camera at the first reset env so OBS captures a
    # fresh side-view of the robot+crown each episode.
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

    if CAMERA_FOLLOWS_ROBOT and env_ids:
        first_id = env_ids[0]
        first_idx = list(env_ids).index(first_id)
        _update_recording_camera(
            robot_xy=(float(new_root[first_idx, 0].cpu()),
                      float(new_root[first_idx, 1].cpu())),
            crown_xy=trunk_xys[first_idx],
            base_z=float(new_root[first_idx, 2].cpu()),
            episode_rng=episode_rng,
        )


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

    # 0.20 m advance threshold (vs original 0.50): the rise from base
    # level to the hover target is only ~0.35 m, so a 0.50 threshold
    # would let the FSM "arrive" before any actual rise. 0.20 forces
    # the EE to genuinely approach the blue marker before settling
    # while still being reachable with base_yaw out of the IK chain.
    POSITION_THRESHOLD = 0.20
    # 400 steps so the 4-DOF IK has time to lift the EE 35 cm without
    # the FSM timing out into fail_hold mid-rise.
    MAX_STATE_STEPS = 400

    def __init__(self):
        self.state = 0
        self.spray_counter = 0
        self.state_steps = 0
        self.completed = False
        self.timed_out = False
        self.approach_waypoint = None
        self.fail_pos = None
        self.spray_anchor = None  # EE pos captured at first spray frame; held through hold
        self.frozen_joint_pos = None  # full joint state captured at first spray frame; teleported each frame
        self.wrist_lerp_start = None  # wrist_pitch captured at first settle frame; smooth lerp source

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

        # Orientation is handled outside the FSM by the wrist_pitch joint-space
        # override (lerp during settle, locked during spray/hold). Leave
        # action[3:6]=0 always so IK only solves position and doesn't fight
        # the locked wrist.

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
            if self.state_steps >= 180:  # matches WRIST_LERP_STEPS — settle ends when wrist is fully down
                self.spray_counter = SPRAY_DURATION
                self._advance(3)

        elif self.state == 3:
            # Anchor on first spray frame: freeze EE wherever it ended up after
            # settle. Don't keep targeting hover_target — that drags the EE
            # back and (with the wrist locked) can pull it down further.
            if self.spray_anchor is None:
                self.spray_anchor = np.copy(ee_pos)
            err_to_anchor = self.spray_anchor - ee_pos
            action[0:3] = self._position_command(err_to_anchor)
            self.spray_counter -= 1
            if self.spray_counter <= 0:
                self.completed = True
                self._advance(4)

        elif self.state == 4:
            if self.spray_anchor is None:
                self.spray_anchor = np.copy(ee_pos)
            err_to_anchor = self.spray_anchor - ee_pos
            action[0:3] = self._position_command(err_to_anchor)

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
    # Extended from 7.5 s → 12.0 s so the success_hold tail is long
    # enough for the wrist to finish rotating to DOWN_QUAT and stay
    # there before the next reset. Net cycle (approach → spray → hold)
    # was ~95 steps; with 12 s @ 30 Hz = 360 steps, success_hold gets
    # ~265 steps of "just hovering and pointing down".
    env_cfg.episode_length_s = 22.0   # was 12.0 — extended to fit slower POSITION_GAIN/ACTION_CLAMP

    # Keep the column above the base from drifting left/right while the
    # IK reaches for the hover target — base_yaw is locked at REST_POSE
    # so the entire base+column stays rigidly aimed at the yellow
    # face-target. Shoulder / elbow / wrist still drive the EE.
    try:
        env_cfg.actions.arm_action.joint_names = [
            "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll",
        ]
        # Match the IK target body to what the FSM tracks (gripper TIP, not the
        # gripper body upstream of it). Eliminates the ~3cm tip-vs-body offset
        # that made the gripper appear "past" the crown after IK convergence.
        env_cfg.actions.arm_action.body_name = "moving_gripper"
    except Exception as e:
        print(f"[setup] couldn't restrict IK joint set: {e}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    sim_device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    palm_root_paths = [get_palm_root_path(env_id) for env_id in range(num_envs)]
    for palm_path in palm_root_paths:
        disable_palm_physics(stage, palm_path)
    episode_rng = np.random.default_rng(getattr(args_cli, "seed", None))

    randomize_robot_root_pose(
        env=env, stage=stage, palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
    )

    current_hover_targets = np.zeros((num_envs, 3), dtype=np.float32)
    robot_xys_now = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].cpu().numpy()
    current_hover_targets[:] = prepare_episode_targets(
        stage=stage,
        palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
        cull_prob=args_cli.top_leaf_cull_prob,
        robot_xys=robot_xys_now,
    )

    oracles = [SprayOracle() for _ in range(num_envs)]
    spawn_target_markers(stage, current_hover_targets, marker_type="hover", color=(0.0, 0.0, 1.0))

    moving_gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper")
    moving_gripper_idx = moving_gripper_indices[0]

    # Resolve wrist_pitch joint index for the joint-space "snap down" override
    # used while the FSM is in settle/spray/success_hold (states 2/3/4). IK
    # alone can't drive wrist_pitch to its negative limit with 4 DOF, so we
    # bypass it and write the joint angle directly during those states.
    wrist_pitch_idx = env.unwrapped.scene["robot"].joint_names.index("wrist_pitch")
    WRIST_PITCH_REST = REST_POSE_VALUES[3]   # +1.2 rad — wrist up at rest
    WRIST_PITCH_DOWN = -1.5708               # lower limit — EE points straight down
    WRIST_LERP_STEPS = 180                   # frames spent rotating the wrist (≈6s @ 30Hz; slower = less EE drift)
    SNAP_STATES = {2, 3, 4}

    prev_states = [oracle.state for oracle in oracles]
    leaf_stuck_steps = [0] * num_envs
    step_counter = 0
    state_names = {0: "approach_wp", 1: "approach_hover", 2: "settle",
                   3: "spray", 4: "success_hold", 5: "fail_hold"}

    while simulation_app.is_running():
        robot_data = env.unwrapped.scene["robot"].data
        ee_pos_all = robot_data.body_pos_w[:, moving_gripper_idx].cpu().numpy()
        ee_quat_all = robot_data.body_quat_w[:, moving_gripper_idx].cpu().numpy()
        root_quat_all = robot_data.root_quat_w.cpu().numpy()

        action_batch = np.zeros((num_envs, 7), dtype=np.float32)
        for env_id in range(num_envs):
            env_action = oracles[env_id].compute_action(
                ee_pos_all[env_id], ee_quat_all[env_id],
                hover_target=current_hover_targets[env_id],
            )
            # IK relative-mode interprets BOTH the position delta and the
            # axis-angle orientation delta in body (root) frame; FSM
            # produces both in world frame. Rotate world→body using
            # R_z(-base_yaw). At yaw=0 this is the identity so original
            # un-randomized recordings reproduce exactly.
            base_yaw = _yaw_from_quat_wxyz(root_quat_all[env_id])
            env_action[0:3] = _rotate_vec_z(env_action[0:3], -base_yaw)
            env_action[3:6] = _rotate_vec_z(env_action[3:6], -base_yaw)
            env_action[:3] = np.clip(env_action[:3], -ACTION_CLAMP, ACTION_CLAMP)
            action_batch[env_id] = env_action

            if DEBUG_VERBOSE and oracles[env_id].state != prev_states[env_id]:
                hover = current_hover_targets[env_id]
                dist = float(np.linalg.norm(ee_pos_all[env_id] - hover))
                old = state_names.get(prev_states[env_id], str(prev_states[env_id]))
                new = state_names.get(oracles[env_id].state,
                                      str(oracles[env_id].state))
                print(f"[env {env_id} step {step_counter:5d}] "
                      f"state {old} -> {new}  dist_to_hover={dist:.3f} m  "
                      f"yaw={np.rad2deg(base_yaw):+7.2f}deg",
                      flush=True)
                prev_states[env_id] = oracles[env_id].state

        step_counter += 1

        action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=sim_device)

        _, _, terminated, truncated, _ = env.step(action_tensor)

        # Joint-space override:
        #   State 2 (settle):  lerp wrist_pitch from rest → WRIST_PITCH_DOWN
        #                      while IK still controls the other joints so the
        #                      EE stays in hover.
        #   State 3/4 (spray/hold): freeze the ENTIRE arm at whatever pose it
        #                           landed in at the end of settle. No more IK
        #                           drift — every joint is teleported back to
        #                           its captured value every frame.
        snap_env_ids = [i for i, o in enumerate(oracles) if o.state in SNAP_STATES]
        if snap_env_ids:
            robot_asset = env.unwrapped.scene["robot"]
            joint_pos = robot_asset.data.joint_pos.clone()
            joint_vel = robot_asset.data.joint_vel.clone()
            for env_id in snap_env_ids:
                oracle = oracles[env_id]
                if oracle.state == 2:  # settle — wrist lerp only
                    # Capture wrist's actual value at the START of settle so
                    # the lerp begins where the IK left it — no teleport jump.
                    if oracle.wrist_lerp_start is None:
                        oracle.wrist_lerp_start = float(joint_pos[env_id, wrist_pitch_idx].item())
                    progress = min(1.0, oracle.state_steps / WRIST_LERP_STEPS)
                    target = oracle.wrist_lerp_start + (WRIST_PITCH_DOWN - oracle.wrist_lerp_start) * progress
                    joint_pos[env_id, wrist_pitch_idx] = target
                    joint_vel[env_id, wrist_pitch_idx] = 0.0
                else:  # spray, success_hold — freeze whole arm
                    if oracle.frozen_joint_pos is None:
                        oracle.frozen_joint_pos = joint_pos[env_id].clone()
                        oracle.frozen_joint_pos[wrist_pitch_idx] = WRIST_PITCH_DOWN
                    joint_pos[env_id] = oracle.frozen_joint_pos
                    joint_vel[env_id] = 0.0
            snap_ids_t = torch.as_tensor(snap_env_ids, device=sim_device, dtype=torch.long)
            robot_asset.write_joint_state_to_sim(
                joint_pos[snap_ids_t], joint_vel[snap_ids_t], env_ids=snap_ids_t,
            )

        terminated_t = torch.as_tensor(terminated, device=sim_device, dtype=torch.bool)
        truncated_t = torch.as_tensor(truncated, device=sim_device, dtype=torch.bool)
        done_mask = torch.logical_or(terminated_t, truncated_t).reshape(-1)
        done_env_ids_t = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
        done_env_ids = done_env_ids_t.cpu().tolist() if done_env_ids_t.numel() > 0 else []

        # Re-read EE positions (post-step) to detect leaf-stuck cases —
        # only counted when the FSM has already entered fail_hold, so a
        # normal approach trajectory through the canopy doesn't trigger
        # a premature reset.
        ee_pos_post = robot_data.body_pos_w[:, moving_gripper_idx].cpu().numpy()
        stuck_env_ids = []
        for env_id in range(num_envs):
            if env_id in done_env_ids:
                leaf_stuck_steps[env_id] = 0
                continue
            if oracles[env_id].state != LEAF_STUCK_FAIL_STATE:
                leaf_stuck_steps[env_id] = 0
                continue
            d = _closest_leaf_dist_3d(stage, palm_root_paths[env_id], ee_pos_post[env_id])
            if d is not None and d < LEAF_STUCK_DISTANCE:
                leaf_stuck_steps[env_id] += 1
            else:
                leaf_stuck_steps[env_id] = 0
            if leaf_stuck_steps[env_id] >= LEAF_STUCK_STEPS:
                stuck_env_ids.append(env_id)
                if DEBUG_VERBOSE:
                    print(f"[env {env_id} step {step_counter:5d}] "
                          f"LEAF-STUCK in fail_hold (closest leaf {d:.3f} m for "
                          f"{leaf_stuck_steps[env_id]} steps) → forcing reset",
                          flush=True)

        reset_env_ids = sorted(set(done_env_ids) | set(stuck_env_ids))

        if reset_env_ids:
            if DEBUG_VERBOSE:
                reasons = []
                for env_id in reset_env_ids:
                    if env_id in stuck_env_ids and env_id not in done_env_ids:
                        reasons.append(f"env {env_id}: leaf-stuck")
                    elif env_id in done_env_ids:
                        reasons.append(f"env {env_id}: done")
                print(f"[step {step_counter:5d}] resetting "
                      f"({', '.join(reasons)})", flush=True)
            randomize_robot_root_pose(
                env=env, stage=stage, palm_root_paths=palm_root_paths,
                episode_rng=episode_rng, env_ids=reset_env_ids,
            )
            robot_xys_now = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].cpu().numpy()
            refreshed_targets = prepare_episode_targets(
                stage=stage,
                palm_root_paths=palm_root_paths,
                episode_rng=episode_rng,
                cull_prob=args_cli.top_leaf_cull_prob,
                robot_xys=robot_xys_now,
                env_ids=reset_env_ids,
            )
            for i, env_id in enumerate(reset_env_ids):
                current_hover_targets[env_id] = refreshed_targets[i]
                oracles[env_id] = SprayOracle()
                prev_states[env_id] = 0
                leaf_stuck_steps[env_id] = 0
            spawn_target_markers(
                stage, current_hover_targets,
                env_ids=reset_env_ids, marker_type="hover", color=(0.0, 0.0, 1.0),
            )

    # Shutdown workaround for the known omni.syntheticdata crash on
    # Py_FinalizeEx (visible in the carb crashreporter trace as
    # releaseFrameworkAndTerminate → carbOnPluginShutdown →
    # omni.syntheticdata.plugin.dll). The plugin doesn't reliably tear
    # down with --enable_cameras, so we close the env (releases sensor
    # buffers) and then exit the process hard with os._exit(0). This
    # skips Python finalization entirely, so the broken plugin shutdown
    # path is never invoked. We don't need an orderly Python exit here:
    # all per-episode data is already on disk (or in the OBS capture).
    try:
        env.close()
    except Exception:
        pass
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
