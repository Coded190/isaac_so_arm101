# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""VLA data-generation entry point for Isaac Lab.

Runs the SprayOracle FSM and palm-spray simulation, optionally writing per-env
LeRobot datasets when ``--save_data`` is passed. Inherits all the kinematic
improvements from the recording-branch variant (4-DOF IK on the gripper tip,
randomized base placement, slow-motion approach, wrist-pitch joint-space
override during settle/spray, hover-target pull-back from the crown). The
expert-action labels recorded in the dataset still include an axis-angle
delta toward DOWN_QUAT so a VLA policy can learn the orientation command —
the wrist override only shapes the sim execution, not the recorded action.

Typical launch:
    uv run vla_data_gen_v2 \
        --task Isaac-PING-TI-VLA-v0 --num_envs 10 \\
        --enable_cameras --save_data
"""

import argparse
import os
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

parser = argparse.ArgumentParser(description="VLA palm-spray data generation for Isaac Lab.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate, default is 1.")
parser.add_argument("--task", type=str, default="None", help="Name of the task.")
parser.add_argument("--save_data", action="store_true", help="Enable saving data with LeRobotDataset. If not set, data is not saved.")
parser.add_argument("--top_leaf_cull_prob", type=float, default=0.5, help="Probability of culling the top leaves for a given episode.")
parser.add_argument("--dataset_root", type=str, default="outputs/vla_palm_dataset", help="Root folder where per-env LeRobot datasets are stored.")

AppLauncher.add_app_launcher_args(parser)
# 1. Parse standard arguments normally
args_cli, _ = parser.parse_known_args()

# 2. INJECT the PhysX limits directly into the system arguments BEFORE boot
sys.argv.append("--/persistent/omni/physx/persistentErrorMaxCount=10000000")
sys.argv.append("--/persistent/omni/physx/rejectUnsupportedActors=false")

# 3. Boot the application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
# Just the global level. Per-channel hammering was triggering
# "getStringRawInternal: item ... is not a string" errors and timing
# out the viewport. Console noise is filtered by the Python stderr
# wrapper at the top of this file plus 2>/dev/null in the launch command.
carb.settings.get_settings().set_string("/log/level", "error")

# (Note: The persistentErrorMaxCount and rejectUnsupportedActors have been 
# removed from here, as they are now handled by sys.argv above)

carb.settings.get_settings().set_int("/physics/numThreads", 0)  # default
# Suppress the cosmetic palm-joint errors entirely from the PhysX channel:
carb.settings.get_settings().set_string("/log/channels/omni.physx.plugin", "fatal")
carb.settings.get_settings().set_string("/log/channels/omni.kit.notification_manager.manager", "fatal")

import cv2
import torch
import numpy as np
import gymnasium as gym
from PIL import Image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    print("[ERROR]: 'lerobot' package not found. Please install it in your Isaac Lab environment using 'pip install lerobot'.")
    simulation_app.close()
    exit(1)

import isaac_so_arm101.tasks
from isaaclab_tasks.utils import parse_env_cfg


# ─── Kinematic / FSM constants ────────────────────────────────────────────
ACTION_CLAMP = 0.25      # recording-style slow motion (was 0.5; smoother per-step deltas)
POSITION_GAIN = 0.35     # recording-style slow motion (was 0.75)
SPRAY_DURATION = 60

REST_POSE_VALUES = [
    0.0,                       # base_yaw
    float(np.deg2rad(-45.0)),  # shoulder_pitch  — chain sums to -90° world pitch:
    float(np.deg2rad(-45.0)),  # elbow_pitch     — gripper points straight down at rest
    -0.2,                      # wrist_pitch     — slight back-tilt (~11°) toward observer
    0.0,                       # wrist_roll
    0.0,                       # gripper_moving
]

# Pose the arm transitions to during settle (state 2) and holds during
# spray/hold (states 3/4). Shoulder pulled further back, elbow folded down,
# wrist at lower limit so the gripper looks straight down at the crown.
SPRAY_POSE_VALUES = [
    0.0,                       # base_yaw (locked)
    float(np.deg2rad(-70.0)),  # shoulder_pitch  — pulled back hard
    float(np.deg2rad(-78.0)),  # elbow_pitch     — folded down
    -1.5708,                   # wrist_pitch     — at URDF lower limit (looks down)
    0.0,                       # wrist_roll
    0.0,                       # gripper_moving
]

LEAF_CULL_Z_OFFSET = 0.03
# When an episode culls leaves, the keep-ratio is sampled uniformly from
# [LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX]. Lower keep-ratio = more leaves
# removed: 0.5 culls 50% of the top leaves, 0.8 culls 20%.
LEAF_KEEP_RATIO_MIN = 0.5
LEAF_KEEP_RATIO_MAX = 0.8
HOVER_OFFSET = np.array([0.0, 0.0, 0.35])  # back to original reachable height
# Meters to pull the hover target from the crown XY back toward the robot XY.
# 0.0 = directly above crown (used to overshoot past). Increase to land short of crown.
HOVER_PULLBACK_M = 0.0

# ─── Dataset metadata ─────────────────────────────────────────────────────
TASK_DESCRIPTION = "Move end effector above palm crown, angle end effector downward, and hold while end effector is spraying."
MAX_TOTAL_SAVED_FRAMES = 25000  # cap across all envs; script exits after

DOWN_QUAT = np.array([0.2588, 0.0, -0.9659, 0.0])
# Bumped 0.2 → 0.5 so the wrist rotates toward DOWN_QUAT ~2.5× faster
# per step. With ANGLE_RANDOM_RANGE shrunk to ±45° the IK has easy
# reach, so the orientation command no longer needs to be conservative.
ORIENTATION_CLAMP = 1.0   # aggressive wrist-down correction (only fires in spray/hold; EE already parked)

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
TREE_INWARD_OFFSET = 0.50  # positive = pull robot CLOSER to tree (reachable distance)
# Lower bound on the radius — never go closer than this to the trunk.
MIN_TREE_RADIUS = 0.08
# Reject any placement whose XY position lands within this many meters of
# any palm leaf — the base would otherwise spawn inside / through a leaf.
LEAF_CLEARANCE = 0.10
PLACEMENT_MAX_ATTEMPTS = 15
PALM_ROOT_NAME = "palm_tree_crown"


def get_palm_root_path(env_id):
    return f"/World/envs/env_{env_id}/Scene/{PALM_ROOT_NAME}"


def _get_palm_crown_prim(stage, palm_root_path):
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm or not palm.IsValid():
        return None
    crown = palm.GetChild("crown")
    if crown and crown.IsValid():
        return crown
    return palm

def print_joint_properties(joint_prim):
    """Prints all authored properties of a USD Joint to the console."""
    print(f"\n[DEBUG] --- Properties for Joint: {joint_prim.GetName()} ---")
    
    # Iterate through every property that has an explicitly assigned value
    for prop in joint_prim.GetAuthoredProperties():
        name = prop.GetName()
        val = prop.Get()
        
        # Highlight properties related to physics limits or drives
        if "physics" in name or "limit" in name or "drive" in name:
            print(f"  -> {name}: {val}")
        else:
            print(f"     {name}: {val}")
    print("---------------------------------------------------\n")

from pxr import UsdLux, Sdf, Gf
import random


def _dump_dome_light_state(light, header):
    """Print a compact dump of DomeLight attrs relevant to HDRI rendering."""
    path = light.GetPrim().GetPath()
    print(f"[dome-light] {header} {path}", flush=True)
    try:
        authored_props = list(light.GetPrim().GetAuthoredProperties())
        print(f"[dome-light]   authored_properties={len(authored_props)}", flush=True)
        for prop in authored_props:
            prop_name = prop.GetName()
            try:
                prop_value = prop.Get()
            except Exception as exc:
                prop_value = f"<error {exc}>"
            print(f"[dome-light]     {prop_name}={prop_value}", flush=True)
    except Exception as exc:
        print(f"[dome-light]   authored_properties=<error {exc}>", flush=True)
    try:
        print(f"[dome-light]   textureFile={light.GetTextureFileAttr().Get()}", flush=True)
    except Exception as exc:
        print(f"[dome-light]   textureFile=<error {exc}>", flush=True)
    try:
        print(f"[dome-light]   intensity={light.GetIntensityAttr().Get()}", flush=True)
    except Exception as exc:
        print(f"[dome-light]   intensity=<error {exc}>", flush=True)
    try:
        exposure_attr = light.GetExposureAttr()
        if exposure_attr:
            print(f"[dome-light]   exposure={exposure_attr.Get()}", flush=True)
    except Exception as exc:
        print(f"[dome-light]   exposure=<error {exc}>", flush=True)
    try:
        texture_format_attr = light.GetTextureFormatAttr()
        if texture_format_attr:
            print(f"[dome-light]   textureFormat={texture_format_attr.Get()}", flush=True)
    except Exception as exc:
        print(f"[dome-light]   textureFormat=<error {exc}>", flush=True)
    try:
        texture_value = light.GetTextureFileAttr().Get()
        texture_text = str(texture_value).strip("@") if texture_value else ""
        stage_root = light.GetPrim().GetStage().GetRootLayer()
        root_dir = os.path.dirname(stage_root.realPath) if stage_root and stage_root.realPath else os.getcwd()
        resolved_from_root = os.path.abspath(os.path.join(root_dir, texture_text))
        resolved_from_cwd = os.path.abspath(os.path.join(os.getcwd(), texture_text))
        print(f"[dome-light]   cwd={os.getcwd()}", flush=True)
        print(f"[dome-light]   stage_root={stage_root.realPath if stage_root else None}", flush=True)
        print(f"[dome-light]   resolved_from_root={resolved_from_root}", flush=True)
        print(f"[dome-light]   resolved_from_cwd={resolved_from_cwd}", flush=True)
        if os.path.isabs(texture_text):
            print(f"[dome-light]   absolute_path_exists={os.path.exists(texture_text)}", flush=True)
        else:
            print(f"[dome-light]   from_root_exists={os.path.exists(resolved_from_root)}", flush=True)
            print(f"[dome-light]   from_cwd_exists={os.path.exists(resolved_from_cwd)}", flush=True)
    except Exception as exc:
        print(f"[dome-light]   resolved_paths=<error {exc}>", flush=True)

def randomize_lighting(stage, hdri_folder_path, env_ids=None):
    """Assign one random HDRI to every env-local DomeLight in the stage.
    
    Tuning approach: Keeps the background image rich, keeps specular reflections
    muted to prevent glare, but boosts diffuse ambient light so shadows on 3D models 
    are properly filled and visible. Includes a single global shadowless ambient 
    fill floor to prevent USD instancing breaks (which causes scaled assets to enlarge).
    """
    # Verify the HDRI folder exists and has files
    if not os.path.exists(hdri_folder_path):
        print(f"[WARNING] HDRI folder not found: {hdri_folder_path}", flush=True)
        return
    
    # Get a list of all HDR/EXR files in your folder
    valid_exts = (".hdr", ".exr")
    hdri_files = [f for f in os.listdir(hdri_folder_path) if f.endswith(valid_exts)]
    if not hdri_files:
        print(f"[WARNING] No HDRI files (.hdr/.exr) found in {hdri_folder_path}", flush=True)
        return

    chosen_hdri = random.choice(hdri_files)
    full_path = os.path.join(hdri_folder_path, chosen_hdri)
    
    # Verify the chosen HDRI file exists
    if not os.path.exists(full_path):
        print(f"[WARNING] HDRI file not found: {full_path}", flush=True)
        return
    
    # 1. Base intensity for a rich, vibrant background sky texture
    target_intensity = random.uniform(600.0, 1200.0)
    
    if DEBUG_VERBOSE:
        print(f"[randomize_lighting] chosen_hdri={chosen_hdri} full_path={full_path}", flush=True)

    if env_ids is None:
        env_ids = []
        envs_root = stage.GetPrimAtPath("/World/envs")
        if envs_root:
            for env_prim in envs_root.GetChildren():
                name = env_prim.GetName()
                if name.startswith("env_"):
                    suffix = name.split("env_", 1)[1]
                    if suffix.isdigit():
                        env_ids.append(int(suffix))

    # Collect all valid environment lights first to get an accurate count
    valid_env_lights = []
    for env_id in env_ids:
        light_path = f"/World/envs/env_{env_id}/Scene/DomeLight"
        prim = stage.GetPrimAtPath(light_path)
        if prim and prim.IsA(UsdLux.DomeLight):
            valid_env_lights.append((env_id, prim))
        elif DEBUG_VERBOSE:
            print(f"[WARNING] DomeLight not found at {light_path}", flush=True)

    if not valid_env_lights:
        print("[WARNING] No env-local DomeLight prims were found to update.", flush=True)
        return

    # 2. Calculate base fraction per environment
    num_envs = len(valid_env_lights)
    fractional_multiplier = 1.0 / num_envs

    # --- TUNING KNOBS FOR THE PALM TREE ---
    specular_value = fractional_multiplier * 1.0
    diffuse_boost = 10.0 
    diffuse_value = fractional_multiplier * diffuse_boost

    if DEBUG_VERBOSE:
        print(f"[dome-light] Intensity: {target_intensity} | Diffuse Val: {diffuse_value:.4f} | Specular Val: {specular_value:.4f}", flush=True)

    for env_id, prim in valid_env_lights:
        light = UsdLux.DomeLight(prim)
        
        # Pass the HDRI texture path
        light.GetTextureFileAttr().Set(full_path)
        
        # Keep intensity high so the background texture is fully visible and rich
        light.GetIntensityAttr().Set(target_intensity)
        
        # --- APPLY THE TUNED VALUES ---
        diffuse_attr = light.GetDiffuseAttr()
        if diffuse_attr:
            diffuse_attr.Set(diffuse_value)
        else:
            light.CreateDiffuseAttr(diffuse_value)
            
        specular_attr = light.GetSpecularAttr()
        if specular_attr:
            specular_attr.Set(specular_value)
        else:
            light.CreateSpecularAttr(specular_value)
        
        # Force latlong format to prevent viewport blackouts
        format_attr = light.GetTextureFormatAttr()
        if format_attr:
            format_attr.Set("latlong")
        else:
            light.CreateTextureFormatAttr("latlong")
            
        # Ensure exposure offset is normalized
        exposure_attr = light.GetExposureAttr()
        if exposure_attr:
            exposure_attr.Set(0.0)
        else:
            light.CreateExposureAttr(0.0)

    # --- FIX: ONE GLOBAL AMBIENT SAFETY NET FLOOR ---
    # Creates ONE shadowless light outside of the clones to preserve instancing.
    global_ambient_path = "/World/GlobalAmbientFill"
    ambient_prim = stage.GetPrimAtPath(global_ambient_path)
    if not ambient_prim:
        ambient_light = UsdLux.DistantLight.Define(stage, global_ambient_path)
    else:
        ambient_light = UsdLux.DistantLight(ambient_prim)
        
    # Set the baseline brightness. Since this is one global light (not multiplied 
    # by env count), we set the raw total intensity here. 
    ambient_light.GetIntensityAttr().Set(500.0)
    ambient_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    
    # Explicitly turn off shadows for this baseline fill light
    shadow_attr = ambient_light.GetPrim().GetAttribute("inputs:shadow:enable")
    if not shadow_attr:
        shadow_attr = ambient_light.GetPrim().CreateAttribute("inputs:shadow:enable", Sdf.ValueTypeNames.Bool)
    shadow_attr.Set(False)


def disable_palm_physics(stage, palm_root_path):
    """Updates palm tree leaves to be lightweight, and completely loosens their joints."""
    from pxr import Usd, UsdGeom, UsdPhysics
    
    palm = stage.GetPrimAtPath(palm_root_path)
    if not palm:
        return

    crown = _get_palm_crown_prim(stage, palm_root_path)
    crown_paths = set()
    if crown:
        for prim in Usd.PrimRange(crown):
            crown_paths.add(prim.GetPath())

    # Use Usd.PrimRange to recursively traverse the updated palm subtree.
    for child in Usd.PrimRange(palm):
        if child == palm:
            continue

        prim_name = child.GetName().lower()
        prim_path = child.GetPath()
        
        # 1. MAKE LEAVES LIGHTWEIGHT AND DYNAMIC
        if crown and prim_path in crown_paths and child.IsA(UsdGeom.Mesh) and (
            prim_name.startswith("leaf_") or prim_name.startswith("leaf_b_")
        ):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(child)
            else:
                rb_api = UsdPhysics.RigidBodyAPI.Apply(child)
                
            kin_attr = rb_api.GetKinematicEnabledAttr()
            if kin_attr:
                kin_attr.Set(False) 
            else:
                rb_api.CreateKinematicEnabledAttr(False)

            if child.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(child)
            else:
                mass_api = UsdPhysics.MassAPI.Apply(child)
            
            mass_attr = mass_api.GetMassAttr()
            if mass_attr:
                mass_attr.Set(0.05)
            else:
                mass_api.CreateMassAttr(0.05)
                
            if child.HasAPI(UsdPhysics.CollisionAPI):
                col_api = UsdPhysics.CollisionAPI(child)
                col_attr = col_api.GetCollisionEnabledAttr()
                if col_attr:
                    col_attr.Set(True)
                else:
                    col_api.CreateCollisionEnabledAttr(True)

        # 2. KEEP THE TRUNK IMMOVABLE
        elif prim_name in {"trunk", "trunk_top"}:
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                rb_api = UsdPhysics.RigidBodyAPI(child)
                kin_attr = rb_api.GetKinematicEnabledAttr()
                if kin_attr:
                    kin_attr.Set(True)
                else:
                    rb_api.CreateKinematicEnabledAttr(True)

        # 3. THE FIX: RELIABLY FIND AND NEUTRALIZE JOINTS
        # Check by actual USD Type instead of relying on the string name
        if child.IsA(UsdPhysics.Joint): 
            
            # if not hasattr(disable_palm_physics, "has_printed"):
            #     print_joint_properties(child)
            #     disable_palm_physics.has_printed = True

            for prop in child.GetAuthoredProperties():
                prop_name = prop.GetName().lower()
                
                # 1. Kill the spring forces so the leaf doesn't fight back
                if "stiffness" in prop_name:
                    prop.Set(0.5)
                    
                # 2. Grease the hinge (leave a tiny amount of damping so it doesn't vibrate infinitely)
                elif "damping" in prop_name:
                    prop.Set(0.1)
                
                # We completely leave the limits alone! 
                # 45 degrees of limp bending is more than enough for the arm to pass.

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
    """Activate or deactivate every leaf under palm_root_path using Visibility."""
    from pxr import UsdGeom
    for child in _iter_leaf_prims(stage, palm_root_path):
        if UsdGeom.Xformable(child):
            # Use Imageable to toggle visibility safely without breaking PhysX
            imageable = UsdGeom.Imageable(child)
            if active:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()


def get_crown_centroid(stage, palm_root_path):
    """Get the mean world position of all active leaves (the crown centroid)."""
    leaves = _leaf_world_positions(stage, palm_root_path)
    if not leaves:
        return np.array([0.0, 0.0, 5.0])
    positions = np.stack([p for _, p in leaves], axis=0)
    return positions.mean(axis=0)


def remove_top_leaves(stage, palm_root_path, crown_z, keep_ratio,
                      z_threshold_offset=LEAF_CULL_Z_OFFSET):
    from pxr import UsdGeom
    cull_z = crown_z + z_threshold_offset
    leaves = _leaf_world_positions(stage, palm_root_path)
    
    top_leaves = [(prim, pos[2]) for prim, pos in leaves if pos[2] > cull_z]
    top_leaves.sort(key=lambda x: -x[1])
    n_remove = int(len(top_leaves) * (1.0 - keep_ratio))
    
    for prim, _ in top_leaves[:n_remove]:
        # Safely hide the culled leaves without destroying their physics representations
        UsdGeom.Imageable(prim).MakeInvisible()


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


def cull_episode_leaves(stage, palm_root_paths, episode_rng, cull_prob, env_ids=None):
    """Cull top leaves for each environment based on cull_prob.
    
    Should be called BEFORE positioning the robot so that leaf clearance checks
    and robot positioning are based on the final (culled) leaf geometry.
    """
    if env_ids is None:
        env_ids = list(range(len(palm_root_paths)))
    for env_id in env_ids:
        palm_root_path = palm_root_paths[env_id]
        # Re-activate all leaves first in case they were culled in a previous episode
        set_leaf_prims_active(stage, palm_root_path, active=True)
        if episode_rng.random() < cull_prob:
            crown_centroid = get_crown_centroid(stage, palm_root_path)
            keep_ratio = float(episode_rng.uniform(
                LEAF_KEEP_RATIO_MIN, LEAF_KEEP_RATIO_MAX,
            ))
            remove_top_leaves(
                stage, palm_root_path,
                crown_z=crown_centroid[2],
                keep_ratio=keep_ratio,
            )


def prepare_episode_targets(stage, palm_root_paths, robot_xys=None, env_ids=None):
    """Compute hover targets per env based on crown centroid + HOVER_OFFSET.
    
    If robot_xys is provided, targets can be pulled toward the robot (HOVER_PULLBACK_M).
    Assumes cull_episode_leaves() was already called.
    """
    if env_ids is None:
        env_ids = list(range(len(palm_root_paths)))
    targets = []
    for env_id in env_ids:
        palm_root_path = palm_root_paths[env_id]
        crown_centroid = get_crown_centroid(stage, palm_root_path)
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
        if DEBUG_VERBOSE:
            robot_xy = robot_xys[env_id] if robot_xys is not None else None
            print(f"[prepare_targets env {env_id}] leaf_mean={crown_centroid} target={target} robot_xy={robot_xy}", flush=True)
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
    circle around the trunk. Uses the crown-centroid XYZ (same as the
    blue hover marker) for the target position. The XY distance to the trunk
    is preserved (optionally shrunk by ``inward_offset``) and the robot base
    Z is aligned to the crown's Z position. Yaw is set so the robot faces
    the tree from its new position. A yellow marker is dropped at the target
    so you can see what the base is aimed at.
    """
    robot = env.unwrapped.scene["robot"]
    device = robot.data.default_root_state.device
    if env_ids is None:
        env_ids = list(range(env.unwrapped.num_envs))
    env_ids_t = torch.as_tensor(env_ids, device=device, dtype=torch.long)

    new_root = robot.data.default_root_state[env_ids_t].clone()
    # Get CURRENT robot position (world frame) to calculate initial radius, not default_root_state
    current_root_pos_w = robot.data.root_pos_w[env_ids_t].cpu().numpy().astype(np.float64)
    
    # default_root_state is in env-local frame; trunk_xy from get_crown_centroid
    # is world-frame. With multi-env (env_spacing > 0), env origins are offset
    # in world, so we must add the env origin to default_xy to compare apples
    # to apples. Single-env case has env_origin=(0,0,0) and is unchanged.
    env_origins = env.unwrapped.scene.env_origins[env_ids_t, :2].cpu().numpy().astype(np.float64)
    trunk_xys = []
    trunk_zs = []
    for i, env_id in enumerate(env_ids):
        palm_root_path = palm_root_paths[env_id]
        crown = get_crown_centroid(stage, palm_root_path)
        trunk_xy = np.array([float(crown[0]), float(crown[1])], dtype=np.float64)
        trunk_z = float(crown[2])
        trunk_xys.append(trunk_xy)
        trunk_zs.append(trunk_z)
        
        # Use current robot position (world frame) for radius calculation
        current_xy = current_root_pos_w[i, :2]
        rel = current_xy - trunk_xy
        radius0 = float(np.linalg.norm(rel))
        bearing0 = float(np.arctan2(rel[1], rel[0]))
        
        if DEBUG_VERBOSE:
            radius_after_offset = max(radius0 - float(inward_offset), MIN_TREE_RADIUS)
            print(f"[randomize env {env_id}] current_pos={current_xy} trunk={trunk_xy} "
                  f"radius0={radius0:.3f}m radius_after_offset={radius_after_offset:.3f}m", 
                  flush=True)

        new_x = new_y = None
        last_dist = None
        cand_x = cand_y = 0.0  # Initialize before loop
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
        new_root[i, 2] = trunk_zs[i]
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

    # Drop a yellow marker at (trunk_xy, trunk_z) so the user can verify the
    # base is being aimed at the right horizontal target and Z level.
    # Also re-aim the active viewport camera at the first reset env so OBS
    # captures a fresh side-view of the robot+crown each episode.
    for i, env_id in enumerate(env_ids):
        trunk_xy = trunk_xys[i]
        trunk_z = trunk_zs[i]
        base_z = float(new_root[i, 2].cpu())
        marker_xyz = (float(trunk_xy[0]), float(trunk_xy[1]), trunk_z)
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
    POSITION_THRESHOLD = 0.20   # tighter convergence (recording-style)
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
        self.spray_pose_start = None  # [shoulder, elbow, wrist] captured at first settle frame for the spray-pose lerp
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

        # No orientation command — action[3:6] stays 0. The chosen rest pose
        # already has the gripper roughly pointing down (shoulder + elbow
        # chain sums to ~-90°), and any late IK rotation just makes the arm
        # shift. Pure position-only IK keeps the trajectory clean.

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
            if self.state_steps >= 90:   # matches WRIST_LERP_STEPS (3× slower settle)
                self.spray_counter = SPRAY_DURATION
                self._advance(3)

        elif self.state == 3:
            # Keep tracking hover_target during spray so IK can finish
            # correcting the geometric drop that settle's wrist rotation
            # introduced. Otherwise spray "anchors" the EE below the crown.
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

    # Set a neutral initial position; randomize_robot_root_pose() will
    # immediately teleport the robot to the correct location near palm_tree_crown.
    # The Y offset keeps the robot at the center of the multi-env grid.
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
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
    env_cfg.episode_length_s = 22.0   # recording-style episode length (fits slower motion)

    # Keep the column above the base from drifting left/right while the
    # IK reaches for the hover target — base_yaw is locked at REST_POSE
    # so the entire base+column stays rigidly aimed at the yellow
    # face-target. Shoulder / elbow / wrist still drive the EE.
    try:
        # 4-DOF IK (base_yaw locked at rest). Orientation isn't commanded
        # anywhere in the FSM (action[3:6]=0 always), so we don't need
        # base_yaw — and removing it eliminates the column twist + the
        # IK-vs-clamp fight that was causing approach_hover drift.
        env_cfg.actions.arm_action.joint_names = [
            "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll",
        ]
        # IK target body = the gripper TIP (matches what the FSM tracks).
        env_cfg.actions.arm_action.body_name = "moving_gripper"
    except Exception as e:
        print(f"[setup] couldn't configure arm action: {e}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    sim_device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    import omni.usd
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()
    palm_root_paths = [get_palm_root_path(env_id) for env_id in range(num_envs)]
    
    # Get the crown centroid position for the first env to initialize robot nearby
    if palm_root_paths:
        crown_pos = get_crown_centroid(stage, palm_root_paths[0])
        print(f"[init] palm_tree_crown centroid at: {crown_pos}", flush=True)
        
        # Pre-position the robot near the crown before randomize_robot_root_pose()
        # This ensures the radius0 calculation starts from a sensible distance
        robot = env.unwrapped.scene["robot"]
        new_root = robot.data.default_root_state.clone()
        
        # Place robot at a default distance ~1.5m away from crown XY, at crown's Z height
        for i in range(num_envs):
            crown = get_crown_centroid(stage, palm_root_paths[i])
            # Start at crown + 1.5m in the +X direction (arbitrary angle)
            new_root[i, 0] = float(crown[0]) + 1.5
            new_root[i, 1] = float(crown[1])
            new_root[i, 2] = float(crown[2])
            # Quaternion stays at default (identity)
        
        robot.write_root_pose_to_sim(new_root[:, :7], env_ids=None)
        print(f"[init] pre-positioned robot at crown +1.5m", flush=True)
    
    for palm_path in palm_root_paths:
        disable_palm_physics(stage, palm_path)
    episode_rng = np.random.default_rng(getattr(args_cli, "seed", None))
    
    HDRI_FOLDER_PATH = "/home/cirplab/moore/isaac_data/palm_tree_models/blender/pretoria_gardens_4k/hdri"
    randomize_lighting(stage, HDRI_FOLDER_PATH, env_ids=range(num_envs))

    # Cull leaves BEFORE positioning the robot so that robot placement and arm
    # targets are both based on the same (post-cull) leaf geometry
    cull_episode_leaves(
        stage=stage,
        palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
        cull_prob=args_cli.top_leaf_cull_prob,
    )

    randomize_robot_root_pose(
        env=env, stage=stage, palm_root_paths=palm_root_paths,
        episode_rng=episode_rng,
    )

    current_hover_targets = np.zeros((num_envs, 3), dtype=np.float32)
    robot_xys_now = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].cpu().numpy()
    current_hover_targets[:] = prepare_episode_targets(
        stage=stage,
        palm_root_paths=palm_root_paths,
        robot_xys=robot_xys_now,
    )

    oracles = [SprayOracle() for _ in range(num_envs)]
    spawn_target_markers(stage, current_hover_targets, marker_type="hover", color=(0.0, 0.0, 1.0))

    moving_gripper_indices, _ = env.unwrapped.scene["robot"].find_bodies("moving_gripper")
    moving_gripper_idx = moving_gripper_indices[0]

    # ─── LeRobot dataset initialization (only when --save_data) ──────────
    robot_asset = env.unwrapped.scene["robot"]
    num_dof = robot_asset.data.default_joint_pos.shape[-1]
    datasets = None
    if args_cli.save_data:
        img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][0]
        IMG_H, IMG_W = img_tensor.shape[0], img_tensor.shape[1]
        features = {
            "env_id": {"dtype": "int64", "shape": (1,), "names": None},
            "observation.state": {
                "dtype": "float32", "shape": (num_dof,),
                "names": [f"joint_{i}" for i in range(num_dof)],
            },
            "observation.state.ee_pose": {
                "dtype": "float32", "shape": (7,),
                "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            },
            "observation.images.wrist_camera": {
                "dtype": "video", "shape": (3, IMG_H, IMG_W),
                "names": ["c", "h", "w"],
            },
            "action": {
                "dtype": "float32", "shape": (7,),
                "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
            },
        }
        datasets = []
        for env_id in range(num_envs):
            env_repo_id = f"local/vla_palm_dataset_env_{env_id:04d}"
            env_root = os.path.join(args_cli.dataset_root, f"env_{env_id:04d}")
            env_dataset = LeRobotDataset.create(
                repo_id=env_repo_id,
                root=env_root,
                fps=30,
                features=features,
            )
            # Force immediate metadata flush after every save_episode so that
            # episode index files exist on disk even if the script exits via
            # os._exit(0) before normal Python finalization can flush buffers.
            env_dataset.meta.metadata_buffer_size = 1
            datasets.append(env_dataset)
        abs_save_path = os.path.abspath(args_cli.dataset_root)
        print(f"[INFO]: Initialized {num_envs} per-env LeRobot datasets.")
        print(f"[INFO]: >>> SAVING DATA TO DIRECTORY: {abs_save_path} <<<")
    else:
        print("[INFO]: Running in TEST MODE. Data will NOT be saved.")

    episode_frame_count = np.zeros(num_envs, dtype=np.int64)
    saved_frame_count = np.zeros(num_envs, dtype=np.int64)

    # Resolve wrist_pitch joint index for the joint-space "snap down" override
    # used while the FSM is in settle/spray/success_hold (states 2/3/4). IK
    # alone can't drive wrist_pitch to its negative limit with 4 DOF, so we
    # bypass it and write the joint angle directly during those states.
    joint_names = env.unwrapped.scene["robot"].joint_names
    shoulder_pitch_idx = joint_names.index("shoulder_pitch")
    elbow_pitch_idx = joint_names.index("elbow_pitch")
    wrist_pitch_idx = joint_names.index("wrist_pitch")
    base_yaw_idx = joint_names.index("base_yaw")
    BASE_YAW_LIMIT = float(np.deg2rad(10.0))   # cap base_yaw joint to ±10° each step
    WRIST_PITCH_REST = REST_POSE_VALUES[3]   # +1.2 rad — wrist up at rest
    WRIST_PITCH_DOWN = -1.5708               # lower limit — EE points straight down
    WRIST_LERP_STEPS = 90                    # slow wrist rotation (≈3s @ 30Hz; smoother trajectory)
    SNAP_STATES = {2, 3, 4}

    prev_states = [oracle.state for oracle in oracles]
    leaf_stuck_steps = [0] * num_envs
    pending_reset_env_ids = set()
    step_counter = 0
    state_names = {0: "approach_wp", 1: "approach_hover", 2: "settle",
                   3: "spray", 4: "success_hold", 5: "fail_hold"}
    first_step = True
    
    while simulation_app.is_running():
        robot_data = env.unwrapped.scene["robot"].data
        ee_pos_all = robot_data.body_pos_w[:, moving_gripper_idx].cpu().numpy()
        ee_quat_all = robot_data.body_quat_w[:, moving_gripper_idx].cpu().numpy()
        root_quat_all = robot_data.root_quat_w.cpu().numpy()
        
        # Debug: Print positions on first step
        if first_step and DEBUG_VERBOSE:
            robot_base_all = robot_data.root_pos_w.cpu().numpy()
            for env_id in range(num_envs):
                ee_pos = ee_pos_all[env_id]
                target_pos = current_hover_targets[env_id]
                base_pos = robot_base_all[env_id]
                dist_to_target = float(np.linalg.norm(ee_pos - target_pos))
                print(f"[first_step env {env_id}] ee={ee_pos} target={target_pos} "
                      f"dist={dist_to_target:.3f}m base={base_pos}", flush=True)
            first_step = False

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

        # ─── Dataset recording (only during state < 4: approach + settle + spray) ───
        if datasets is not None or args_cli.save_data is False:
            joint_positions_all = robot_asset.data.joint_pos.cpu().numpy()
            for env_id in range(num_envs):
                if oracles[env_id].state < 4:
                    if args_cli.save_data and datasets is not None:
                        img_tensor = env.unwrapped.scene["wrist_camera"].data.output["rgb"][env_id]
                        img_numpy = img_tensor.cpu().numpy()
                        if img_numpy.shape[-1] == 4:
                            img_rgb = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2RGB)
                        else:
                            img_rgb = img_numpy
                        ee_pose_env = np.concatenate([ee_pos_all[env_id], ee_quat_all[env_id]])
                        frame_dict = {
                            "env_id": np.array([env_id], dtype=np.int64),
                            "task": TASK_DESCRIPTION,
                            "observation.state": joint_positions_all[env_id].astype(np.float32),
                            "observation.state.ee_pose": ee_pose_env.astype(np.float32),
                            "observation.images.wrist_camera": Image.fromarray(img_rgb),
                            "action": action_batch[env_id].astype(np.float32),
                        }
                        datasets[env_id].add_frame(frame_dict)
                    episode_frame_count[env_id] += 1

        # ─── Hard cap on total frames; finalize datasets and exit cleanly ───
        total_saved_frames = int(saved_frame_count.sum())
        if total_saved_frames >= MAX_TOTAL_SAVED_FRAMES:
            if args_cli.save_data and datasets is not None:
                for env_id in range(num_envs):
                    if episode_frame_count[env_id] > 0:
                        datasets[env_id].clear_episode_buffer()
                    datasets[env_id].finalize()
                abs_save_path = os.path.abspath(args_cli.dataset_root)
                print(f"[INFO]: Collected {total_saved_frames} high-quality frames.")
                print(f"[INFO]: >>> DATA SUCCESSFULLY SAVED TO: {abs_save_path} <<<")
            try:
                env.close()
            except Exception:
                pass
            os._exit(0)

        action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=sim_device)

        _, _, terminated, truncated, _ = env.step(action_tensor)

        # No joint-space override. 4-DOF IK + position-only commands → arm
        # converges cleanly to hover without orientation drift or clamp fights.

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

        pending_reset_env_ids.update(done_env_ids)
        pending_reset_env_ids.update(stuck_env_ids)

        if len(pending_reset_env_ids) == num_envs and pending_reset_env_ids:
            reset_env_ids = list(range(num_envs))

            # Save/clear LeRobot episodes: keep successful runs, drop failed/stuck.
            if args_cli.save_data and datasets is not None:
                for env_id in reset_env_ids:
                    keep = (
                        oracles[env_id].completed
                        and not oracles[env_id].timed_out
                        and env_id not in stuck_env_ids
                    )
                    if episode_frame_count[env_id] > 0:
                        if keep:
                            datasets[env_id].save_episode()
                            saved_frame_count[env_id] += episode_frame_count[env_id]
                        else:
                            datasets[env_id].clear_episode_buffer()
                print(
                    f"[INFO]: Resetting all {len(reset_env_ids)} envs together; "
                    f"total frames saved so far: {int(saved_frame_count.sum())}"
                )
            if DEBUG_VERBOSE:
                reasons = []
                for env_id in reset_env_ids:
                    if env_id in stuck_env_ids:
                        reasons.append(f"env {env_id}: leaf-stuck")
                    elif env_id in done_env_ids:
                        reasons.append(f"env {env_id}: done")
                print(f"[step {step_counter:5d}] global reset "
                      f"({', '.join(reasons)})", flush=True)

            randomize_lighting(stage, HDRI_FOLDER_PATH, env_ids=range(num_envs))
            # Cull leaves BEFORE repositioning robot so placement is based on final geometry
            cull_episode_leaves(
                stage=stage,
                palm_root_paths=palm_root_paths,
                episode_rng=episode_rng,
                cull_prob=args_cli.top_leaf_cull_prob,
                env_ids=reset_env_ids,
            )
            randomize_robot_root_pose(
                env=env, stage=stage, palm_root_paths=palm_root_paths,
                episode_rng=episode_rng, env_ids=reset_env_ids,
            )
            robot_xys_now = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].cpu().numpy()
            refreshed_targets = prepare_episode_targets(
                stage=stage,
                palm_root_paths=palm_root_paths,
                robot_xys=robot_xys_now,
                env_ids=reset_env_ids,
            )
            for i, env_id in enumerate(reset_env_ids):
                current_hover_targets[env_id] = refreshed_targets[i]
                oracles[env_id] = SprayOracle()
                prev_states[env_id] = 0
                leaf_stuck_steps[env_id] = 0
                episode_frame_count[env_id] = 0
            spawn_target_markers(
                stage, current_hover_targets,
                env_ids=reset_env_ids, marker_type="hover", color=(0.0, 0.0, 1.0),
            )
            pending_reset_env_ids.clear()

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
    os._exit(0)   # os is imported at module top


if __name__ == "__main__":
    main()
