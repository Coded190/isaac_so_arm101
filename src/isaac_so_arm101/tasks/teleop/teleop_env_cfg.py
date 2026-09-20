# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural (and optional palm) teleop environments for PingTi."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaac_so_arm101.robots.pingti.pingti import PING_TI_CFG
from isaac_so_arm101.teleop_constants import (
    GRIPPER_CLOSED_RAD,
    GRIPPER_OPEN_RAD,
    PINGTI_ARM_JOINTS,
    PINGTI_EE_BODY,
    PINGTI_GRIPPER_JOINT,
    PINGTI_PALM_POS,
    PINGTI_PALM_ROT,
    PINGTI_TABLE_POS,
    PINGTI_TABLE_ROT,
)

TELEOP_DEVICES = ("keyboard", "so101leader")


def _zero_joint_state(*, pos, rot) -> ArticulationCfg.InitialStateCfg:
    return ArticulationCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos={
            "base_yaw": 0.0,
            "shoulder_pitch": 0.0,
            "elbow_pitch": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
            "gripper_moving": 0.0,
        },
        joint_vel={".*": 0.0},
    )


def _teleop_pingti_cfg(*, pos, rot) -> ArticulationCfg:
    """Free root so Kit Transform writes stick.

    ``PING_TI_CFG`` (Reach/VLA) uses ``fix_base=True``, which authors a USD
    ``FixedJoint`` on the root. PhysX keeps that weld at the spawn pose, so
    ``write_root_pose_to_sim_index`` does not persist mid-sim. Teleop drops the
    weld and re-applies the Property-panel pose every step (``reason=hold``).
    """
    spawn = PING_TI_CFG.spawn.replace(
        fix_base=False,
        rigid_props=PING_TI_CFG.spawn.rigid_props.replace(disable_gravity=True),
    )
    return PING_TI_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=_zero_joint_state(pos=pos, rot=rot),
        spawn=spawn,
    )


def _table_pingti_cfg() -> ArticulationCfg:
    return _teleop_pingti_cfg(pos=PINGTI_TABLE_POS, rot=PINGTI_TABLE_ROT)


def _palm_pingti_cfg() -> ArticulationCfg:
    return _teleop_pingti_cfg(pos=PINGTI_PALM_POS, rot=PINGTI_PALM_ROT)


@configclass
class TeleopSceneCfg(InteractiveSceneCfg):
    """Ground + dome light. No Nucleus table, no palm USD."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot: ArticulationCfg = MISSING


@configclass
class PalmTeleopSceneCfg(InteractiveSceneCfg):
    """Palm garden scene. Requires ``uv run fetch_assets`` (or a local pack)."""

    custom_env = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(usd_path="UNSET"),
    )
    # Low fill only so a failed HDRI bind is not black. Garden DomeLight is 1000.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=200.0),
    )
    robot: ArticulationCfg = MISSING

    def __post_init__(self):
        from isaac_so_arm101.assets import report_scene_pack_gaps, require_scene_usd

        usd = require_scene_usd("palm_environment")
        self.custom_env.spawn.usd_path = str(usd)
        for line in report_scene_pack_gaps(usd):
            print(line, flush=True)


@configclass
class CommandsCfg:
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=PINGTI_EE_BODY,
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.15),
            pos_y=(0.0, 0.0),
            pos_z=(0.25, 0.25),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    # Dummy term so ManagerBasedRLEnv has a reward group; weight 0 = no learning signal.
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


def _apply_se3_actions(cfg) -> None:
    cfg.actions.arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(PINGTI_ARM_JOINTS),
        body_name=PINGTI_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )
    cfg.actions.gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[PINGTI_GRIPPER_JOINT],
        open_command_expr={PINGTI_GRIPPER_JOINT: GRIPPER_OPEN_RAD},
        close_command_expr={PINGTI_GRIPPER_JOINT: GRIPPER_CLOSED_RAD},
    )


def _apply_joint_pos_actions(cfg) -> None:
    """Absolute 6-D joint targets from the SO-ARM101 leader (not DiffIK)."""
    cfg.actions.arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(PINGTI_ARM_JOINTS),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    cfg.actions.gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[PINGTI_GRIPPER_JOINT],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


def apply_teleop_device(cfg, teleop_device: str) -> None:
    """Swap action terms after ``parse_env_cfg``. Keyboard stays 7-D SE3."""
    if teleop_device == "keyboard":
        _apply_se3_actions(cfg)
        return
    if teleop_device == "so101leader":
        _apply_joint_pos_actions(cfg)
        return
    raise ValueError(
        f"Unsupported --teleop_device={teleop_device!r}. Use one of {TELEOP_DEVICES}."
    )


@configclass
class PingTiTeleopEnvCfg(ManagerBasedRLEnvCfg):
    """Single-env keyboard teleop on a procedural ground plane."""

    scene: TeleopSceneCfg = TeleopSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 1.0e6
        self.sim.dt = 1.0 / 60.0
        self.viewer.eye = (1.6, 1.6, 1.1)
        self.viewer.lookat = (0.0, 0.0, 0.25)
        self.scene.robot = _table_pingti_cfg()
        _apply_se3_actions(self)
        self.observations.policy.enable_corruption = False


@configclass
class PingTiPalmTeleopEnvCfg(PingTiTeleopEnvCfg):
    """Same SE(3) actions in the palm garden. Requires the scene pack."""

    def __post_init__(self):
        super().__post_init__()
        self.scene = PalmTeleopSceneCfg(num_envs=1, env_spacing=2.5)
        self.scene.robot = _palm_pingti_cfg()
        self.viewer.lookat = PINGTI_PALM_POS
        self.viewer.eye = (
            PINGTI_PALM_POS[0] + 1.7,
            PINGTI_PALM_POS[1] + 1.5,
            PINGTI_PALM_POS[2] + 0.85,
        )
        _apply_se3_actions(self)
