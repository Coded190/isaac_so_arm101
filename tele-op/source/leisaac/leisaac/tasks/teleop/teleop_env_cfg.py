import sys
sys.path.insert(0, "/home/cirp-lab/moore/training/isaac_so_arm101/src")

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaac_so_arm101.robots.pingti.pingti import PING_TI_CFG

from ..template import SingleArmTaskEnvCfg, SingleArmTaskSceneCfg
from ..template import mdp as leisaac_mdp
from ..template.single_arm_env_cfg import SingleArmTerminationsCfg


@configclass
class PingTiObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=leisaac_mdp.joint_pos)
        joint_vel = ObsTerm(func=leisaac_mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=leisaac_mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=leisaac_mdp.joint_vel_rel)
        actions = ObsTerm(func=leisaac_mdp.last_action)
        ee_frame_state = ObsTerm(
            func=leisaac_mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(func=leisaac_mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class PingTiTeleopSceneCfg(SingleArmTaskSceneCfg):

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/cirp-lab/moore/palm_tree_models/blender/pretoria_gardens_4k/pretoria_gardens_4k_env_v2.usdc",
            copy_from_source=False,
        ),
    )

    robot: ArticulationCfg = PING_TI_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=PING_TI_CFG.init_state.replace(pos=(2.2, -0.61, 0.89)),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/moving_gripper",
                name="gripper",
            ),
        ],
    )

    # Disable SO101-specific cameras; PingTi USD has no wrist_camera or front_camera prims
    wrist = None
    front = None


@configclass
class PingTiTeleopEnvCfg(SingleArmTaskEnvCfg):

    scene: PingTiTeleopSceneCfg = PingTiTeleopSceneCfg(env_spacing=2.5)

    observations: PingTiObservationsCfg = PingTiObservationsCfg()
    terminations: SingleArmTerminationsCfg = SingleArmTerminationsCfg()

    task_description: str = "PingTi arm v4 teleoperation environment"
    robot_name: str = "pingti_arm_v4"

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        if teleop_device == "so101leader":
            self.actions.arm_action = leisaac_mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"],
                scale=1.0,
            )
            self.actions.gripper_action = leisaac_mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["gripper_moving"],
                scale=1.0,
            )
        else:
            raise ValueError(f"Teleop device '{teleop_device}' is not supported for PingTi. Use 'so101leader'.")

    def preprocess_device_action(self, action, teleop_device) -> torch.Tensor:
        processed = super().preprocess_device_action(action, teleop_device)
        # PingTi shoulder_pitch (idx 1) and elbow_pitch (idx 2) are physically
        # reversed relative to SO101 leader, so negate them.
        processed[:, 1] = -processed[:, 1]
        processed[:, 2] = -processed[:, 2]
        processed[:, 3] = -processed[:, 3]
        return processed

    def __post_init__(self):
        super().__post_init__()
