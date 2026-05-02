from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from leisaac.utils.constant import ASSETS_ROOT

"""Configuration for the PingTi Arm v3 Robot."""
PINGTI_ARM_V3_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "pingti" / "PingTi_Arm_v3.usd"

PINGTI_ARM_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(PINGTI_ARM_V3_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(2.2, -0.61, 0.89),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "base_yaw": 0.0,
            "shoulder_pitch": 0.0,
            "elbow_pitch": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
            "gripper_moving": 0.0,
        },
    ),
    actuators={
        "pingti-gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_moving"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
        "pingti-arm": ImplicitActuatorCfg(
            joint_names_expr=["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"],
            effort_limit_sim=10,
            velocity_limit_sim=10,
            stiffness=17.8,
            damping=0.60,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

PINGTI_ARM_V3_JOINT_NAMES = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "gripper_moving"]

# Joint limits from URDF (degrees)
PINGTI_ARM_V3_USD_JOINT_LIMITS = {
    "base_yaw": (-110.0, 110.0),
    "shoulder_pitch": (-100.0, 100.0),
    "elbow_pitch": (-90.0, 110.0),
    "wrist_pitch": (-110.0, 110.0),
    "wrist_roll": (-360.0, 360.0),
    "gripper_moving": (-12.0, 90.0),
}
