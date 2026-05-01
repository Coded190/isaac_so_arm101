import torch

from isaaclab.sim import UsdFileCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass

from ..template import SingleArmTaskEnvCfg, SingleArmTaskSceneCfg


# -------------------------
# SCENE
# -------------------------
@configclass
class TeleopSceneCfg(SingleArmTaskSceneCfg):

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=UsdFileCfg(
            usd_path="scenes/kitchen_with_orange/scene.usd",
            copy_from_source=False,
        ),
    )


# -------------------------
# ENV CONFIG
# -------------------------
@configclass
class TeleopEnvCfg(SingleArmTaskEnvCfg):

    scene: TeleopSceneCfg = TeleopSceneCfg(env_spacing=2.5)

    # disable RL systems
    observations = None
    actions = None
    rewards = None
    terminations = None
    events = None

    task_description: str = "SO101 teleoperation environment"

    # 👇 THIS is where robot must go
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=None,  # replace with SO101 USD if needed
    )

    def __post_init__(self):
        super().__post_init__()
