import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.kitchen import (
    KITCHEN_WITH_ORANGE_CFG,
    KITCHEN_WITH_ORANGE_USD_PATH,
)
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp


@configclass
class PickOrangeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the pick orange task."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class ObservationsCfg(SingleArmObservationsCfg):
    pass
    
@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    pass

@configclass
class PickOrangeEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()
    
    terminations: TerminationsCfg = TerminationsCfg()
    
    task_description: str = "Pick three oranges and put them into the plate, then reset the arm to rest state."

    def __post_init__(self) -> None:
        super().__post_init__()

        domain_randomization(self, random_options=[])
