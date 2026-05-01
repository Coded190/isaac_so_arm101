import gymnasium as gym

from .teleop_env_cfg import TeleopEnvCfg

gym.register(
    id="LeIsaac-SO101-Teleop-v0",
    entry_point="leisaac.tasks.teleop_so101.teleop_env_cfg:TeleopEnvCfg",
    disable_env_checker=True,
)
