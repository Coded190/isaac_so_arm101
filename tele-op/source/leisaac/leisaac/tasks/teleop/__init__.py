import gymnasium as gym

gym.register(
    id="LeIsaac-PingTi-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "leisaac.tasks.teleop.teleop_env_cfg:PingTiTeleopEnvCfg",
    },
)
