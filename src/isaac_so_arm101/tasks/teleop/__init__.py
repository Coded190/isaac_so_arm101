# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Isaac-PING-TI-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teleop_env_cfg:PingTiTeleopEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PING-TI-Teleop-Palm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.teleop_env_cfg:PingTiPalmTeleopEnvCfg",
    },
    disable_env_checker=True,
)
