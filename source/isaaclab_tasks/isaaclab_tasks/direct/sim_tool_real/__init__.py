# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Kuka IIWA-14 + SHARPA hand dexterous tool-manipulation environment (IsaacLab port).

Ported from the Isaac Gym SimToolReal task.  The robot grasps and reorients
tools on a table to match a sampled goal pose.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-SimToolReal-Direct-v0",
    entry_point=f"{__name__}.sim_tool_real_env:SimToolRealEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sim_tool_real_env_cfg:SimToolRealEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
