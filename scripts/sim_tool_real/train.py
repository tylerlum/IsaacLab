# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train SimToolReal with the custom rl_games fork (LSTM asymmetric PPO).

This script is a thin wrapper around the canonical IsaacLab rl_games train.py
that puts the custom rl_games package (third_party/rl_games_custom) on the
front of sys.path so its asymmetric actor-critic modifications take precedence
over the system-installed rl_games.

Usage (from the IsaacLab repo root):

  ./isaaclab.sh -p scripts/sim_tool_real/train.py \\
      --task Isaac-SimToolReal-Direct-v0 \\
      --num_envs 4096 \\
      --headless

Resume from checkpoint:

  ./isaaclab.sh -p scripts/sim_tool_real/train.py \\
      --task Isaac-SimToolReal-Direct-v0 \\
      --checkpoint logs/rl_games/SimToolRealLSTMAsymmetricPPO/<run>/nn/SimToolRealLSTMAsymmetricPPO.pth
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Insert custom rl_games BEFORE the system-installed one ─────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CUSTOM_RL_GAMES = _REPO_ROOT / "third_party" / "rl_games_custom"
if _CUSTOM_RL_GAMES.exists():
    sys.path.insert(0, str(_CUSTOM_RL_GAMES))
    print(f"[SimToolReal] Using custom rl_games from: {_CUSTOM_RL_GAMES}")
else:
    print(f"[SimToolReal] WARNING: custom rl_games not found at {_CUSTOM_RL_GAMES}, using system rl_games")
# ───────────────────────────────────────────────────────────────────────────

import argparse
import contextlib
import logging
import math
import random
import time
from datetime import datetime
from distutils.util import strtobool

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

logger = logging.getLogger(__name__)

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train SimToolReal with RL-Games (custom fork).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--sigma", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
)
parser.add_argument("--wandb-project-name", type=str, default=None)
parser.add_argument("--wandb-entity", type=str, default=None)
parser.add_argument("--wandb-name", type=str, default=None)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
        agent_cfg["params"]["config"]["max_epochs"] = (
            args_cli.max_iterations if args_cli.max_iterations is not None
            else agent_cfg["params"]["config"]["max_epochs"]
        )

        if args_cli.checkpoint is not None:
            resume_path = retrieve_file_path(args_cli.checkpoint)
            agent_cfg["params"]["load_checkpoint"] = True
            agent_cfg["params"]["load_path"] = resume_path
            print(f"[INFO]: Loading checkpoint from: {resume_path}")
        train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

        if args_cli.distributed:
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            agent_cfg["params"]["seed"] += int(os.getenv("RANK", "0"))
            agent_cfg["params"]["config"]["device"] = f"cuda:{local_rank}"
            agent_cfg["params"]["config"]["device_name"] = f"cuda:{local_rank}"
            agent_cfg["params"]["config"]["multi_gpu"] = True
            env_cfg.sim.device = f"cuda:{local_rank}"

        env_cfg.seed = agent_cfg["params"]["seed"]

        # Clamp minibatch_size so num_minibatches >= 1.
        # The YAML is tuned for large-scale training (4096+ envs). When running
        # with fewer envs (e.g. smoke tests), batch_size < minibatch_size gives
        # num_minibatches=0 and a ZeroDivisionError inside rl_games.
        horizon = agent_cfg["params"]["config"]["horizon_length"]
        num_envs_effective = env_cfg.scene.num_envs
        batch_size = horizon * num_envs_effective
        cfg_p = agent_cfg["params"]["config"]
        if cfg_p.get("minibatch_size", batch_size) > batch_size:
            # Snap to the largest power-of-2 that fits and satisfies seq_length
            seq_len = cfg_p.get("seq_length", 1)
            minibatch_size = batch_size - (batch_size % seq_len) if seq_len > 1 else batch_size
            minibatch_size = max(minibatch_size, seq_len)
            cfg_p["minibatch_size"] = minibatch_size
            if "central_value_config" in cfg_p:
                cfg_p["central_value_config"]["minibatch_size"] = minibatch_size
            print(f"[INFO] minibatch_size clamped to {minibatch_size} (batch_size={batch_size})")

        config_name = agent_cfg["params"]["config"]["name"]
        log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
        print(f"[INFO] Logging to: {log_root_path}")
        log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        agent_cfg["params"]["config"]["train_dir"] = log_root_path
        agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

        dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)
        runner.reset()

        if args_cli.track:
            import wandb
            wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
            wandb.init(
                project=wandb_project,
                entity=args_cli.wandb_entity,
                name=log_dir if args_cli.wandb_name is None else args_cli.wandb_name,
                sync_tensorboard=True,
                monitor_gym=True,
                save_code=True,
            )

        try:
            start_time = time.time()
            if args_cli.checkpoint is not None:
                runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
            else:
                runner.run({"train": True, "play": False, "sigma": train_sigma})
            print(f"Training time: {round(time.time() - start_time, 2)}s")
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
