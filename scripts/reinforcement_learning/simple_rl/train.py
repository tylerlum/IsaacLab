# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E402

"""Script to train RL agent with SimpleRL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with SimpleRL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--sigma", type=str, default=None, help="The policy's initial standard deviation."
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml, load_yaml
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_rl.simple_rl.ppo_agent import PpoAgent, PpoConfig
from isaaclab_rl.simple_rl.utils.dict_to_dataclass import dict_to_dataclass
from isaaclab_rl.simple_rl.utils.network import NetworkConfig
from isaaclab_tasks.utils.hydra import hydra_task_config
from omegaconf import OmegaConf

import wandb
from wandb.sdk.lib.runid import generate_id

OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("eval", eval)


# Same wrapper as RL-Games
class SimpleRlVecEnvWrapper(RlGamesVecEnvWrapper):
    pass


# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "simple_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with SimpleRL agent."""
    # override configurations with non-hydra CLI arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg["ppo"]["max_epochs"] = args_cli.max_iterations

    if args_cli.checkpoint is not None:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    else:
        checkpoint_path = None

    if args_cli.sigma is not None:
        sigma = float(args_cli.sigma)
    else:
        sigma = None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["seed"] += app_launcher.global_rank
        agent_cfg["ppo"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "simple_rl", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(log_root_path, datetime_str)
    print(f"[INFO] Logging experiment in directory: {experiment_dir}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(experiment_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(experiment_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(experiment_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(experiment_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Hardcoded values for clip_obs and clip_actions
    clip_obs = 5.0
    clip_actions = 1.0

    # Use same device for sim and rl
    sim_device = env_cfg.sim.device
    rl_device = sim_device

    # wrap around environment for simple-rl
    env = SimpleRlVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # set number of actors into agent config
    agent_cfg["ppo"]["num_actors"] = env.unwrapped.num_envs

    USE_WANDB = True
    if USE_WANDB:
        wandb_name = f"{args_cli.task}_{datetime_str}"
        wandb_config = {
            "agent": agent_cfg,
            "env": load_yaml(
                os.path.join(experiment_dir, "params", "env.yaml"), unsafe=True
            ),
        }
        wandb.init(
            project="isaaclab",
            entity="tylerlum",
            name=wandb_name,
            group=None,
            config=wandb_config,
            sync_tensorboard=True,
            id=f"{wandb_name}_{generate_id()}",
        )

    # Create agent
    network_config = dict_to_dataclass(agent_cfg["network"], NetworkConfig)
    ppo_config = dict_to_dataclass(agent_cfg["ppo"], PpoConfig)
    ppo_config.device = rl_device

    agent = PpoAgent(
        experiment_dir=Path(experiment_dir),
        ppo_config=ppo_config,
        network_config=network_config,
        env=env,
    )
    if checkpoint_path is not None and checkpoint_path != "":
        agent.restore(Path(checkpoint_path))
    if sigma is not None:
        agent.override_sigma(sigma)
    agent.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
