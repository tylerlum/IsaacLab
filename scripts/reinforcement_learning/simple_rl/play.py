# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL player from Simple-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL player from Simple-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's standard deviation.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


from pathlib import Path
import gymnasium as gym
import math
import os
import time
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from isaaclab_rl.simple_rl.ppo_agent import PpoConfig
from isaaclab_rl.simple_rl.ppo_player import PlayerConfig, PpoPlayer
from isaaclab_rl.simple_rl.utils.network import NetworkConfig
from isaaclab_rl.simple_rl.utils.dict_to_dataclass import dict_to_dataclass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# Same wrapper as RL-Games
class SimpleRlVecEnvWrapper(RlGamesVecEnvWrapper):
    pass

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with Simple-RL player."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "simple_rl_cfg_entry_point")

    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        checkpoint_path = get_published_pretrained_checkpoint("simple_rl", args_cli.task)
        if not checkpoint_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is not None:
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", "simple_rl", args_cli.task)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Looking for checkpoint in directory: {log_root_path}")

        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = "best.pth"
        # get path to previous checkpoint
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint_file, other_dirs=["nn"])
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")

    if args_cli.sigma is not None:
        sigma = float(args_cli.sigma)
    else:
        sigma = None

    experiment_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    experiment_dir = os.path.abspath(experiment_dir)
    print(f"[INFO]: Logging experiment in directory: {experiment_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(experiment_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # Create player
    network_config = dict_to_dataclass(agent_cfg["network"], NetworkConfig)
    player_config = dict_to_dataclass(agent_cfg["player"], PlayerConfig)
    ppo_player_config = dict_to_dataclass(
        agent_cfg["ppo"], PpoConfig
    ).to_ppo_player_config()
    ppo_player_config.device = rl_device

    player = PpoPlayer(
        ppo_player_config=ppo_player_config,
        player_config=player_config,
        network_config=network_config,
        env=env,
    )
    if checkpoint_path is not None and checkpoint_path != "":
        player.restore(Path(checkpoint_path))
    if sigma is not None:
        player.override_sigma(sigma)

    # Run player
    # player.run()

    dt = env.unwrapped.physics_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = player.get_batch_size(obs, 1)

    # initialize RNN states if used
    player.reset()
    if player.is_rnn:
        player.init_rnn()

    # simulate environment
    # note: We simplified the logic in simple-rl player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to player format
            obs = player.obs_to_torch(obs)
            # player stepping
            actions = player.get_action(obs, is_deterministic=player.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if player.is_rnn and player.states is not None:
                    for s in player.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

