# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a pretrained SimToolReal policy in Isaac Lab without training.

Ported from depthbasedRL/dextoolbench/eval.py.  Loads a dextoolbench
trajectory file (start pose + goal sequence) and evaluates the policy
over multiple episodes, printing per-episode success stats.

Example — claw_hammer swing_down:

  ./isaaclab.sh -p scripts/sim_tool_real/play.py \\
      --checkpoint /path/to/model.pth \\
      --agent_cfg   /path/to/config.yaml \\
      --trajectory  /home/tylerlum/github_repos/depthbasedRL/dextoolbench/trajectories/hammer/claw_hammer/swing_down.json \\
      --object_urdf /home/tylerlum/github_repos/depthbasedRL/assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf \\
      --num_envs 1 \\
      --num_episodes 5

Without a trajectory (random goals):

  ./isaaclab.sh -p scripts/sim_tool_real/play.py \\
      --checkpoint /path/to/model.pth \\
      --num_envs 4 \\
      --headless
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
# ───────────────────────────────────────────────────────────────────────────

import argparse
import contextlib
import json
import math
import random
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate a pretrained SimToolReal policy.")
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-Direct-v0")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to the rl_games .pth checkpoint file.",
)
parser.add_argument(
    "--agent_cfg", type=str, default=None,
    help="Optional path to override the agent YAML config (default: from task registry).",
)
parser.add_argument(
    "--trajectory", type=str, default=None,
    help="Path to a dextoolbench trajectory JSON (start_pose + goals). "
         "If omitted, random goals are used.",
)
parser.add_argument(
    "--object_urdf", type=str, default=None,
    help="Path to the object URDF to swap into the env. "
         "Overrides the default object_urdf_path in the env cfg.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--real_time", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument(
    "--z_offset", type=float, default=0.03,
    help="Extra z offset added to trajectory start_pose to avoid table clipping.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)

    # ── Apply env overrides ────────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Swap object URDF if requested
    if args_cli.object_urdf is not None:
        env_cfg.object_urdf_path = args_cli.object_urdf
        print(f"[INFO] Using object URDF: {args_cli.object_urdf}")

    # Load trajectory → fixed initial pose + goal sequence
    if args_cli.trajectory is not None:
        traj_path = Path(args_cli.trajectory)
        assert traj_path.exists(), f"Trajectory not found: {traj_path}"
        with open(traj_path) as f:
            traj_data = json.load(f)

        start_pose = list(traj_data["start_pose"])
        start_pose[2] += args_cli.z_offset  # lift slightly above table

        env_cfg.use_fixed_object_pose = True
        env_cfg.fixed_object_pose = tuple(start_pose)

        env_cfg.use_fixed_goals = True
        env_cfg.fixed_goal_poses = tuple(tuple(g) for g in traj_data["goals"])

        print(f"[INFO] Trajectory: {traj_path.name} ({len(traj_data['goals'])} goals)")
        print(f"[INFO] Start pose: {[f'{v:.3f}' for v in start_pose]}")

    # Full eval mode: no reset randomization
    env_cfg.eval_mode = True
    # Disable external forces during eval
    env_cfg.force_scale = 0.0
    env_cfg.torque_scale = 0.0

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    # ── Override agent config if custom YAML provided ──────────────────────
    if args_cli.agent_cfg is not None:
        import yaml
        with open(args_cli.agent_cfg) as f:
            override_cfg = yaml.safe_load(f)
        agent_cfg.update(override_cfg)
        print(f"[INFO] Overriding agent cfg from: {args_cli.agent_cfg}")

    with launch_simulation(env_cfg, args_cli):
        rl_device = agent_cfg["params"]["config"]["device"]
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        obs_groups = agent_cfg["params"]["env"].get("obs_groups")
        concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

        # Build env
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if args_cli.video:
            video_dir = Path("videos") / "sim_tool_real_eval"
            video_dir.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                str(video_dir),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )

        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # ── Load policy ────────────────────────────────────────────────────
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        print(f"[INFO] Loading checkpoint: {resume_path}")

        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        step_dt = env.unwrapped.step_dt

        # ── Eval loop ──────────────────────────────────────────────────────
        episode_successes = []
        episode_lengths = []

        for ep in range(args_cli.num_episodes):
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]

            _ = agent.get_batch_size(obs, 1)
            if agent.is_rnn:
                agent.init_rnn()

            done = False
            step = 0
            consecutive_successes = 0

            while not done:
                t0 = time.time()

                with torch.inference_mode():
                    obs_t = agent.obs_to_torch(obs)
                    actions = agent.get_action(obs_t, is_deterministic=True)
                    obs, rewards, dones, extras = env.step(actions)
                    if isinstance(obs, dict):
                        obs = obs["obs"]

                    # Reset RNN state on done envs
                    if agent.is_rnn and agent.states is not None:
                        done_mask = dones.bool()
                        for s in agent.states:
                            s[:, done_mask, :] = 0.0

                step += 1

                # Track consecutive successes from env extras (if logged)
                if "log" in extras and "consecutive_successes" in extras["log"]:
                    consecutive_successes = float(extras["log"]["consecutive_successes"])

                done = bool(dones[0].item()) if len(dones) > 0 else False

                elapsed = time.time() - t0
                if args_cli.real_time and (sleep := step_dt - elapsed) > 0:
                    time.sleep(sleep)

            episode_successes.append(consecutive_successes)
            episode_lengths.append(step)
            print(
                f"  Episode {ep + 1}/{args_cli.num_episodes}: "
                f"steps={step}, consec_success={consecutive_successes:.1f}"
            )

        print("\n── Eval summary ──────────────────────────")
        print(f"  Episodes : {args_cli.num_episodes}")
        print(f"  Avg steps: {sum(episode_lengths) / len(episode_lengths):.1f}")
        if episode_successes:
            print(f"  Avg consec_success: {sum(episode_successes) / len(episode_successes):.2f}")

        env.close()


if __name__ == "__main__":
    main()
