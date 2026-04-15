# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Asset paths (absolute, pointing into the depthbasedRL repo)
# ---------------------------------------------------------------------------
_ASSETS_ROOT = Path("/home/tylerlum/github_repos/depthbasedRL/assets/urdf")

ROBOT_URDF_PATH = str(_ASSETS_ROOT / "kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")
TABLE_URDF_PATH = str(_ASSETS_ROOT / "table_narrow.urdf")

# Default object — swap out per experiment
OBJECT_URDF_PATH = str(
    Path("/home/tylerlum/github_repos/depthbasedRL/assets/urdf/dextoolbench/hammer/claw_hammer/claw_hammer.urdf")
)

# USD cache — avoids re-converting URDFs every run
USD_CACHE_DIR = "/tmp/simtoolreal_usd_cache"


@configclass
class SimToolRealEnvCfg(DirectRLEnvCfg):
    """Configuration for the SimToolReal dexterous manipulation environment.

    Kuka IIWA-14 arm + SHARPA hand (29 DOF total) grasps and reorients
    objects on a table to match a target goal pose.
    """

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    # 60 Hz physics, policy runs at 60 Hz (decimation=1)
    decimation: int = 1
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(
            solver_type=1,  # TGS — matches Isaac Gym training
            bounce_threshold_velocity=0.2,
        ),
    )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=1.5,
        replicate_physics=True,
    )

    # ------------------------------------------------------------------
    # Spaces — sizes derived from obs lists below
    # ------------------------------------------------------------------
    # Policy obs (no privileged info): joint_pos(11) + joint_vel(11) +
    #   prev_actions(11) + palm_pos(3) + palm_rot(4) + object_rot(4) +
    #   fingertip_pos_rel_palm(12) + keypoints_rel_palm(3K) + keypoints_rel_goal(3K) +
    #   object_scales(3)
    # With default K=4 keypoints: 11+11+11+3+4+4+12+12+12+3 = 83
    num_keypoints: int = 4
    observation_space: int = 83  # policy obs (no privileged info)
    state_space: int = 0         # set >0 to enable asymmetric critic
    action_space: int = 11       # 7 arm DOFs + 4 hand DOFs controlled

    # ------------------------------------------------------------------
    # Asset paths
    # ------------------------------------------------------------------
    robot_urdf_path: str = ROBOT_URDF_PATH
    table_urdf_path: str = TABLE_URDF_PATH
    object_urdf_path: str = OBJECT_URDF_PATH
    usd_cache_dir: str = USD_CACHE_DIR

    # ------------------------------------------------------------------
    # Robot position in world (matches Isaac Gym training setup)
    # ------------------------------------------------------------------
    robot_translation: tuple[float, float, float] = (0.0, 0.8, 0.0)
    table_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # DOF control
    # ------------------------------------------------------------------
    # Number of DOFs actually controlled by the policy
    num_arm_dofs: int = 7
    num_hand_dofs: int = 4   # policy controls 4 (thumb, index, middle, ring)
    num_policy_dofs: int = 11  # arm + hand controlled DOFs

    # Action scaling: policy outputs in [-1,1], scaled to joint speed targets
    dof_speed_scale: float = 1.5      # rad/s scale
    arm_moving_average: float = 0.1   # EMA smoothing for arm targets
    hand_moving_average: float = 0.1  # EMA smoothing for hand targets
    use_relative_control: bool = False

    # ------------------------------------------------------------------
    # Episode / reset
    # ------------------------------------------------------------------
    episode_length_s: float = 10.0   # 600 steps at 60 Hz
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    reset_dof_pos_noise_arm: float = 0.1
    reset_dof_pos_noise_fingers: float = 0.1
    reset_dof_vel_noise: float = 0.5
    randomize_object_rotation: bool = True

    # Table geometry
    table_reset_z: float = 0.38
    table_object_z_offset: float = 0.25  # height above table base for object reset

    # Default arm pose (matches Isaac Gym training)
    default_arm_joint_pos: tuple[float, ...] = (
        -1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308
    )

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    object_fall_z_threshold: float = 0.1   # object z < this → fallen off table
    max_fingertip_dist_threshold: float = 1.5  # hand too far from object
    consecutive_success_steps: int = 50

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15  # m above table
    keypoint_rew_scale: float = 200.0
    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0
    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003
    success_tolerance: float = 0.075       # keypoint L2 threshold [m]

    # Keypoint geometry
    keypoint_scale: float = 1.5
    object_base_size: float = 0.04

    # ------------------------------------------------------------------
    # Observation noise / delay (leave at 0 initially; add for DR later)
    # ------------------------------------------------------------------
    obs_delay_max: int = 0
    action_delay_max: int = 0
    object_state_delay_max: int = 0
    clamp_abs_observations: float = 10.0

    # ------------------------------------------------------------------
    # Random external forces on object
    # ------------------------------------------------------------------
    force_scale: float = 20.0
    force_prob_range: tuple[float, float] = (0.001, 0.1)
    force_only_when_lifted: bool = True
    torque_scale: float = 2.0
    torque_prob_range: tuple[float, float] = (0.001, 0.1)
    torque_only_when_lifted: bool = True
