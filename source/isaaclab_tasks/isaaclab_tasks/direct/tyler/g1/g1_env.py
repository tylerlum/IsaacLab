# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, LightCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab_assets.robots.unitree import G1_CFG

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

SIM_DT = 0.005
physics_material = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)


@configclass
class G1EnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 37
    observation_space = 123
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=decimation,
        physics_material=physics_material,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
        ),
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=physics_material,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # contact sensor
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=SIM_DT,
    )

    # light
    light: LightCfg = DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )

    # command
    base_velocity_command = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    command_vel_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_command")
    )
    """The configuration for the command velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    """The configuration for the pose visualization marker. Defaults to FRAME_MARKER_CFG."""
    pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


REWARD_NAMES = [
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "dof_torques_l2",
    "dof_acc_l2",
    "action_rate_l2",
    # "undesired_contacts",
    "flat_orientation_l2",
    "termination_penalty",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "feet_air_time",
    "feet_slide",
    "dof_pos_limits_ankle",
    "joint_deviation_hip",
    "joint_deviation_arms",
    "joint_deviation_fingers",
    "joint_deviation_torso",
]


def sample_commands(
    num_envs: int,
    device: torch.device,
    lin_vel_x: tuple[float, float],
    lin_vel_y: tuple[float, float],
    ang_vel_z: tuple[float, float],
    heading: tuple[float, float],
    rel_heading_envs: float,
    rel_standing_envs: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample new commands at every reset

    There are 3 types of envs:
    * Heading envs: follow heading command (vel_commands_b will be changing at each timestep to track the heading)
    * Standing envs: zero velocity command
    * Normal envs: follow fixed randomly-sampled velocity command
    """
    # Sample vel commands to be used for normal envs
    r = torch.empty(num_envs, device=device)
    vel_commands_b = torch.zeros(num_envs, 3, device=device)
    vel_commands_b[:, 0] = r.uniform_(*lin_vel_x)
    vel_commands_b[:, 1] = r.uniform_(*lin_vel_y)
    vel_commands_b[:, 2] = r.uniform_(*ang_vel_z)

    # Sample which envs are heading envs or standing envs
    heading_commands = r.uniform_(*heading)
    is_heading_env = r.uniform_(0.0, 1.0) <= rel_heading_envs
    is_standing_env = r.uniform_(0.0, 1.0) <= rel_standing_envs
    is_heading_env[is_standing_env] = False

    return vel_commands_b, heading_commands, is_heading_env, is_standing_env


def update_commands(
    vel_commands_b: torch.Tensor,
    heading_commands: torch.Tensor,
    heading: torch.Tensor,
    is_heading_env: torch.Tensor,
    is_standing_env: torch.Tensor,
    heading_control_stiffness: float,
    ang_vel_z: tuple[float, float],
) -> torch.Tensor:
    heading_error = math_utils.wrap_to_pi(heading_commands - heading)

    new_vel_commands_b = vel_commands_b.clone()
    new_vel_commands_b[is_heading_env, 2] = torch.clip(
        heading_control_stiffness * heading_error[is_heading_env],
        min=ang_vel_z[0],
        max=ang_vel_z[1],
    )
    new_vel_commands_b[is_standing_env, :] = 0.0
    return new_vel_commands_b


def resolve_xy_velocity_to_arrow(
    xy_velocity_b: torch.Tensor,
    scale: tuple[float, float, float],
    device: torch.device,
    base_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # scale
    arrow_scale = torch.tensor(scale, device=device).repeat(xy_velocity_b.shape[0], 1)
    arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity_b, dim=1) * 3.0

    # diretion
    heading_angle = torch.atan2(xy_velocity_b[:, 1], xy_velocity_b[:, 0])
    zeros = torch.zeros_like(heading_angle)
    arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

    # convert everything back from base to world frame
    arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
    return arrow_scale, arrow_quat


class G1Env(DirectRLEnv):
    cfg: G1EnvCfg

    def __init__(self, cfg: G1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._setup_keyboard()

        # Robot joint idxs
        self._joint_dof_idx, self._joint_dof_names = self.robot.find_joints(".*")
        self._torso_joint_idx, _ = self.robot.find_joints("torso_joint")
        self._finger_joint_idx, _ = self.robot.find_joints(
            [
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ]
        )
        self._arm_joint_idx, _ = self.robot.find_joints(
            [
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ]
        )
        self._hip_joint_idx, _ = self.robot.find_joints(
            [".*_hip_yaw_joint", ".*_hip_roll_joint"]
        )
        self._ankle_joint_idx, _ = self.robot.find_joints(
            [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
        )
        print("!" * 100)
        print(f"len(self._joint_dof_idx): {len(self._joint_dof_idx)}")
        print(f"self._joint_dof_names: {self._joint_dof_names}")
        print(f"len(self._torso_joint_idx): {len(self._torso_joint_idx)}")
        print(f"len(self._finger_joint_idx): {len(self._finger_joint_idx)}")
        print(f"len(self._arm_joint_idx): {len(self._arm_joint_idx)}")
        print(f"len(self._hip_joint_idx): {len(self._hip_joint_idx)}")
        print(f"len(self._ankle_joint_idx): {len(self._ankle_joint_idx)}")
        print("!" * 100)

        # Robot link idxs
        self._link_idx, self._link_names = self.robot.find_bodies(".*")
        self._ankle_link_idx, _ = self.robot.find_bodies(".*_ankle_roll_link")
        print("!" * 100)
        print(f"len(self._link_idx): {len(self._link_idx)}")
        print(f"self._link_names: {self._link_names}")
        print(f"len(self._ankle_link_idx): {len(self._ankle_link_idx)}")
        print("!" * 100)

        # Contact sensor link idxs
        self._contact_link_idx, _ = self.contact_sensor.find_bodies(".*")
        self._contact_torso_link_idx, _ = self.contact_sensor.find_bodies("torso_link")
        self._contact_ankle_link_idx, _ = self.contact_sensor.find_bodies(
            ".*_ankle_roll_link"
        )
        # self._contact_thigh_link_idx, _ = self.contact_sensor.find_bodies(".*THIGH")
        print("!" * 100)
        print(f"len(self._contact_link_idx): {len(self._contact_link_idx)}")
        print(f"len(self._contact_torso_link_idx): {len(self._contact_torso_link_idx)}")
        print(f"len(self._contact_ankle_link_idx): {len(self._contact_ankle_link_idx)}")
        print("!" * 100)

        # State
        self.actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )
        self.prev_actions = torch.zeros(
            self.num_envs, self.cfg.action_space, device=self.device
        )

        # Commands
        (
            self.vel_commands_b,
            self.heading_commands,
            self.is_heading_env,
            self.is_standing_env,
        ) = (
            torch.zeros(self.num_envs, 3, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
            torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in REWARD_NAMES
        }

        # Debug
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        # add articulation to scene
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # add contact sensor to scene
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        self.cfg.light.func("/World/Light", self.cfg.light)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.cfg.action_scale * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        pass

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         # TODO: add observations
        #     ),
        #     dim=-1,
        # )
        obs = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Intermediate values
        joint_pos = self.robot.data.joint_pos
        default_joint_pos = self.robot.data.default_joint_pos
        joint_deviation = (joint_pos - default_joint_pos).abs()

        joint_pos_max = self.robot.data.soft_joint_pos_limits[:, :, 1]
        joint_pos_min = self.robot.data.soft_joint_pos_limits[:, :, 0]
        under_min = (joint_pos_min - joint_pos).clip(min=0.0)
        over_max = (joint_pos - joint_pos_max).clip(min=0.0)

        # net_forces_w_history.shape == (num_envs, history_length, num_bodies, 3)
        contacts = (
            self.contact_sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1).values
            > 1.0
        )

        # feet air time positive biped
        air_time = self.contact_sensor.data.current_air_time[
            :, self._contact_ankle_link_idx
        ]
        contact_time = self.contact_sensor.data.current_contact_time[
            :, self._contact_ankle_link_idx
        ]
        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)
        single_stance = in_contact.int().sum(dim=1) == 1

        # Robot's linear velocity in gravity-aligned robot frame (z up, but x/y aligned with robot base frame)
        robot_lin_vel_rotated = quat_rotate_inverse(
            yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w[:, :3]
        )

        # fmt: off
        rewards = {
            # velocity_env_cfg.py
            "lin_vel_z_l2": self.robot.data.root_lin_vel_b[:, 2].square(),  # (don't move up/down)
            "ang_vel_xy_l2": self.robot.data.root_ang_vel_b[:, :2].square().sum(dim=1), # (don't tip sideways or forwards)
            "dof_torques_l2": self.robot.data.applied_torque[:, self._joint_dof_idx].square().sum(dim=1),  # (don't apply too much torque)
            "dof_acc_l2": self.robot.data.joint_acc[:, self._joint_dof_idx].square().sum(dim=1),  # (don't apply too much acceleration)
            "action_rate_l2": (self.actions - self.prev_actions).square().sum(dim=1),  # (don't change actions too quickly)
            # "undesired_contacts": contacts[:, self._contact_thigh_link_idx].sum(dim=1),  # (don't contact thighs)
            "flat_orientation_l2": self.robot.data.projected_gravity_b[:, :2].square().sum(dim=1),  # (don't tip sideways or forwards)

            # rough_env_cfg.py
            "termination_penalty": self.reset_terminated.float(),  # (don't terminate) This works because _get_dones() is called before _get_rewards()
            "track_lin_vel_xy_exp": torch.exp(-(self.vel_commands_b[:, :2] - robot_lin_vel_rotated[:, :2]).square().sum(dim=1) / 0.5**2),  # (track the commanded lin_vel_xy)
            "track_ang_vel_z_exp": torch.exp(-(self.vel_commands_b[:, 2] - self.robot.data.root_ang_vel_w[:, 2]).square() / 0.5**2),  #  (track the commanded ang_vel_z)
            "feet_air_time": torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0).min(dim=1).values.clamp(max=0.4) * (self.vel_commands_b[:, :2].norm(dim=1) > 0.1),  # (Promote stable single-stance gait)
            "feet_slide": (self.robot.data.body_lin_vel_w[:, self._ankle_link_idx, :2].norm(dim=-1) * contacts[:, self._contact_ankle_link_idx]).sum(dim=1),  # (don't slide feet)
            "dof_pos_limits_ankle": (under_min + over_max)[:, self._ankle_joint_idx].sum(dim=1),  # (don't exceed dof pos limits of ankles)
            "joint_deviation_hip": joint_deviation[:, self._hip_joint_idx].sum(dim=1),  # (don't deviate from default hip positions)
            "joint_deviation_arms": joint_deviation[:, self._arm_joint_idx].sum(dim=1),  # (don't deviate from default arm positions)
            "joint_deviation_fingers": joint_deviation[:, self._finger_joint_idx].sum(dim=1),  # (don't deviate from default finger positions)
            "joint_deviation_torso": joint_deviation[:, self._torso_joint_idx].sum(dim=1),  # (don't deviate from default torso position)
        }

        reward_weights = {
            # velocity_env_cfg.py
            "lin_vel_z_l2": -0.2,
            "ang_vel_xy_l2": -0.05,
            "dof_torques_l2": -2.0e-6,
            "dof_acc_l2": -1.0e-7,
            "action_rate_l2": -0.005,
            # "undesired_contacts": -1.0,
            "flat_orientation_l2": -1.0,

            # rough_env_cfg.py
            "termination_penalty": -200.0,
            "track_lin_vel_xy_exp": 1.0,
            "track_ang_vel_z_exp": 1.0,
            "feet_air_time": 0.75,
            "feet_slide": -0.1,
            "dof_pos_limits_ankle": -1.0,
            "joint_deviation_hip": -0.1,
            "joint_deviation_arms": -0.1,
            "joint_deviation_fingers": -0.05,
            "joint_deviation_torso": -0.1,
        }
        # fmt: on

        assert set(rewards.keys()) == set(REWARD_NAMES), (
            f"Rewards and reward weights do not match: {rewards.keys()} vs {REWARD_NAMES}\nOnly in rewards: {set(rewards.keys()) - set(REWARD_NAMES)}\nOnly in reward_weights: {set(reward_weights.keys()) - set(REWARD_NAMES)}"
        )
        assert set(reward_weights.keys()) == set(REWARD_NAMES), (
            f"Rewards and reward weights do not match: {reward_weights.keys()} vs {REWARD_NAMES}\nOnly in rewards: {set(rewards.keys()) - set(REWARD_NAMES)}\nOnly in reward_weights: {set(reward_weights.keys()) - set(REWARD_NAMES)}"
        )

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # HACK: The typical step looks like:
        # 1. _pre_physics_step() (comput actions)
        # 2. _apply_action() (apply actions)
        # 3. physics_step() (simulate)
        # 4. _compute_intermediate_values() (compute intermediate values)
        # 5. _get_dones() (compute done/time_out)
        # 6. _get_rewards() (compute rewards)
        # 7. _get_observations() (compute observations)
        # In this pipeline, we want to update some internal state after each physics step, but before the observation step.
        self._update_internal_state()
        return reward

    def _update_internal_state(self):
        self.vel_commands_b[:] = update_commands(
            vel_commands_b=self.vel_commands_b,
            heading_commands=self.heading_commands,
            heading=self.robot.data.heading_w,
            is_heading_env=self.is_heading_env,
            is_standing_env=self.is_standing_env,
            heading_control_stiffness=self.cfg.base_velocity_command.heading_control_stiffness,
            ang_vel_z=self.cfg.base_velocity_command.ranges.ang_vel_z,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # net_forces_w_history.shape == (num_envs, history_length, num_bodies, 3)
        contacts = (
            self.contact_sensor.data.net_forces_w_history.norm(dim=-1).max(dim=1).values
            > 1.0
        )
        any_torso_contacts = contacts[:, self._contact_torso_link_idx].any(dim=1)
        died = any_torso_contacts

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset state
        self.actions[env_ids] = torch.zeros(
            len(env_ids), self.cfg.action_space, device=self.device
        )
        self.prev_actions[env_ids] = torch.zeros(
            len(env_ids), self.cfg.action_space, device=self.device
        )

        (
            self.vel_commands_b[env_ids],
            self.heading_commands[env_ids],
            self.is_heading_env[env_ids],
            self.is_standing_env[env_ids],
        ) = sample_commands(
            num_envs=self.num_envs,
            device=self.device,
            lin_vel_x=self.cfg.base_velocity_command.ranges.lin_vel_x,
            lin_vel_y=self.cfg.base_velocity_command.ranges.lin_vel_y,
            ang_vel_z=self.cfg.base_velocity_command.ranges.ang_vel_z,
            heading=self.cfg.base_velocity_command.ranges.heading,
            rel_heading_envs=self.cfg.base_velocity_command.rel_heading_envs,
            rel_standing_envs=self.cfg.base_velocity_command.rel_standing_envs,
        )
        self.vel_commands_b[:] = update_commands(
            vel_commands_b=self.vel_commands_b,
            heading_commands=self.heading_commands,
            heading=self.robot.data.heading_w,
            is_heading_env=self.is_heading_env,
            is_standing_env=self.is_standing_env,
            heading_control_stiffness=self.cfg.base_velocity_command.heading_control_stiffness,
            ang_vel_z=self.cfg.base_velocity_command.ranges.ang_vel_z,
        )

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"] = extras

        self._compute_intermediate_values()

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "command_vel_visualizer"):
                self.command_vel_visualizer = VisualizationMarkers(
                    self.cfg.command_vel_visualizer_cfg
                )
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            if not hasattr(self, "pose_visualizer"):
                self.pose_visualizer = VisualizationMarkers(
                    self.cfg.pose_visualizer_cfg
                )

            # set their visibility to true
            self.command_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "command_vel_visualizer"):
                self.command_vel_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)
            if hasattr(self, "pose_visualizer"):
                self.pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Make sure the robot is initialized
        if not self.robot.is_initialized:
            return

        # Place marker above the robot
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # Convert the velocity command to an arrow
        vel_command_arrow_scale, vel_command_arrow_quat = resolve_xy_velocity_to_arrow(
            xy_velocity_b=self.vel_commands_b[:, :2],
            scale=self.command_vel_visualizer.cfg.markers["arrow"].scale,
            device=self.device,
            base_quat_w=self.robot.data.root_quat_w,
        )
        vel_arrow_scale, vel_arrow_quat = resolve_xy_velocity_to_arrow(
            xy_velocity_b=self.robot.data.root_lin_vel_b[:, :2],
            scale=self.current_vel_visualizer.cfg.markers["arrow"].scale,
            device=self.device,
            base_quat_w=self.robot.data.root_quat_w,
        )

        # update the markers
        self.command_vel_visualizer.visualize(
            translations=base_pos_w,
            orientations=vel_command_arrow_quat,
            scales=vel_command_arrow_scale,
        )
        self.current_vel_visualizer.visualize(
            translations=base_pos_w, orientations=vel_arrow_quat, scales=vel_arrow_scale
        )
        self.pose_visualizer.visualize(
            translations=base_pos_w,
            orientations=self.robot.data.root_quat_w,
            scales=torch.tensor([0.1, 0.1, 0.1], device=self.device)
            .unsqueeze(0)
            .repeat_interleave(self.num_envs, dim=0),
        )

    def _setup_keyboard(self):
        import carb
        from isaaclab.devices.keyboard.general_keyboard import (
            GeneralKeyboard,
            KeyboardCommand,
        )

        # kbc = keyboard callback
        self.keyboard = GeneralKeyboard(
            commands=[
                KeyboardCommand(
                    key=carb.input.KeyboardInput.R, func=self._reset_kbc, args=[]
                ),
            ]
        )

    def _reset_kbc(self):
        print("In reset_kbc")
        self._reset_idx(env_ids=None)
