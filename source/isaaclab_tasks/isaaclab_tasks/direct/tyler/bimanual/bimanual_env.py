# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import torch.nn as nn
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    SPHERE_MARKER_CFG,
    FRAME_MARKER_CFG,
    CYLINDER_MARKER_CFG,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, LightCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.bimanual import BIMANUAL_CFG
from isaaclab_tasks.direct.tyler.bimanual.utils.torch_utils import sample_uniform_tensor
from isaaclab_tasks.direct.tyler.bimanual.utils.constants import NUM_XYZ, NUM_QUAT
from isaaclab_tasks.direct.tyler.bimanual.utils.color_constants import (
    RED_RGB,
    GREEN_RGB,
    BLUE_RGB,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.robot_constants import (
    INDEX_FINGERTIP_IDX,
    MIDDLE_FINGERTIP_IDX,
    RING_FINGERTIP_IDX,
    THUMB_FINGERTIP_IDX,
    RIGHT_FINGERTIP_LINK_NAMES,
    LEFT_FINGERTIP_LINK_NAMES,
    RIGHT_PALM_LINK_NAME,
    LEFT_PALM_LINK_NAME,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.table_constants import (
    TABLE_X,
    TABLE_Y,
    TABLE_Z,
    TABLE_QX,
    TABLE_QY,
    TABLE_QZ,
    TABLE_QW,
    TABLE_LENGTH_Z,
)
import wandb

SIM_DT = 0.005
physics_material = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)

ENV_REGEX_NS = "/World/envs/env_.*"


@configclass
class BimanualEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 46
    observation_space = 136
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
    robot: ArticulationCfg = BIMANUAL_CFG.replace(prim_path=f"{ENV_REGEX_NS}/Robot")

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/starbucks_bottle/usd/starbucks_bottle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1, 1, 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                float(TABLE_X) + 0.1,
                float(TABLE_Y),
                float(TABLE_Z) + TABLE_LENGTH_Z / 2 + 0.05,
            ),
            rot=(float(TABLE_QW), float(TABLE_QX), float(TABLE_QY), float(TABLE_QZ)),
        ),
    )

    # goal object
    goal_object: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/GoalObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/green_starbucks_bottle/usd/starbucks_bottle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1, 1, 1),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=GREEN_RGB,
                roughness=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                float(TABLE_X),
                float(TABLE_Y),
                float(TABLE_Z) + TABLE_LENGTH_Z / 2 + 0.1,
            ),
            rot=(float(TABLE_QW), float(TABLE_QX), float(TABLE_QY), float(TABLE_QZ)),
        ),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path=f"{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/table/usd/table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # make it static
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1, 1, 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(TABLE_X), float(TABLE_Y), float(TABLE_Z)),
            rot=(float(TABLE_QW), float(TABLE_QX), float(TABLE_QY), float(TABLE_QZ)),
        ),
    )

    # contact sensor
    contact_sensor = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=SIM_DT,
    )

    # light
    light: LightCfg = DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )

    pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    """The configuration for the pose visualization marker. Defaults to FRAME_MARKER_CFG."""
    pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)

    right_fingertip_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/right_fingertip"
    )
    right_fingertip_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=RED_RGB)
    left_fingertip_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/left_fingertip"
    )
    left_fingertip_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=RED_RGB)

    progress_visualizer: VisualizationMarkersCfg = CYLINDER_MARKER_CFG.replace(
        prim_path="/Visuals/Command/progress"
    )
    progress_visualizer.markers[
        "cylinder"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=BLUE_RGB)
    progress_full_visualizer: VisualizationMarkersCfg = CYLINDER_MARKER_CFG.replace(
        prim_path="/Visuals/Command/progress_full"
    )
    progress_full_visualizer.markers[
        "cylinder"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN_RGB)


REWARD_NAMES = [
    "right_index_fingertip_to_object_dist",
    "left_index_fingertip_to_object_dist",
    "object_to_goal_dist",
]


class AverageMeter(nn.Module):
    def __init__(self, in_shape: int = 1, max_size: int = 1000) -> None:
        super().__init__()
        self.max_size = max_size

        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
        assert len(values.shape) == 1, f"values.shape: {values.shape}"
        size = values.size()[0]
        if size == 0:
            return

        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self) -> None:
        self.current_size = 0
        self.mean.fill_(0.0)

    def __len__(self) -> int:
        return self.current_size

    def get_mean(self) -> np.ndarray:
        return self.mean.squeeze(0).cpu().numpy()


class BimanualEnv(DirectRLEnv):
    cfg: BimanualEnvCfg

    def __init__(self, cfg: BimanualEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_keyboard()
        self._setup_robot_idxs()

        # Action offset
        self.action_offset = self.robot.data.default_joint_pos[:, self._joint_dof_idxs]
        assert self.action_offset.shape == (self.num_envs, self.cfg.action_space), (
            f"self.action_offset.shape: {self.action_offset.shape} != (self.num_envs, self.cfg.action_space): {(self.num_envs, self.cfg.action_space)}"
        )

        # State
        self._reset_state(env_ids=None)

        # Logging
        self.wandb_dict = {}

        self._update_metrics(env_ids=None)

        # Debug
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_robot_idxs(self):
        # Robot joint idxs
        self._joint_dof_idxs, self._joint_dof_names = self.robot.find_joints(".*")
        print("!" * 100)
        print(f"len(self._joint_dof_idxs): {len(self._joint_dof_idxs)}")
        print(f"self._joint_dof_names: {self._joint_dof_names}")
        print("!" * 100)

        # Robot link idxs
        self._link_idxs, self._link_names = self.robot.find_bodies(".*")
        self._right_fingertip_link_idxs, self._right_fingertip_link_names = (
            self.robot.find_bodies(RIGHT_FINGERTIP_LINK_NAMES)
        )
        self._left_fingertip_link_idxs, self._left_fingertip_link_names = (
            self.robot.find_bodies(LEFT_FINGERTIP_LINK_NAMES)
        )
        self._right_palm_link_idxs, self._right_palm_link_names = (
            self.robot.find_bodies(RIGHT_PALM_LINK_NAME)
        )
        self._left_palm_link_idxs, self._left_palm_link_names = self.robot.find_bodies(
            LEFT_PALM_LINK_NAME
        )
        print("!" * 100)
        print(f"len(self._link_idxs): {len(self._link_idxs)}")
        print(f"self._link_names: {self._link_names}")
        print("!" * 100)

        # Contact sensor link idxs
        self._contact_link_idxs, self._contact_link_names = (
            self.contact_sensor.find_bodies(".*")
        )
        print("!" * 100)
        print(f"len(self._contact_link_idxs): {len(self._contact_link_idxs)}")
        print("!" * 100)

    def _update_metrics(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        if not hasattr(self, "FIRST_METRIC_UPDATE"):
            self.FIRST_METRIC_UPDATE = True
            self.reward_metric = AverageMeter().to(self.device)
            self.individual_reward_metrics = {
                reward_name: AverageMeter().to(self.device)
                for reward_name in REWARD_NAMES
            }
            self.individual_weighted_reward_metrics = {
                reward_name: AverageMeter().to(self.device)
                for reward_name in REWARD_NAMES
            }
            self.episode_length_metric = AverageMeter().to(self.device)
        else:
            self.reward_metric.update(self.aggregated_reward_buf[env_ids])
            for reward_name, metric in self.individual_reward_metrics.items():
                metric.update(
                    self.individual_aggregated_reward_bufs[reward_name][env_ids]
                )
            for reward_name, metric in self.individual_weighted_reward_metrics.items():
                metric.update(
                    self.individual_weighted_aggregated_reward_bufs[reward_name][
                        env_ids
                    ]
                )
            self.episode_length_metric.update(self.episode_length_buf[env_ids])

    def _setup_scene(self):
        # add articulation to scene
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # add object to scene
        self.object = RigidObject(self.cfg.object)
        self.scene.rigid_objects["object"] = self.object

        # add goal object to scene
        self.goal_object = RigidObject(self.cfg.goal_object)
        self.scene.rigid_objects["goal_object"] = self.goal_object

        # add table to scene
        self.table = RigidObject(self.cfg.table)
        self.scene.rigid_objects["table"] = self.table

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
        self.prev_raw_actions = self.raw_actions.clone()
        self.raw_actions = actions.clone()
        assert self.raw_actions.shape == self.prev_raw_actions.shape, (
            f"self.raw_actions.shape: {self.raw_actions.shape} != self.prev_raw_actions.shape: {self.prev_raw_actions.shape}"
        )
        assert self.raw_actions.shape == self.action_offset.shape, (
            f"self.raw_actions.shape: {self.raw_actions.shape} != self.action_offset.shape: {self.action_offset.shape}"
        )
        assert self.raw_actions.shape == (self.num_envs, self.cfg.action_space), (
            f"self.raw_actions.shape: {self.raw_actions.shape} != (self.num_envs, self.cfg.action_space): {(self.num_envs, self.cfg.action_space)}"
        )
        assert torch.all(torch.le(self.raw_actions, 1.0)) and torch.all(
            torch.ge(self.raw_actions, -1.0)
        ), f"self.raw_actions: {self.raw_actions}"

    def _apply_action(self):
        position_targets = self.cfg.action_scale * self.raw_actions + self.action_offset

        DISABLE_ACTIONS = False  # Set to True to debug actions
        if DISABLE_ACTIONS:
            position_targets[:] = 0.0

        self.robot.set_joint_position_target(
            position_targets, joint_ids=self._joint_dof_idxs
        )

    def _compute_intermediate_values(self):
        pass

    def _get_observations(self) -> dict:
        obs_dict = {
            "q": self.robot.data.joint_pos,
            "qd": self.robot.data.joint_vel,
            "right_fingertip_positions": (
                self.right_fingertip_positions - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, -1),
            "left_fingertip_positions": (
                self.left_fingertip_positions - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, -1),
            "right_palm_position": self.right_palm_position - self.scene.env_origins,
            "left_palm_position": self.left_palm_position - self.scene.env_origins,
            "object_position": self.object_position - self.scene.env_origins,
            "goal_object_position": self.goal_object_position - self.scene.env_origins,
            "object_orientation": self.object_orientation,
            "goal_object_orientation": self.goal_object_orientation,
        }

        for k, v in obs_dict.items():
            if v.ndim != 2:
                print(f"{k}: {v.shape} (WRONG)")

        obs = torch.cat(
            [obs_dict[key] for key in obs_dict],
            dim=-1,
        )

        ZERO_OBS = False  # Set to True to debug
        if ZERO_OBS:
            obs = torch.zeros(
                self.num_envs, self.cfg.observation_space, device=self.device
            )

        assert obs.shape == (self.num_envs, self.cfg.observation_space), (
            f"obs.shape: {obs.shape} != (self.num_envs, self.cfg.observation_space): {(self.num_envs, self.cfg.observation_space)}"
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # fmt: off
        self.individual_reward_bufs = {
            "right_index_fingertip_to_object_dist": -(self.right_index_fingertip_position - self.object_position).norm(dim=-1, p=2),
            "left_index_fingertip_to_object_dist": -(self.left_index_fingertip_position - self.object_position).norm(dim=-1, p=2),
            "object_to_goal_dist": -(self.object_position - self.goal_object_position).norm(dim=-1, p=2),
        }
        # fmt: on
        assert set(self.individual_reward_bufs.keys()) == set(REWARD_NAMES), (
            f"Individual reward buffers and reward names do not match: {self.individual_reward_bufs.keys()} vs {REWARD_NAMES}\nOnly in individual reward buffers: {set(self.individual_reward_bufs.keys()) - set(REWARD_NAMES)}\nOnly in reward names: {set(REWARD_NAMES) - set(self.individual_reward_bufs.keys())}"
        )

        if not hasattr(self, "reward_weights"):
            self.individual_reward_weights = {
                "right_index_fingertip_to_object_dist": 1.0,
                "left_index_fingertip_to_object_dist": 1.0,
                "object_to_goal_dist": 3.0,
            }
            assert set(self.individual_reward_weights.keys()) == set(REWARD_NAMES), (
                f"Individual reward weights and reward names do not match: {self.individual_reward_weights.keys()} vs {REWARD_NAMES}\nOnly in individual reward weights: {set(self.individual_reward_weights.keys()) - set(REWARD_NAMES)}\nOnly in reward names: {set(REWARD_NAMES) - set(self.individual_reward_weights.keys())}"
            )

            self.reward_weights = torch.tensor(
                [self.individual_reward_weights[name] for name in REWARD_NAMES],
                device=self.device,
            ).reshape(1, -1)

        self.reward_matrix = torch.stack(
            [self.individual_reward_bufs[name] for name in REWARD_NAMES], dim=1
        )
        assert self.reward_matrix.shape == (self.num_envs, len(REWARD_NAMES)), (
            f"reward_matrix.shape: {self.reward_matrix.shape} != (self.num_envs, len(REWARD_NAMES)): {(self.num_envs, len(REWARD_NAMES))}"
        )

        self.weighted_reward_matrix = self.reward_matrix * self.reward_weights
        total_reward = self.weighted_reward_matrix.sum(dim=1)

        # HACK: Set the self.reward_buf to the total_reward now so that _end_of_step() can use it
        self.reward_buf = total_reward

        # HACK: The typical step looks like:
        # 1. _pre_physics_step() (comput actions)
        # 2. _apply_action() (apply actions)
        # 3. physics_step() (simulate)
        # 4. _compute_intermediate_values() (compute intermediate values)
        # 5. _get_dones() (compute done/time_out)
        # 6. _get_rewards() (compute rewards)
        # 7. _get_observations() (compute observations)
        # In this pipeline, we add _end_of_step() to update some internal state after each physics step, but before the observation step.
        self._end_of_step()
        return total_reward

    #### END OF STEP START  ####
    def _end_of_step(self):
        # Update aggregated rewards
        self.aggregated_reward_buf += self.reward_buf
        for reward_name in REWARD_NAMES:
            self.individual_aggregated_reward_bufs[reward_name] += (
                self.individual_reward_bufs[reward_name]
            )
            self.individual_weighted_aggregated_reward_bufs[reward_name] += (
                self.individual_reward_bufs[reward_name]
                * self.individual_reward_weights[reward_name]
            )

        self.populate_wandb_dict()
        self.log_wandb_dict()

    def populate_wandb_dict(self) -> None:
        if self.common_step_counter % 10 != 0:
            return

        self.wandb_dict.update(
            {
                "common_step_counter": self.common_step_counter,
                "episode_length_buf (mean)": self.episode_length_buf.float()
                .mean()
                .item(),
            }
        )

        self.wandb_dict.update(
            {
                "metrics/mean/reward": self.reward_metric.get_mean().item(),
                "metrics/mean/episode_length": self.episode_length_metric.get_mean().item(),
            }
        )
        self.wandb_dict.update(
            {
                f"metrics/mean/{reward_name}": metric.get_mean().item()
                for reward_name, metric in self.individual_reward_metrics.items()
            }
        )
        self.wandb_dict.update(
            {
                f"metrics/mean/weighted_{reward_name}": metric.get_mean().item()
                for reward_name, metric in self.individual_weighted_reward_metrics.items()
            }
        )

    def log_wandb_dict(self) -> None:
        if wandb.run is None:
            return

        # Skip if empty
        if len(self.wandb_dict) == 0:
            return

        wandb.log(self.wandb_dict)
        self.wandb_dict = {}

    #### END OF STEP END  ####

    #### DONES START ####
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        """
        Reset Conditions:
        - Time out: episode_length_buf >= max_episode_length - 1
        - Died: never
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    #### DONES END ####

    #### RESET START ####
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # Update metrics
        self._update_metrics(env_ids)
        self._reset_state(env_ids)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset robot
        root_state = self.robot.data.default_root_state[env_ids].clone()
        default_position = root_state[:, :3] + self.scene.env_origins[env_ids]
        default_orientation = root_state[:, 3:7]
        default_velocity = root_state[:, 7:13]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos *= math_utils.sample_uniform(
            *(0.5, 1.5), joint_pos.shape, joint_pos.device
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_vel *= math_utils.sample_uniform(
            *(0.0, 0.0), joint_vel.shape, joint_vel.device
        )

        joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids].clone()
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel_limits = self.robot.data.soft_joint_vel_limits[env_ids].clone()
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        self.robot.write_root_pose_to_sim(
            torch.cat([default_position, default_orientation], dim=-1), env_ids
        )
        self.robot.write_root_velocity_to_sim(default_velocity, env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, None, env_ids)
        self.robot.write_joint_velocity_to_sim(joint_vel, None, env_ids)

        self.object.write_root_pose_to_sim(
            self._sample_initial_object_pose(env_ids), env_ids
        )
        self.goal_object.write_root_pose_to_sim(
            self._sample_final_object_pose(env_ids), env_ids
        )

        self._compute_intermediate_values()

    def _reset_state(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        if not hasattr(self, "FIRST_RESET_COMPLETED"):
            self.FIRST_RESET_COMPLETED = False

            self.raw_actions = torch.zeros(
                self.num_envs, self.cfg.action_space, device=self.device
            )
            self.prev_raw_actions = torch.zeros(
                self.num_envs, self.cfg.action_space, device=self.device
            )

            self.aggregated_reward_buf = torch.zeros(self.num_envs, device=self.device)
            self.individual_aggregated_reward_bufs = {
                reward_name: torch.zeros(self.num_envs, device=self.device)
                for reward_name in REWARD_NAMES
            }
            self.individual_weighted_aggregated_reward_bufs = {
                reward_name: torch.zeros(self.num_envs, device=self.device)
                for reward_name in REWARD_NAMES
            }
        else:
            self.raw_actions[env_ids] = torch.zeros(
                len(env_ids), self.cfg.action_space, device=self.device
            )
            self.prev_raw_actions[env_ids] = torch.zeros(
                len(env_ids), self.cfg.action_space, device=self.device
            )

            self.aggregated_reward_buf[env_ids] = 0
            for reward_name in REWARD_NAMES:
                self.individual_aggregated_reward_bufs[reward_name][env_ids] = 0
                self.individual_weighted_aggregated_reward_bufs[reward_name][
                    env_ids
                ] = 0

    def _sample_initial_object_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        position = self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.4, -0.5, 0.05], device=self.device),
            high=torch.tensor([0.4, 0.5, 0.06], device=self.device),
            N=len(env_ids),
        )
        orientation = (
            torch.tensor([TABLE_QW, TABLE_QX, TABLE_QY, TABLE_QZ], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(len(env_ids), dim=0)
        )
        return torch.cat([position, orientation], dim=-1).float()

    def _sample_final_object_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        position = self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.4, -0.5, 0.05 + 0.01], device=self.device),
            high=torch.tensor([0.4, 0.5, 0.06 + 0.5], device=self.device),
            N=len(env_ids),
        )
        orientation = (
            torch.tensor([TABLE_QW, TABLE_QX, TABLE_QY, TABLE_QZ], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(len(env_ids), dim=0)
        )
        return torch.cat([position, orientation], dim=-1).float()

    #### RESET END ####

    #### DEBUG START ####
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "pose_visualizer"):
                self.pose_visualizer = VisualizationMarkers(self.cfg.pose_visualizer)
            if not hasattr(self, "right_fingertip_visualizer"):
                self.right_fingertip_visualizer = VisualizationMarkers(
                    self.cfg.right_fingertip_visualizer
                )
            if not hasattr(self, "left_fingertip_visualizer"):
                self.left_fingertip_visualizer = VisualizationMarkers(
                    self.cfg.left_fingertip_visualizer
                )
            if not hasattr(self, "progress_visualizer"):
                self.progress_visualizer = VisualizationMarkers(
                    self.cfg.progress_visualizer
                )
            if not hasattr(self, "progress_full_visualizer"):
                self.progress_full_visualizer = VisualizationMarkers(
                    self.cfg.progress_full_visualizer
                )

            # set their visibility to true
            self.pose_visualizer.set_visibility(True)
            self.right_fingertip_visualizer.set_visibility(True)
            self.left_fingertip_visualizer.set_visibility(True)
            self.progress_visualizer.set_visibility(True)
            self.progress_full_visualizer.set_visibility(True)
        else:
            if hasattr(self, "pose_visualizer"):
                self.pose_visualizer.set_visibility(False)
            if hasattr(self, "right_fingertip_visualizer"):
                self.right_fingertip_visualizer.set_visibility(False)
            if hasattr(self, "left_fingertip_visualizer"):
                self.left_fingertip_visualizer.set_visibility(False)
            if hasattr(self, "progress_visualizer"):
                self.progress_visualizer.set_visibility(False)
            if hasattr(self, "progress_full_visualizer"):
                self.progress_full_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Make sure the robot is initialized
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        self.pose_visualizer.visualize(
            translations=base_pos_w,
            orientations=self.robot.data.root_quat_w,
            scales=torch.tensor([0.2, 0.2, 0.2], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )

        self.right_fingertip_visualizer.visualize(
            translations=self.right_index_fingertip_position,
            scales=torch.tensor([2.0, 2.0, 2.0], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.left_fingertip_visualizer.visualize(
            translations=self.left_index_fingertip_position,
            scales=torch.tensor([2.0, 2.0, 2.0], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )

        # Growing bar to show progress
        # Full bar to show max progress
        # Make growing bar thicker
        progress_frac = self.episode_length_buf / self.max_episode_length
        progress_full = torch.ones_like(progress_frac)
        progress_pos = self.robot_position + torch.tensor(
            [0.0, 0.0, 1.1], device=self.device
        ).unsqueeze(dim=0)
        MAX_SCALE = 50
        progress_scale = torch.ones_like(progress_pos) * 0.4
        progress_scale[:, 1] = progress_frac * MAX_SCALE
        self.progress_visualizer.visualize(
            translations=progress_pos,
            scales=progress_scale,
        )
        progress_scale_full = torch.ones_like(progress_pos) * 0.2
        progress_scale_full[:, 1] = progress_full * MAX_SCALE
        self.progress_full_visualizer.visualize(
            translations=progress_pos,
            scales=progress_scale_full,
        )

    #### DEBUG END ####

    #### KEYBOARD START ####
    def _setup_keyboard(self):
        try:
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
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.B,
                        func=self._breakpoint_kbc,
                        args=[],
                    ),
                ]
            )
        except AttributeError as e:
            print("~" * 100)
            print(f"Error importing keyboard: {e}")
            print("Keyboard not available, likely because we are in headless mode.")
            print("~" * 100)
            return

    def _reset_kbc(self):
        print("In reset_kbc")
        self._reset_idx(env_ids=None)

    def _breakpoint_kbc(self):
        print("In breakpoint_kbc")
        breakpoint()

    #### KEYBOARD END ####

    #### TENSOR SLICE PROPERTIES START ####
    @property
    def table_position(self) -> torch.Tensor:
        assert self.table.data.body_pos_w.shape == (self.num_envs, 1, NUM_XYZ), (
            f"Table position shape: {self.table.data.body_pos_w.shape}"
        )
        return self.table.data.body_pos_w[:, 0]

    @property
    def object_position(self) -> torch.Tensor:
        assert self.object.data.body_pos_w.shape == (self.num_envs, 1, NUM_XYZ), (
            f"Object position shape: {self.object.data.body_pos_w.shape}"
        )
        return self.object.data.body_pos_w[:, 0]

    @property
    def object_orientation(self) -> torch.Tensor:
        assert self.object.data.body_quat_w.shape == (self.num_envs, 1, NUM_QUAT), (
            f"Object orientation shape: {self.object.data.body_quat_w.shape}"
        )
        return self.object.data.body_quat_w[:, 0]

    @property
    def goal_object_position(self) -> torch.Tensor:
        assert self.goal_object.data.body_pos_w.shape == (self.num_envs, 1, NUM_XYZ), (
            f"Goal object position shape: {self.goal_object.data.body_pos_w.shape}"
        )
        return self.goal_object.data.body_pos_w[:, 0]

    @property
    def goal_object_orientation(self) -> torch.Tensor:
        assert self.goal_object.data.body_quat_w.shape == (
            self.num_envs,
            1,
            NUM_QUAT,
        ), f"Goal object orientation shape: {self.goal_object.data.body_quat_w.shape}"
        return self.goal_object.data.body_quat_w[:, 0]

    @property
    def robot_position(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, 0]

    @property
    def right_palm_position(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._right_palm_link_idxs].squeeze(dim=1)

    @property
    def left_palm_position(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._left_palm_link_idxs].squeeze(dim=1)

    @property
    def right_fingertip_positions(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._right_fingertip_link_idxs]

    @property
    def left_fingertip_positions(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._left_fingertip_link_idxs]

    @property
    def right_index_fingertip_position(self) -> torch.Tensor:
        return self.right_fingertip_positions[:, INDEX_FINGERTIP_IDX]

    @property
    def right_middle_fingertip_position(self) -> torch.Tensor:
        return self.right_fingertip_positions[:, MIDDLE_FINGERTIP_IDX]

    @property
    def right_ring_fingertip_position(self) -> torch.Tensor:
        return self.right_fingertip_positions[:, RING_FINGERTIP_IDX]

    @property
    def right_thumb_fingertip_position(self) -> torch.Tensor:
        return self.right_fingertip_positions[:, THUMB_FINGERTIP_IDX]

    @property
    def left_index_fingertip_position(self) -> torch.Tensor:
        return self.left_fingertip_positions[:, INDEX_FINGERTIP_IDX]

    @property
    def left_middle_fingertip_position(self) -> torch.Tensor:
        return self.left_fingertip_positions[:, MIDDLE_FINGERTIP_IDX]

    @property
    def left_ring_fingertip_position(self) -> torch.Tensor:
        return self.left_fingertip_positions[:, RING_FINGERTIP_IDX]

    @property
    def left_thumb_fingertip_position(self) -> torch.Tensor:
        return self.left_fingertip_positions[:, THUMB_FINGERTIP_IDX]

    #### TENSOR SLICE PROPERTIES END ####
