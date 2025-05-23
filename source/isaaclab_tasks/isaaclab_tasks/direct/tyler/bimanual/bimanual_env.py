# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import List
from live_plotter import FastLivePlotter

import yaml
from pathlib import Path
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
from isaaclab_tasks.direct.tyler.bimanual.utils.torch_utils import (
    sample_uniform_tensor,
    rescale,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.joint_order_constants import (
    isaaclab_to_fabric_joint_order_torch,
    fabric_to_isaaclab_joint_order_torch,
)
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

from isaaclab.terrains.terrain_importer import TerrainImporter

class AdjustedTerrainImporter(TerrainImporter):
    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Add a plane to the terrain importer.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            size: The size of the plane. Defaults to (2.0e6, 2.0e6).

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # obtain ground plane color from the configured visual material
        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            # defaults to the `GroundPlaneCfg` color if diffuse color attribute is not found
            if "diffuse_color" in material:
                color = material["diffuse_color"]
            else:
                pass
                # omni.log.warn(
                #     "Visual material specified for ground plane but no diffuse color found."
                #     " Using default color: (0.0, 0.0, 0.0)"
                # )

        # get the mesh
        ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size, color=color)
        ground_plane_cfg.func(prim_path, ground_plane_cfg, translation=(0.0, 0.0, -1.0))


FINGER_GOALS = False
FILTER_ARM_ACTIONS = False

USE_FABRIC = True
USE_FABRIC_CUDA_GRAPH = False

VISUALIZE_FABRIC_SPHERES = False
if VISUALIZE_FABRIC_SPHERES:
    NUM_FABRIC_SPHERES = 80
else:
    NUM_FABRIC_SPHERES = 0

OBJECT_LENGTH_Z = 0.22

NUM_BIMANUAL = 2
SIM_DT = 1 / 120

FABRIC_DT = 1 / 60
NUM_FABRIC_DECIMATION = 1

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
    episode_length_s = 1.0  # TODO: Should be 6
    decimation = 2
    arm_action_scale = 0.1
    hand_action_scale = 2.0
    action_space = 11 * 2 if USE_FABRIC else 23 * 2
    observation_space = (
        136
        + (6 if FINGER_GOALS else 0)
        + (7 * 2 if FILTER_ARM_ACTIONS else 0)
        + (23 * 2 * 2 if USE_FABRIC else 0)
    )
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
        class_type=AdjustedTerrainImporter,
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
                float(TABLE_Z) + TABLE_LENGTH_Z / 2 + OBJECT_LENGTH_Z / 2 + 0.02,
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
                diffuse_color=GREEN_RGB,  # TODO: This actually doesn't work, so just change the USD: https://github.com/isaac-sim/IsaacLab/issues/622
                roughness=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                float(TABLE_X),
                float(TABLE_Y),
                float(TABLE_Z) + TABLE_LENGTH_Z / 2 + OBJECT_LENGTH_Z / 2 + 0.1,
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

    right_goal_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/right_goal"
    )
    right_goal_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN_RGB)
    left_goal_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/left_goal"
    )
    left_goal_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN_RGB)

    object_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/object_pose"
    )
    object_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)
    goal_object_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_object_pose"
    )
    goal_object_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)

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

    collision_sphere_visualizers: List[VisualizationMarkersCfg] = [
        SPHERE_MARKER_CFG.replace(prim_path=f"/Visuals/CollisionSphere_{i}")
        for i in range(NUM_FABRIC_SPHERES)
    ]


if FINGER_GOALS:
    REWARD_NAMES = [
        "right_index_fingertip_to_goal_dist",
        "left_index_fingertip_to_goal_dist",
    ]
else:
    REWARD_NAMES = [
        "right_index_fingertip_to_object_dist",
        "left_index_fingertip_to_object_dist",
        "object_lifted",
        "object_to_goal_dist",
    ]


def assert_equals(a, b):
    assert a == b, f"a: {a} != b: {b}"


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
        self.live_plotter_data = {
            "actual": [],
            "cmd": [],
            "episode_length_counter": [],
        }

        # State
        self._reset_state(env_ids=None)
        if USE_FABRIC:
            self._setup_fabric_action_space()

        # Logging
        self.wandb_dict = {}

        self._update_metrics(env_ids=None)

        # Debug
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_robot_idxs(self):
        # Robot joint idxs
        self._joint_idxs, self._joint_names = self.robot.find_joints(".*")
        print("!" * 100)
        print(f"len(self._joint_idxs): {len(self._joint_idxs)}")
        print(f"self._joint_names: {self._joint_names}")
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

    def _setup_fabric_action_space(self) -> None:
        # Hide imports so that the code still runs without fabrics if unused
        from fabrics_sim.fabrics.bimanual_kuka_allegro_pose_fabric_v2 import (
            BimanualKukaAllegroPoseFabricV2,
        )
        from fabrics_sim.integrator.integrators import DisplacementIntegrator
        from fabrics_sim.utils.path_utils import get_params_path
        from fabrics_sim.utils.utils import capture_fabric
        from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
        from isaaclab_tasks.direct.tyler.bimanual.utils.fabric_world import (
            world_dict_robot_frame,
        )

        USE_FABRIC_WORLD = True
        if USE_FABRIC_WORLD:
            self.fabric_world_dict = world_dict_robot_frame.copy()
        else:
            raise ValueError(
                "If not fabric world given, the self-collisions do not work for some reason"
            )
            self.fabric_world_dict = {}

        # Load fabric params and potentially modify
        path = Path(get_params_path())
        fabric_params_filename = "bimanual_kuka_allegro_pose_params.yaml"
        config_path = path / fabric_params_filename
        with open(config_path, "r") as file:
            fabric_params = yaml.safe_load(file)
        fabric_params = fabric_params["fabric_params"]

        # Declare device for fabric
        self.fabric_world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=20,
            device=self.device,
            world_dict=self.fabric_world_dict,
        )
        self.fabric_object_ids, self.fabric_object_indicator = (
            self.fabric_world_model.get_object_ids()
        )
        self.fabric = BimanualKukaAllegroPoseFabricV2(
            batch_size=self.num_envs,
            device=self.device,
            timestep=FABRIC_DT,
            graph_capturable=USE_FABRIC_CUDA_GRAPH,
            fabric_params=fabric_params,
        )
        self.fabric_hand_mins = torch.tensor(
            NUM_BIMANUAL * [0.2475, -0.3286, -0.7238, -0.0192, -0.5532],
            device=self.device,
        )
        self.fabric_hand_maxs = torch.tensor(
            NUM_BIMANUAL * [3.8336, 3.0025, 0.8977, 1.0243, 0.0629],
            device=self.device,
        )

        self.fabric_palm_mins = torch.tensor(
            NUM_BIMANUAL * [0.0, -1.0, 0, -3.1416, -3.1416, -3.1416], device=self.device
        )
        self.fabric_palm_maxs = torch.tensor(
            NUM_BIMANUAL * [1.2, 1.0, 1.0, 3.1416, 3.1416, 3.1416], device=self.device
        )
        assert (self.fabric_hand_maxs > self.fabric_hand_mins).all(), (
            f"{self.fabric_hand_maxs} <= {self.fabric_hand_mins}"
        )
        assert (self.fabric_palm_maxs > self.fabric_palm_mins).all(), (
            f"{self.fabric_palm_maxs} <= {self.fabric_palm_mins}"
        )

        # Targets (stored as variables to enable CUDA graph computation)
        # Palm target is (origin, Euler ZYX)
        self.fabric_hand_target = rescale(
            torch.rand(
                self.num_envs, self.fabric_hand_maxs.numel(), device=self.device
            ),
            old_mins=torch.zeros_like(self.fabric_hand_mins),
            old_maxs=torch.ones_like(self.fabric_hand_mins),
            new_mins=self.fabric_hand_mins,
            new_maxs=self.fabric_hand_maxs,
        )

        default_palm_target = np.array(
            NUM_BIMANUAL * [-0.6868, 0.0320, 0.6685, -2.3873, -0.0824, 3.1301]
        )
        self.fabric_palm_target = (
            torch.from_numpy(default_palm_target)
            .float()
            .to(self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0)
        )

        self.fabric_integrator = DisplacementIntegrator(self.fabric)

        if USE_FABRIC_CUDA_GRAPH:
            fabric_inputs = [
                self.fabric_hand_target,
                self.fabric_palm_target,
                "euler_zyx",
                self.fabric_q.detach(),
                self.fabric_qd.detach(),
                self.fabric_object_ids,
                self.fabric_object_indicator,
            ]
            (
                self.fabric_cuda_graph,
                self.fabric_q_new,
                self.fabric_qd_new,
                self.fabric_qdd_new,
            ) = capture_fabric(
                fabric=self.fabric,
                q=self.fabric_q,
                qd=self.fabric_qd,
                qdd=self.fabric_qdd,
                timestep=FABRIC_DT,
                fabric_integrator=self.fabric_integrator,
                inputs=fabric_inputs,
                device=self.device,
            )

    def fabric_robot_collision_spheres(self) -> torch.Tensor:
        USE_ISAACLAB_STATE = False
        if USE_ISAACLAB_STATE:
            q = isaaclab_to_fabric_joint_order_torch(self.robot.data.joint_pos)
        else:
            q = self.fabric_q

        N = q.shape[0]
        assert_equals(q.shape, (N, NUM_BIMANUAL * 23))
        sphere_positions, _ = self.fabric.get_taskmap("body_points")(q.detach(), None)
        sphere_positions = sphere_positions.reshape(N, -1, NUM_XYZ)
        return sphere_positions

    def fabric_robot_collision_sphere_radii(self) -> torch.Tensor:
        body_sphere_radii = self.fabric.get_sphere_radii()
        return body_sphere_radii

    def fabric_collision_status(self) -> torch.Tensor:
        return self.fabric.collision_status

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
        import time
        start_time = time.time()
        self.prev_raw_actions = self.raw_actions.clone()
        self.raw_actions = actions.clone()
        assert self.raw_actions.shape == self.prev_raw_actions.shape, (
            f"self.raw_actions.shape: {self.raw_actions.shape} != self.prev_raw_actions.shape: {self.prev_raw_actions.shape}"
        )
        assert self.raw_actions.shape == (self.num_envs, self.cfg.action_space), (
            f"self.raw_actions.shape: {self.raw_actions.shape} != (self.num_envs, self.cfg.action_space): {(self.num_envs, self.cfg.action_space)}"
        )
        assert torch.all(torch.le(self.raw_actions, 1.0)) and torch.all(
            torch.ge(self.raw_actions, -1.0)
        ), f"self.raw_actions: {self.raw_actions}"

        if USE_FABRIC:
            # Actions are in robot frame
            # [RIGHT xyz, RIGHT euler_ZYX, LEFT xyz, LEFT euler_ZYX]

            # World: X = forward, Y = left, Z = up
            # Palm: x = palm normal, y = palm-to_thumb, z= palm-to-finger
            # 0 = forward, 1 = left, 2 = up
            # 3 = euler_Z, 4 = euler_Y, 5 = euler_X

            # Update fabric targets
            # Action is in [-1, 1] => [min, max]
            self.fabric_palm_target.copy_(
                rescale(
                    values=self.raw_actions[:, : NUM_BIMANUAL * 6],
                    old_mins=torch.ones_like(self.fabric_palm_mins) * -1,
                    old_maxs=torch.ones_like(self.fabric_palm_maxs) * 1,
                    new_mins=self.fabric_palm_mins,
                    new_maxs=self.fabric_palm_maxs,
                )
            )
            self.fabric_hand_target.copy_(
                rescale(
                    values=self.raw_actions[:, NUM_BIMANUAL * 6 :],
                    old_mins=torch.ones_like(self.fabric_hand_mins) * -1,
                    old_maxs=torch.ones_like(self.fabric_hand_maxs) * 1,
                    new_mins=self.fabric_hand_mins,
                    new_maxs=self.fabric_hand_maxs,
                )
            )
            self.fabric_steps_counter = 0
            # TODO: Remove
            # print("!" * 100)
            # print(f"fabric_palm_target: {self.fabric_palm_target}")
            # print(f"fabric_hand_target: {self.fabric_hand_target}")
            # print("!" * 100)

        if USE_FABRIC:
            if self.fabric_steps_counter < NUM_FABRIC_DECIMATION:
                self.fabric_steps_counter += 1
                # TODO: Remove
                # print("*" * 100)
                # print(f"fabric_steps_counter: {self.fabric_steps_counter}")
                # print("BEFORE")
                # print(f"fabric_q: {self.fabric_q}")
                # print(f"fabric_qd: {self.fabric_qd}")

                # Step fabric
                if USE_FABRIC_CUDA_GRAPH:
                    self.fabric_cuda_graph.replay()
                    self.fabric_q.copy_(self.fabric_q_new)
                    self.fabric_qd.copy_(self.fabric_qd_new)
                    self.fabric_qdd.copy_(self.fabric_qdd_new)

                else:
                    # Set the targets
                    self.fabric.set_features(
                        self.fabric_hand_target,
                        self.fabric_palm_target,
                        "euler_zyx",
                        self.fabric_q.detach(),
                        self.fabric_qd.detach(),
                        self.fabric_object_ids,
                        self.fabric_object_indicator,
                    )

                    # Integrate fabrics one step producing new position and velocity.
                    self.fabric_q, self.fabric_qd, self.fabric_qdd = (
                        self.fabric_integrator.step(
                            self.fabric_q.detach(),
                            self.fabric_qd.detach(),
                            self.fabric_qdd.detach(),
                            FABRIC_DT,
                        )
                    )
                # TODO: Remove
                # print("AFTER")
                # print(f"fabric_q: {self.fabric_q}")
                # print(f"fabric_qd: {self.fabric_qd}")
                # print("*" * 100)

            # TODO: HACK
            position_targets = self.robot.data.default_joint_pos.clone() + sample_uniform_tensor(
                low=torch.ones_like(self.robot.data.joint_pos[0]) * -0.02,
                high=torch.ones_like(self.robot.data.joint_pos[0]) * 0.02,
                N=self.num_envs,
            )
            # position_targets = fabric_to_isaaclab_joint_order_torch(
            #     self.fabric_q.detach().clone()
            # )

            # TODO: HACK
            position_targets = self.robot.data.default_joint_pos.clone() + sample_uniform_tensor(
                low=torch.ones_like(self.robot.data.joint_pos[0]) * -0.02,
                high=torch.ones_like(self.robot.data.joint_pos[0]) * 0.02,
                N=self.num_envs,
            )

            # TODO: Remove
            # print("~" * 100)
            # print(f"position_targets: {position_targets}")
            # print("~" * 100)
        else:
            # Arm
            ABSOLUTE_ARM_CONTROL = False
            if ABSOLUTE_ARM_CONTROL:
                arm_action_offset = self.robot.data.default_joint_pos[
                    :, self._joint_idxs
                ][:, :14]
            else:
                arm_action_offset = self.robot.data.joint_pos[:, self._joint_idxs][
                    :, :14
                ]
            assert arm_action_offset.shape == (self.num_envs, 14), (
                f"arm_action_offset.shape: {arm_action_offset.shape} != (self.num_envs, 14): {(self.num_envs, 14)}"
            )
            arm_position_targets = (
                self.cfg.arm_action_scale * self.raw_actions[:, :14] + arm_action_offset
            )

            # Hand
            hand_action_offset = self.robot.data.default_joint_pos[:, self._joint_idxs][
                :, 14:
            ]
            hand_position_targets = (
                self.cfg.hand_action_scale * self.raw_actions[:, 14:]
                + hand_action_offset
            )

            if FILTER_ARM_ACTIONS:
                ALPHA = 0.9
                self.filtered_arm_position_targets = (
                    ALPHA * self.filtered_arm_position_targets
                    + (1 - ALPHA) * arm_position_targets
                )
                arm_position_targets = self.filtered_arm_position_targets

            position_targets = torch.cat(
                [arm_position_targets, hand_position_targets], dim=-1
            )

            # TODO: HACK
            position_targets = self.robot.data.default_joint_pos.clone() + sample_uniform_tensor(
                low=torch.ones_like(self.robot.data.joint_pos[0]) * -0.02,
                high=torch.ones_like(self.robot.data.joint_pos[0]) * 0.02,
                N=self.num_envs,
            )

        # TODO: Remove
        # if (
        #     self.robot.data.joint_pos[0, :14] - position_targets[0, :14]
        # ).abs().max() > 0.1:
        #     print("*" * 100)
        #     print(f"position_targets[0, :14]: {position_targets[0, :14]}")
        #     print(
        #         f"self.robot.data.joint_pos[0, :14]: {self.robot.data.joint_pos[0, :14]}"
        #     )
        #     print(
        #         f"diff: {self.robot.data.joint_pos[0, :14] - position_targets[0, :14]}"
        #     )
        #     print(
        #         f"diff > 0.1: {(self.robot.data.joint_pos[0, :14] - position_targets[0, :14]).abs() > 0.1}"
        #     )
        #     print("*" * 100)

        self.live_plotter_data["actual"].append(
            self.robot.data.joint_pos[0, :14].cpu().numpy()
        )
        self.live_plotter_data["cmd"].append(position_targets[0, :14].cpu().numpy())
        self.live_plotter_data["episode_length_counter"].append(self.episode_length_buf[0].cpu().numpy())
        DISABLE_ACTIONS = False  # Set to True to debug actions
        if DISABLE_ACTIONS:
            position_targets[:] = 0.0

        self.robot.set_joint_position_target(
            position_targets, joint_ids=self._joint_idxs
        )

        end_time = time.time()
        print()
        print("%" * 100)
        print(f"pre_physics_step time: {end_time - start_time}")
        print("%" * 100)
        print()


    def _apply_action(self):
        pass

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
        if FINGER_GOALS:
            obs_dict["right_goal_position"] = (
                self.right_goal_position - self.scene.env_origins
            )
            obs_dict["left_goal_position"] = (
                self.left_goal_position - self.scene.env_origins
            )
        if FILTER_ARM_ACTIONS:
            obs_dict["filtered_arm_position_targets"] = (
                self.filtered_arm_position_targets
            )

        if USE_FABRIC:
            obs_dict["fabric_q"] = self.fabric_q
            obs_dict["fabric_qd"] = self.fabric_qd

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
        if FINGER_GOALS:
            self.individual_reward_bufs = {
                "right_index_fingertip_to_goal_dist": -(self.right_index_fingertip_position - self.right_goal_position).norm(dim=-1, p=2),
                "left_index_fingertip_to_goal_dist": -(self.left_index_fingertip_position - self.left_goal_position).norm(dim=-1, p=2),
            }
        else:
            self.individual_reward_bufs = {
                "right_index_fingertip_to_object_dist": -(self.right_index_fingertip_position - self.object_position).norm(dim=-1, p=2),
                "left_index_fingertip_to_object_dist": -(self.left_index_fingertip_position - self.object_position).norm(dim=-1, p=2),
                "object_lifted": torch.logical_and(self.object_is_lifted, ~self.object_has_been_lifted_this_episode),
                "object_to_goal_dist": torch.where(
                    self.object_is_lifted,
                    (2.0 - (self.object_position - self.goal_object_position).norm(dim=-1, p=2)).clip(min=0.0),
                    torch.zeros_like(self.object_position[:, 2]),
                ),
            }
        # fmt: on
        assert set(self.individual_reward_bufs.keys()) == set(REWARD_NAMES), (
            f"Individual reward buffers and reward names do not match: {self.individual_reward_bufs.keys()} vs {REWARD_NAMES}\nOnly in individual reward buffers: {set(self.individual_reward_bufs.keys()) - set(REWARD_NAMES)}\nOnly in reward names: {set(REWARD_NAMES) - set(self.individual_reward_bufs.keys())}"
        )

        if not hasattr(self, "reward_weights"):
            if FINGER_GOALS:
                self.individual_reward_weights = {
                    "right_index_fingertip_to_goal_dist": 1.0,
                    "left_index_fingertip_to_goal_dist": 1.0,
                }
            else:
                self.individual_reward_weights = {
                    "right_index_fingertip_to_object_dist": 1.0,
                    "left_index_fingertip_to_object_dist": 1.0,
                    "object_lifted": 50.0,
                    "object_to_goal_dist": 10.0,
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
        self.object_has_been_lifted_this_episode = torch.where(
            self.object_has_been_lifted_this_episode,
            self.object_has_been_lifted_this_episode,
            self.object_is_lifted,
        )

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

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset robot
        root_state = self.robot.data.default_root_state[env_ids].clone()
        default_position = root_state[:, :3] + self.scene.env_origins[env_ids]
        default_orientation = root_state[:, 3:7]
        default_velocity = root_state[:, 7:13]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos *= math_utils.sample_uniform(
            *(0.8, 1.2), joint_pos.shape, joint_pos.device
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_vel *= math_utils.sample_uniform(
            *(0.0, 0.0), joint_vel.shape, joint_vel.device
        )

        joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids].clone()
        joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        joint_vel_limits = self.robot.data.soft_joint_vel_limits[env_ids].clone()
        joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

        # self.robot.write_root_pose_to_sim(
        #     torch.cat([default_position, default_orientation], dim=-1), env_ids=env_ids
        # )
        # self.robot.write_root_velocity_to_sim(default_velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim(joint_pos, None, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(joint_vel, None, env_ids=env_ids)
        self.robot.set_joint_position_target(
            joint_pos, joint_ids=self._joint_idxs, env_ids=env_ids
        )

        self.object.write_root_pose_to_sim(
            self._sample_initial_object_pose(env_ids), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros_like(default_velocity), env_ids=env_ids
        )
        self.goal_object.write_root_pose_to_sim(
            self._sample_final_object_pose(env_ids), env_ids=env_ids
        )
        self.goal_object.write_root_velocity_to_sim(
            torch.zeros_like(default_velocity), env_ids=env_ids
        )

        self._update_metrics(env_ids)
        self._reset_state(env_ids)

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

            self.filtered_arm_position_targets = torch.zeros_like(
                self.robot.data.joint_pos[:, :14]
            )

            if FINGER_GOALS:
                self.right_goal_position = self._sample_right_goal_position(env_ids)
                self.left_goal_position = self._sample_left_goal_position(env_ids)

            self.object_has_been_lifted_this_episode = torch.zeros_like(
                self.object_is_lifted
            )

            if USE_FABRIC:
                self.fabric_q = isaaclab_to_fabric_joint_order_torch(
                    self.robot.data.joint_pos.clone().float()
                )
                self.fabric_qd = torch.zeros_like(self.fabric_q)
                self.fabric_qdd = torch.zeros_like(self.fabric_q)
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

            self.filtered_arm_position_targets[env_ids] = self.robot.data.joint_pos[
                env_ids, :14
            ]

            if FINGER_GOALS:
                self.right_goal_position[env_ids] = self._sample_right_goal_position(
                    env_ids
                )
                self.left_goal_position[env_ids] = self._sample_left_goal_position(
                    env_ids
                )

            self.object_has_been_lifted_this_episode[env_ids] = torch.zeros_like(
                self.object_is_lifted[env_ids]
            )

            if USE_FABRIC:
                self.fabric_q[env_ids] = isaaclab_to_fabric_joint_order_torch(
                    self.robot.data.joint_pos[env_ids].clone().float()
                )
                self.fabric_qd[env_ids] = torch.zeros_like(self.fabric_q[env_ids])
                self.fabric_qdd[env_ids] = torch.zeros_like(self.fabric_q[env_ids])

    def _sample_right_goal_position(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.5, -0.5, 0.05], device=self.device),
            high=torch.tensor([0.5, -0.1, 0.5], device=self.device),
            N=len(env_ids),
        )

    def _sample_left_goal_position(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.5, 0.1, 0.05], device=self.device),
            high=torch.tensor([0.5, 0.5, 0.5], device=self.device),
            N=len(env_ids),
        )

    def _sample_initial_object_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        position = self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor(
                [-0.4, -0.5, OBJECT_LENGTH_Z / 2 + 0.02], device=self.device
            ),
            high=torch.tensor(
                [0.4, 0.5, OBJECT_LENGTH_Z / 2 + 0.03], device=self.device
            ),
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
            low=torch.tensor(
                [-0.4, -0.5, OBJECT_LENGTH_Z / 2 + 0.02], device=self.device
            ),
            high=torch.tensor(
                [0.4, 0.5, OBJECT_LENGTH_Z / 2 + 0.5], device=self.device
            ),
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
            if FINGER_GOALS:
                if not hasattr(self, "right_goal_visualizer"):
                    self.right_goal_visualizer = VisualizationMarkers(
                        self.cfg.right_goal_visualizer
                    )
                if not hasattr(self, "left_goal_visualizer"):
                    self.left_goal_visualizer = VisualizationMarkers(
                        self.cfg.left_goal_visualizer
                    )

            if not hasattr(self, "object_pose_visualizer"):
                self.object_pose_visualizer = VisualizationMarkers(
                    self.cfg.object_pose_visualizer
                )
            if not hasattr(self, "goal_object_pose_visualizer"):
                self.goal_object_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_object_pose_visualizer
                )
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
            if not hasattr(self, "collision_sphere_visualizers"):
                self.collision_sphere_visualizers = [
                    VisualizationMarkers(cfg)
                    for cfg in self.cfg.collision_sphere_visualizers
                ]

            # set their visibility to true
            self.pose_visualizer.set_visibility(True)
            if FINGER_GOALS:
                self.right_goal_visualizer.set_visibility(True)
                self.left_goal_visualizer.set_visibility(True)
            self.object_pose_visualizer.set_visibility(True)
            self.goal_object_pose_visualizer.set_visibility(True)
            self.right_fingertip_visualizer.set_visibility(True)
            self.left_fingertip_visualizer.set_visibility(True)
            self.progress_visualizer.set_visibility(True)
            self.progress_full_visualizer.set_visibility(True)
            for visualizer in self.collision_sphere_visualizers:
                visualizer.set_visibility(True)
        else:
            if hasattr(self, "pose_visualizer"):
                self.pose_visualizer.set_visibility(False)
            if FINGER_GOALS:
                if hasattr(self, "right_goal_visualizer"):
                    self.right_goal_visualizer.set_visibility(False)
                if hasattr(self, "left_goal_visualizer"):
                    self.left_goal_visualizer.set_visibility(False)
            if hasattr(self, "object_pose_visualizer"):
                self.object_pose_visualizer.set_visibility(False)
            if hasattr(self, "goal_object_pose_visualizer"):
                self.goal_object_pose_visualizer.set_visibility(False)
            if hasattr(self, "right_fingertip_visualizer"):
                self.right_fingertip_visualizer.set_visibility(False)
            if hasattr(self, "left_fingertip_visualizer"):
                self.left_fingertip_visualizer.set_visibility(False)
            if hasattr(self, "progress_visualizer"):
                self.progress_visualizer.set_visibility(False)
            if hasattr(self, "progress_full_visualizer"):
                self.progress_full_visualizer.set_visibility(False)
            if hasattr(self, "collision_sphere_visualizers"):
                for visualizer in self.collision_sphere_visualizers:
                    visualizer.set_visibility(False)

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
        if FINGER_GOALS:
            self.right_goal_visualizer.visualize(
                translations=self.right_goal_position,
                scales=torch.tensor([0.03, 0.03, 0.03], device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
            self.left_goal_visualizer.visualize(
                translations=self.left_goal_position,
                scales=torch.tensor([0.03, 0.03, 0.03], device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )

        self.object_pose_visualizer.visualize(
            translations=self.object_position,
            orientations=self.object_orientation,
            scales=torch.tensor([0.2, 0.2, 0.2], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.goal_object_pose_visualizer.visualize(
            translations=self.goal_object_position,
            orientations=self.goal_object_orientation,
            scales=torch.tensor([0.2, 0.2, 0.2], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )

        self.right_fingertip_visualizer.visualize(
            translations=self.right_index_fingertip_position,
            scales=torch.tensor([0.03, 0.03, 0.03], device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.left_fingertip_visualizer.visualize(
            translations=self.left_index_fingertip_position,
            scales=torch.tensor([0.03, 0.03, 0.03], device=self.device)
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

        if VISUALIZE_FABRIC_SPHERES:
            fabric_collision_spheres = (
                self.fabric_robot_collision_spheres()
                + self.scene.env_origins.unsqueeze(dim=1)
            )
            fabric_collision_sphere_radii = self.fabric_robot_collision_sphere_radii()
            n_spheres = fabric_collision_spheres.shape[1]
            assert_equals(
                fabric_collision_spheres.shape, (self.num_envs, n_spheres, NUM_XYZ)
            )
            assert_equals(len(fabric_collision_sphere_radii), n_spheres)
            assert NUM_FABRIC_SPHERES == n_spheres, (
                f"NUM_FABRIC_SPHERES: {NUM_FABRIC_SPHERES}, n_spheres: {n_spheres}"
            )
            for i in range(n_spheres):
                self.collision_sphere_visualizers[i].visualize(
                    translations=fabric_collision_spheres[:, i, :],
                    scales=torch.tensor(
                        [
                            fabric_collision_sphere_radii[i],
                            fabric_collision_sphere_radii[i],
                            fabric_collision_sphere_radii[i],
                        ],
                        device=self.device,
                    )
                    .unsqueeze(dim=0)
                    .repeat_interleave(self.num_envs, dim=0),
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
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.S,
                        func=self._save_kbc,
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

    def _save_kbc(self):
        print("In save_kbc")
        actual_data = np.stack(self.live_plotter_data["actual"], axis=0)
        cmd_data = np.stack(self.live_plotter_data["cmd"], axis=0)
        episode_length_counter = np.array(self.live_plotter_data["episode_length_counter"])
        N_TIMESTEPS = len(self.live_plotter_data["actual"])
        assert actual_data.shape == (N_TIMESTEPS, 14)
        assert cmd_data.shape == (N_TIMESTEPS, 14)
        assert episode_length_counter.shape == (N_TIMESTEPS,)
        episode_frac = episode_length_counter / self.max_episode_length
        assert episode_frac.shape == (N_TIMESTEPS,)
        plot_data = np.stack([actual_data, cmd_data], axis=0)
        assert plot_data.shape == (2, N_TIMESTEPS, 14)
        import datetime
        output_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz"
        np.savez(output_filename, plot_data=plot_data, joint_names=self.robot.data.joint_names, episode_frac=episode_frac)
        print(f"Saved data to {output_filename}")

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

    @property
    def object_is_lifted(self) -> torch.Tensor:
        return self.object_position[:, 2] > self.table_position[:, 2] + OBJECT_LENGTH_Z

    #### TENSOR SLICE PROPERTIES END ####
