# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import yaml
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    CYLINDER_MARKER_CFG,
    FRAME_MARKER_CFG,
    SPHERE_MARKER_CFG,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, LightCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab_assets.robots.bimanual import BIMANUAL_CFG, BLUE_BIMANUAL_CFG
from scipy.spatial.transform import Rotation as R
from termcolor import colored

import wandb
from isaaclab_tasks.direct.tyler.bimanual.utils.adjusted_terrain_importer import (
    AdjustedTerrainImporter,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.average_meter import AverageMeter
from isaaclab_tasks.direct.tyler.bimanual.utils.color_constants import (
    BLUE_RGB,
    GREEN_RGB,
    RED_RGB,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.constants import (
    NUM_QUAT,
    NUM_XYZ,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.fabric_robot_constants import (
    LEFT_INDEX_FINGERTIP_LINK_IDX,
    LEFT_MIDDLE_FINGERTIP_LINK_IDX,
    LEFT_PALM_LINK_IDX,
    LEFT_PALM_X_LINK_IDX,
    LEFT_PALM_Y_LINK_IDX,
    LEFT_PALM_Z_LINK_IDX,
    LEFT_RING_FINGERTIP_LINK_IDX,
    LEFT_TASKMAP_LINK_NAMES,
    LEFT_THUMB_FINGERTIP_LINK_IDX,
    NUM_FABRIC_SPHERES,
    RIGHT_INDEX_FINGERTIP_LINK_IDX,
    RIGHT_MIDDLE_FINGERTIP_LINK_IDX,
    RIGHT_PALM_LINK_IDX,
    RIGHT_PALM_X_LINK_IDX,
    RIGHT_PALM_Y_LINK_IDX,
    RIGHT_PALM_Z_LINK_IDX,
    RIGHT_RING_FINGERTIP_LINK_IDX,
    RIGHT_TASKMAP_LINK_NAMES,
    RIGHT_THUMB_FINGERTIP_LINK_IDX,
    URDF_PATH,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.joint_order_constants import (
    fabric_to_isaaclab_joint_order_torch,
    isaaclab_to_fabric_joint_order_torch,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.robot_constants import (
    NUM_ARM_HAND_JOINTS,
    NUM_ARM_JOINTS,
    NUM_BIMANUAL,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.table_constants import (
    TABLE_LENGTH_Z,
    TABLE_QW,
    TABLE_QX,
    TABLE_QY,
    TABLE_QZ,
    TABLE_X,
    TABLE_Y,
    TABLE_Z,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.torch_utils import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    rescale,
    sample_uniform_tensor,
)

FINGER_GOALS = True
FILTER_ARM_ACTIONS = False

USE_FABRIC = True
USE_FABRIC_CUDA_GRAPH = False  # Leave this False almost all the time, CUDA graphs don't offer any speedup (actually slows down) with large batch size

VISUALIZE_FABRIC_SPHERES = False

OBJECT_LENGTH_Z = 0.22

SIM_DT = 1 / 60

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
    episode_length_s = 10.0
    decimation = 1
    arm_action_scale = 0.1
    hand_action_scale = 2.0
    debug_vis = False
    action_space = (
        11 * NUM_BIMANUAL if USE_FABRIC else NUM_ARM_HAND_JOINTS * NUM_BIMANUAL
    )
    observation_space = (
        144
        + (NUM_XYZ * NUM_BIMANUAL if FINGER_GOALS else 0)
        + (NUM_ARM_HAND_JOINTS * NUM_BIMANUAL if FILTER_ARM_ACTIONS else 0)
        + (NUM_ARM_HAND_JOINTS * NUM_BIMANUAL * 2 if USE_FABRIC else 0)
    )
    state_space = 0

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

    blue_robot: ArticulationCfg = BLUE_BIMANUAL_CFG.replace(
        prim_path=f"{ENV_REGEX_NS}/Blue_Robot"
    )

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
            # TODO: This actually doesn't work, so just change the USD: https://github.com/isaac-sim/IsaacLab/issues/622
            # visual_material=sim_utils.PreviewSurfaceCfg(
            #     diffuse_color=GREEN_RGB,
            #     roughness=0.0,
            # ),
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

    origin_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose"
    )
    origin_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)

    right_palm_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/right_palm_pose"
    )
    right_palm_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)
    left_palm_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/left_palm_pose"
    )
    left_palm_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)

    right_palm_target_pose_visualizer: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/right_palm_target_pose")
    )
    right_palm_target_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)
    left_palm_target_pose_visualizer: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/left_palm_target_pose")
    )
    left_palm_target_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)

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

    collision_sphere_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/CollisionSphere"
    )
    collision_sphere_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=RED_RGB)


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


class BimanualEnv(DirectRLEnv):
    cfg: BimanualEnvCfg

    def __init__(self, cfg: BimanualEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Plotting data
        self.plot_data = {
            "actual": [],
            "cmd": [],
            "episode_length_counter": [],
        }

        self._setup_keyboard()
        self._setup_robot_idxs()
        self._setup_sanity_checks()

        # State
        self._reset_state(env_ids=None)
        self._setup_fabric_taskmap()  # Still needed for FK
        if USE_FABRIC:
            self._setup_fabric_action_space()

        # Logging
        self.wandb_dict = {}

        self._update_metrics(env_ids=None)

        # Debug
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_sanity_checks(self):
        assert np.isclose(
            self.cfg.decimation * self.cfg.sim.dt, FABRIC_DT * NUM_FABRIC_DECIMATION
        ), (
            f"self.cfg.decimation * self.cfg.sim.dt: {self.cfg.decimation * self.cfg.sim.dt} != FABRIC_DT * NUM_FABRIC_DECIMATION: {FABRIC_DT * NUM_FABRIC_DECIMATION}"
        )

    def _setup_robot_idxs(self):
        # Robot joint idxs
        self._joint_idxs, self._joint_names = self.robot.find_joints(".*")
        print(colored("!" * 100, "green"))
        print(colored(f"len(self._joint_idxs): {len(self._joint_idxs)}", "green"))
        print(colored(f"self._joint_names: {self._joint_names}", "green"))
        print(colored("!" * 100, "green"))

        # Robot link idxs
        self._link_idxs, self._link_names = self.robot.find_bodies(".*")
        print(colored("!" * 100, "green"))
        print(colored(f"len(self._link_idxs): {len(self._link_idxs)}", "green"))
        print(colored(f"self._link_names: {self._link_names}", "green"))
        print(colored("!" * 100, "green"))

        # Contact sensor link idxs
        self._contact_link_idxs, self._contact_link_names = (
            self.contact_sensor.find_bodies(".*")
        )
        print(colored("!" * 100, "green"))
        print(
            colored(
                f"len(self._contact_link_idxs): {len(self._contact_link_idxs)}", "green"
            )
        )
        print(colored("!" * 100, "green"))

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

    def _setup_fabric_taskmap(self) -> None:
        from fabrics_sim.taskmaps.robot_frame_origins_taskmap import (
            RobotFrameOriginsTaskMap,
        )

        # Create task map that consists of the origins of the following frames stacked together.
        # Create separate left and right taskmaps to avoid gaps in indexing
        self.right_taskmap_link_names = RIGHT_TASKMAP_LINK_NAMES
        self.left_taskmap_link_names = LEFT_TASKMAP_LINK_NAMES
        self.right_taskmap = RobotFrameOriginsTaskMap(
            urdf_path=str(URDF_PATH),
            link_names=self.right_taskmap_link_names,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.left_taskmap = RobotFrameOriginsTaskMap(
            urdf_path=str(URDF_PATH),
            link_names=self.left_taskmap_link_names,
            batch_size=self.num_envs,
            device=self.device,
        )

    def fabric_robot_collision_spheres(self) -> torch.Tensor:
        USE_ISAACLAB_STATE = False
        if USE_ISAACLAB_STATE:
            q = isaaclab_to_fabric_joint_order_torch(self.robot.data.joint_pos)
        else:
            q = self.fabric_q

        N = q.shape[0]
        assert_equals(q.shape, (N, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS))
        sphere_positions, _ = self.fabric.get_taskmap("body_points")(q.detach(), None)
        sphere_positions = sphere_positions.reshape(N, -1, NUM_XYZ)

        # World frame
        sphere_positions_w = sphere_positions + self.scene.env_origins.unsqueeze(dim=1)
        return sphere_positions_w

    def fabric_robot_collision_sphere_radii(self) -> torch.Tensor:
        body_sphere_radii = self.fabric.get_sphere_radii()
        return body_sphere_radii

    def fabric_collision_status(self) -> torch.Tensor:
        return self.fabric.collision_status

    def _setup_scene(self):
        # add articulation to scene
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        if self.include_blue_robot:
            self.blue_robot = Articulation(self.cfg.blue_robot)
            self.scene.articulations["blue_robot"] = self.blue_robot

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
        assert self.raw_actions.shape == (self.num_envs, self.cfg.action_space), (
            f"self.raw_actions.shape: {self.raw_actions.shape} != (self.num_envs, self.cfg.action_space): {(self.num_envs, self.cfg.action_space)}"
        )
        assert torch.all(torch.le(self.raw_actions, 1.0)) and torch.all(
            torch.ge(self.raw_actions, -1.0)
        ), f"self.raw_actions: {self.raw_actions}"

        if USE_FABRIC:
            new_fabric_palm_target, new_fabric_hand_target = (
                self._compute_fabric_actions(self.raw_actions)
            )
            self.fabric_palm_target.copy_(new_fabric_palm_target)
            self.fabric_hand_target.copy_(new_fabric_hand_target)

        if USE_FABRIC:
            # NOTE: Could do this in _apply_action with some smart rounding strategy
            # That depends on sim_dt, fabric_dt, and decimation
            for _ in range(NUM_FABRIC_DECIMATION):
                # Step fabric
                self._step_fabric_state()

            position_targets = fabric_to_isaaclab_joint_order_torch(
                self.fabric_q.detach().clone()
            )
        else:
            position_targets = self._compute_actions(self.raw_actions)

        # Clamp
        joint_pos_limits = self.robot.data.soft_joint_pos_limits.clone()
        position_targets = position_targets.clamp_(
            min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1]
        )

        # Save plotting data
        self.plot_data["actual"].append(
            self.robot.data.joint_pos[0, : NUM_ARM_JOINTS * NUM_BIMANUAL].cpu().numpy()
        )
        self.plot_data["cmd"].append(
            position_targets[0, : NUM_ARM_JOINTS * NUM_BIMANUAL].cpu().numpy()
        )
        self.plot_data["episode_length_counter"].append(
            self.episode_length_buf[0].cpu().numpy()
        )

        DISABLE_ACTIONS = False  # Set to True to debug actions
        if DISABLE_ACTIONS:
            position_targets[:] = 0.0

        self.robot.set_joint_position_target(position_targets)
        if self.include_blue_robot:
            self.blue_robot.write_joint_position_to_sim(position_targets)

    def _apply_action(self):
        pass

    def _compute_fabric_actions(
        self, raw_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Actions are in robot frame
        # [RIGHT xyz, RIGHT euler_ZYX, LEFT xyz, LEFT euler_ZYX]

        # World: X = forward, Y = left, Z = up
        # Palm: x = palm normal, y = palm-to_thumb, z= palm-to-finger
        # 0 = forward, 1 = left, 2 = up
        # 3 = euler_Z, 4 = euler_Y, 5 = euler_X

        # Update fabric targets
        # Action is in [-1, 1] => [min, max]

        # Set to True to debug
        OVERWRITE_WITH_SAMPLED_ACTIONS = False
        if OVERWRITE_WITH_SAMPLED_ACTIONS:
            raw_actions[:] = self.sampled_raw_actions

        # Split into palm and hand actions
        raw_fabric_palm_actions = self.raw_actions[:, : NUM_BIMANUAL * 6]
        raw_fabric_hand_actions = self.raw_actions[:, NUM_BIMANUAL * 6 :]

        ABSOLUTE_PALM_CONTROL = False
        if ABSOLUTE_PALM_CONTROL:
            new_fabric_palm_target = rescale(
                values=raw_fabric_palm_actions,
                old_mins=torch.ones_like(self.fabric_palm_mins) * -1,
                old_maxs=torch.ones_like(self.fabric_palm_maxs) * 1,
                new_mins=self.fabric_palm_mins,
                new_maxs=self.fabric_palm_maxs,
            )
        else:
            current_fabric_palm = torch.cat(
                [self.right_fabric_palm, self.left_fabric_palm], dim=1
            )
            POS_DELTA = 0.2
            ANG_DELTA = np.deg2rad(45)
            fabric_palm_delta_mins = torch.tensor(
                [
                    -POS_DELTA,
                    -POS_DELTA,
                    -POS_DELTA,
                    -ANG_DELTA,
                    -ANG_DELTA,
                    -ANG_DELTA,
                ]
                * NUM_BIMANUAL,
                device=self.device,
            )
            fabric_palm_delta_maxs = torch.tensor(
                [
                    POS_DELTA,
                    POS_DELTA,
                    POS_DELTA,
                    ANG_DELTA,
                    ANG_DELTA,
                    ANG_DELTA,
                ]
                * NUM_BIMANUAL,
                device=self.device,
            )
            new_fabric_palm_target = current_fabric_palm + rescale(
                values=raw_fabric_palm_actions,
                old_mins=torch.ones_like(self.fabric_palm_mins) * -1,
                old_maxs=torch.ones_like(self.fabric_palm_maxs) * 1,
                new_mins=fabric_palm_delta_mins,
                new_maxs=fabric_palm_delta_maxs,
            )
            new_fabric_palm_target = new_fabric_palm_target.clamp_(
                min=self.fabric_palm_mins, max=self.fabric_palm_maxs
            )

        new_fabric_hand_target = rescale(
            values=raw_fabric_hand_actions,
            old_mins=torch.ones_like(self.fabric_hand_mins) * -1,
            old_maxs=torch.ones_like(self.fabric_hand_maxs) * 1,
            new_mins=self.fabric_hand_mins,
            new_maxs=self.fabric_hand_maxs,
        )

        return new_fabric_palm_target, new_fabric_hand_target

    def _compute_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        # Split into arm and hand actions
        raw_arm_actions = raw_actions[:, : NUM_BIMANUAL * NUM_ARM_JOINTS]
        raw_hand_actions = raw_actions[:, NUM_BIMANUAL * NUM_ARM_JOINTS :]

        # Arm
        ABSOLUTE_ARM_CONTROL = False
        if ABSOLUTE_ARM_CONTROL:
            arm_action_offset = self.robot.data.default_joint_pos[:, self._joint_idxs][
                :, : (NUM_ARM_JOINTS * NUM_BIMANUAL)
            ]
        else:
            arm_action_offset = self.robot.data.joint_pos[:, self._joint_idxs][
                :, : (NUM_ARM_JOINTS * NUM_BIMANUAL)
            ]
        assert arm_action_offset.shape == (
            self.num_envs,
            NUM_ARM_JOINTS * NUM_BIMANUAL,
        ), (
            f"arm_action_offset.shape: {arm_action_offset.shape} != (self.num_envs, NUM_ARM_JOINTS * NUM_BIMANUAL): {(self.num_envs, NUM_ARM_JOINTS * NUM_BIMANUAL)}"
        )
        arm_position_targets = (
            self.cfg.arm_action_scale * raw_arm_actions + arm_action_offset
        )

        # Hand
        hand_action_offset = self.robot.data.default_joint_pos[:, self._joint_idxs][
            :, NUM_ARM_JOINTS * NUM_BIMANUAL :
        ]
        hand_position_targets = (
            self.cfg.hand_action_scale * raw_hand_actions + hand_action_offset
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
        return position_targets

    def _step_fabric_state(self):
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

    def _compute_intermediate_values(self):
        pass

    def _get_observations(self) -> dict:
        right_palm_pose_w = self.right_palm_pose_w()
        left_palm_pose_w = self.left_palm_pose_w()
        obs_dict = {
            "q": self.robot.data.joint_pos,
            "qd": self.robot.data.joint_vel,
            "right_fingertip_positions": (
                self.right_fingertip_positions_w()
                - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, -1),
            "left_fingertip_positions": (
                self.left_fingertip_positions_w()
                - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, -1),
            "right_palm_position": right_palm_pose_w[:, :3] - self.scene.env_origins,
            "right_palm_orientation": right_palm_pose_w[:, 3:],
            "left_palm_position": left_palm_pose_w[:, :3] - self.scene.env_origins,
            "left_palm_orientation": left_palm_pose_w[:, 3:],
            "object_position": self.object_position_w - self.scene.env_origins,
            "goal_object_position": self.goal_object_position_w
            - self.scene.env_origins,
            "object_orientation": self.object_orientation,
            "goal_object_orientation": self.goal_object_orientation,
        }
        if FINGER_GOALS:
            obs_dict["right_goal_position"] = (
                self.right_goal_position_w - self.scene.env_origins
            )
            obs_dict["left_goal_position"] = (
                self.left_goal_position_w - self.scene.env_origins
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
                print(colored(f"{k}: {v.shape} (WRONG)", "red"))

        any_nan = False
        for k, v in obs_dict.items():
            if torch.isnan(v).any():
                any_nan = True
                nan_env_ids = torch.where(torch.isnan(v))[0]
                print(colored(f"{k}: {v.shape} (NAN) at {nan_env_ids}", "red"))
        if any_nan:
            import datetime
            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            obs_history_filename = f"{datetime_str}_obs_history.pth"
            torch.save(self.obs_history, obs_history_filename)
            idx_filename = f"{datetime_str}_idx.pth"
            torch.save(self.episode_length_buf, idx_filename)
            print(colored(f"Saved obs_history to {obs_history_filename}", "green"))
            print(colored(f"Saved idx to {idx_filename}", "green"))
            breakpoint()

        obs = torch.cat(
            [obs_dict[key] for key in obs_dict],
            dim=-1,
        )
        batch_idx = torch.arange(self.num_envs, device=obs.device)   # shape (B,)
        time_idx  = self.episode_length_buf                         # shape (B,)
        self.obs_history[batch_idx, time_idx, :] = obs.detach().clone()

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
                "right_index_fingertip_to_goal_dist": -(self.right_index_fingertip_position_w() - self.right_goal_position_w).norm(dim=-1, p=2),
                "left_index_fingertip_to_goal_dist": -(self.left_index_fingertip_position_w() - self.left_goal_position_w).norm(dim=-1, p=2),
            }
        else:
            self.individual_reward_bufs = {
                "right_index_fingertip_to_object_dist": -(self.right_index_fingertip_position_w() - self.object_position_w).norm(dim=-1, p=2),
                "left_index_fingertip_to_object_dist": -(self.left_index_fingertip_position_w() - self.object_position_w).norm(dim=-1, p=2),
                "object_lifted": torch.logical_and(self.object_is_lifted, ~self.object_has_been_lifted_this_episode),
                "object_to_goal_dist": torch.where(
                    self.object_is_lifted,
                    (2.0 - (self.object_position_w - self.goal_object_position_w).norm(dim=-1, p=2)).clip(min=0.0),
                    torch.zeros_like(self.object_position_w[:, 2]),
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
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos *= math_utils.sample_uniform(
            *(0.8, 1.2), joint_pos.shape, joint_pos.device
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        joint_vel *= math_utils.sample_uniform(
            *(0.0, 0.0), joint_vel.shape, joint_vel.device
        )

        joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids].clone()
        joint_pos = joint_pos.clamp_(
            min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1]
        )
        joint_vel_limits = self.robot.data.soft_joint_vel_limits[env_ids].clone()
        joint_vel = joint_vel.clamp_(min=-joint_vel_limits, max=joint_vel_limits)

        self.robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        if self.include_blue_robot:
            self.blue_robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
            # self.blue_robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
            self.blue_robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # Reset object
        object_pose = self._sample_initial_object_pose(env_ids)
        final_object_pose = self._sample_final_object_pose(env_ids)
        self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device), env_ids=env_ids
        )
        self.goal_object.write_root_pose_to_sim(final_object_pose, env_ids=env_ids)
        # self.goal_object.write_root_velocity_to_sim(
        #     torch.zeros(self.num_envs, 6, device=self.device), env_ids=env_ids
        # )

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
                self.robot.data.joint_pos[:, : NUM_ARM_JOINTS * NUM_BIMANUAL]
            )
            self.sampled_raw_actions = sample_uniform_tensor(
                low=torch.tensor([-1.0] * self.cfg.action_space, device=self.device),
                high=torch.tensor([1.0] * self.cfg.action_space, device=self.device),
                N=self.num_envs,
            )

            if FINGER_GOALS:
                self.right_goal_position_w = self._sample_right_goal_position(env_ids)
                self.left_goal_position_w = self._sample_left_goal_position(env_ids)

            self.object_has_been_lifted_this_episode = torch.zeros_like(
                self.object_is_lifted
            )

            if USE_FABRIC:
                self.fabric_q = isaaclab_to_fabric_joint_order_torch(
                    self.robot.data.joint_pos.clone().float()
                )
                self.fabric_qd = torch.zeros_like(self.fabric_q)
                self.fabric_qdd = torch.zeros_like(self.fabric_q)

                self.fabric_palm_target = self.default_fabric_palm_target().clone()
            self.obs_history = torch.zeros(
                self.num_envs, self.max_episode_length, self.cfg.observation_space,
                device=self.device,
            )
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
                env_ids, : NUM_ARM_JOINTS * NUM_BIMANUAL
            ]
            self.sampled_raw_actions[env_ids] = sample_uniform_tensor(
                low=torch.tensor([-1.0] * self.cfg.action_space, device=self.device),
                high=torch.tensor([1.0] * self.cfg.action_space, device=self.device),
                N=len(env_ids),
            )

            if FINGER_GOALS:
                self.right_goal_position_w[env_ids] = self._sample_right_goal_position(
                    env_ids
                )
                self.left_goal_position_w[env_ids] = self._sample_left_goal_position(
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
                self.fabric_palm_target[env_ids] = self.default_fabric_palm_target()[
                    env_ids
                ].clone()
            self.obs_history[env_ids] = torch.zeros(
                len(env_ids), self.max_episode_length, self.cfg.observation_space,
                device=self.device,
            )

    def _sample_right_goal_position(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.2, -0.5, 0.05], device=self.device),
            high=torch.tensor([0.5, -0.1, 0.5], device=self.device),
            N=len(env_ids),
        )

    def _sample_left_goal_position(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.2, 0.1, 0.05], device=self.device),
            high=torch.tensor([0.5, 0.5, 0.5], device=self.device),
            N=len(env_ids),
        )

    def _sample_initial_object_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        position = self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor(
                [-0.2, -0.5, OBJECT_LENGTH_Z / 2 + 0.02], device=self.device
            ),
            high=torch.tensor(
                [0.2, 0.5, OBJECT_LENGTH_Z / 2 + 0.03], device=self.device
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
                [-0.2, -0.5, OBJECT_LENGTH_Z / 2 + 0.02], device=self.device
            ),
            high=torch.tensor(
                [0.2, 0.5, OBJECT_LENGTH_Z / 2 + 0.5], device=self.device
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
            if not hasattr(self, "origin_pose_visualizer"):
                self.origin_pose_visualizer = VisualizationMarkers(
                    self.cfg.origin_pose_visualizer
                )
            if not hasattr(self, "right_palm_pose_visualizer"):
                self.right_palm_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_palm_pose_visualizer
                )
            if not hasattr(self, "left_palm_pose_visualizer"):
                self.left_palm_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_palm_pose_visualizer
                )
            if USE_FABRIC:
                if not hasattr(self, "right_palm_target_pose_visualizer"):
                    self.right_palm_target_pose_visualizer = VisualizationMarkers(
                        self.cfg.right_palm_target_pose_visualizer
                    )
                if not hasattr(self, "left_palm_target_pose_visualizer"):
                    self.left_palm_target_pose_visualizer = VisualizationMarkers(
                        self.cfg.left_palm_target_pose_visualizer
                    )

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
            if VISUALIZE_FABRIC_SPHERES:
                if not hasattr(self, "collision_sphere_visualizers"):
                    self.collision_sphere_visualizers = [
                        VisualizationMarkers(
                            self.cfg.collision_sphere_visualizer.replace(
                                prim_path=f"{self.cfg.collision_sphere_visualizer.prim_path}_{i}"
                            )
                        )
                        for i in range(NUM_FABRIC_SPHERES)
                    ]

            # set their visibility to true
            self.origin_pose_visualizer.set_visibility(True)
            self.right_palm_pose_visualizer.set_visibility(True)
            self.left_palm_pose_visualizer.set_visibility(True)
            if USE_FABRIC:
                self.right_palm_target_pose_visualizer.set_visibility(True)
                self.left_palm_target_pose_visualizer.set_visibility(True)
            if FINGER_GOALS:
                self.right_goal_visualizer.set_visibility(True)
                self.left_goal_visualizer.set_visibility(True)
            self.object_pose_visualizer.set_visibility(True)
            self.goal_object_pose_visualizer.set_visibility(True)
            self.right_fingertip_visualizer.set_visibility(True)
            self.left_fingertip_visualizer.set_visibility(True)
            self.progress_visualizer.set_visibility(True)
            self.progress_full_visualizer.set_visibility(True)
            if VISUALIZE_FABRIC_SPHERES:
                for visualizer in self.collision_sphere_visualizers:
                    visualizer.set_visibility(True)
        else:
            if hasattr(self, "origin_pose_visualizer"):
                self.origin_pose_visualizer.set_visibility(False)
            if hasattr(self, "right_palm_pose_visualizer"):
                self.right_palm_pose_visualizer.set_visibility(False)
            if hasattr(self, "left_palm_pose_visualizer"):
                self.left_palm_pose_visualizer.set_visibility(False)
            if USE_FABRIC:
                if hasattr(self, "right_palm_target_pose_visualizer"):
                    self.right_palm_target_pose_visualizer.set_visibility(False)
                if hasattr(self, "left_palm_target_pose_visualizer"):
                    self.left_palm_target_pose_visualizer.set_visibility(False)
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
            if VISUALIZE_FABRIC_SPHERES:
                if hasattr(self, "collision_sphere_visualizers"):
                    for visualizer in self.collision_sphere_visualizers:
                        visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Make sure the robot is initialized
        if not self.robot.is_initialized:
            return

        POSE_SCALE = [0.1, 0.1, 0.1]
        SPHERE_SCALE = [0.03, 0.03, 0.03]

        base_pos_w = self.robot.data.root_pos_w.clone()
        self.origin_pose_visualizer.visualize(
            translations=base_pos_w,
            orientations=self.robot.data.root_quat_w,
            scales=torch.tensor(POSE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        right_palm_pose = self.right_palm_pose_w()
        left_palm_pose = self.left_palm_pose_w()
        self.right_palm_pose_visualizer.visualize(
            translations=right_palm_pose[:, :3],
            orientations=right_palm_pose[:, 3:],
            scales=torch.tensor(POSE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.left_palm_pose_visualizer.visualize(
            translations=left_palm_pose[:, :3],
            orientations=left_palm_pose[:, 3:],
            scales=torch.tensor(POSE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        if USE_FABRIC:
            right_palm_target_pose = self.right_fabric_palm_target_pose_w
            left_palm_target_pose = self.left_fabric_palm_target_pose_w
            self.right_palm_target_pose_visualizer.visualize(
                translations=right_palm_target_pose[:, :3],
                orientations=right_palm_target_pose[:, 3:],
                scales=torch.tensor(POSE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
            self.left_palm_target_pose_visualizer.visualize(
                translations=left_palm_target_pose[:, :3],
                orientations=left_palm_target_pose[:, 3:],
                scales=torch.tensor(POSE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
        if FINGER_GOALS:
            self.right_goal_visualizer.visualize(
                translations=self.right_goal_position_w,
                scales=torch.tensor(SPHERE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
            self.left_goal_visualizer.visualize(
                translations=self.left_goal_position_w,
                scales=torch.tensor(SPHERE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )

        self.object_pose_visualizer.visualize(
            translations=self.object_position_w,
            orientations=self.object_orientation,
            scales=torch.tensor(POSE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.goal_object_pose_visualizer.visualize(
            translations=self.goal_object_position_w,
            orientations=self.goal_object_orientation,
            scales=torch.tensor(POSE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )

        self.right_fingertip_visualizer.visualize(
            translations=self.right_index_fingertip_position_w(),
            scales=torch.tensor(SPHERE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.left_fingertip_visualizer.visualize(
            translations=self.left_index_fingertip_position_w(),
            scales=torch.tensor(SPHERE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )

        # Growing bar to show progress
        # Full bar to show max progress
        # Make growing bar thicker
        progress_frac = self.episode_length_buf / self.max_episode_length
        progress_full = torch.ones_like(progress_frac)
        progress_pos = self.robot_position_w + torch.tensor(
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
            fabric_collision_spheres = self.fabric_robot_collision_spheres()
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
            print(colored("~" * 100, "red"))
            print(colored(f"Error importing keyboard: {e}", "red"))
            print(
                colored(
                    "Keyboard not available, likely because we are in headless mode.",
                    "red",
                )
            )
            print(colored("~" * 100, "red"))
            return

    def _reset_kbc(self):
        print(colored("In reset_kbc", "green"))
        self._reset_idx(env_ids=None)

    def _breakpoint_kbc(self):
        print(colored("In breakpoint_kbc", "green"))
        breakpoint()

    def _save_kbc(self):
        print(colored("In save_kbc", "green"))
        actual_data = np.stack(self.plot_data["actual"], axis=0)
        cmd_data = np.stack(self.plot_data["cmd"], axis=0)
        episode_length_counter = np.array(self.plot_data["episode_length_counter"])
        N_TIMESTEPS = len(self.plot_data["actual"])
        assert actual_data.shape == (N_TIMESTEPS, NUM_ARM_JOINTS * NUM_BIMANUAL)
        assert cmd_data.shape == (N_TIMESTEPS, NUM_ARM_JOINTS * NUM_BIMANUAL)
        assert episode_length_counter.shape == (N_TIMESTEPS,)
        episode_frac = episode_length_counter / self.max_episode_length
        assert episode_frac.shape == (N_TIMESTEPS,)
        plot_data = np.stack([actual_data, cmd_data], axis=0)
        assert plot_data.shape == (2, N_TIMESTEPS, NUM_ARM_JOINTS * NUM_BIMANUAL)
        import datetime

        output_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz"
        np.savez(
            output_filename,
            plot_data=plot_data,
            joint_names=self.robot.data.joint_names,
            episode_frac=episode_frac,
        )
        print(colored(f"Saved data to {output_filename}", "green"))

    #### KEYBOARD END ####

    #### TENSOR SLICE PROPERTIES START ####
    @property
    def table_position(self) -> torch.Tensor:
        assert self.table.data.body_pos_w.shape == (self.num_envs, 1, NUM_XYZ), (
            f"Table position shape: {self.table.data.body_pos_w.shape}"
        )
        return self.table.data.body_pos_w[:, 0]

    @property
    def object_position_w(self) -> torch.Tensor:
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
    def goal_object_position_w(self) -> torch.Tensor:
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
    def robot_position_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, 0]

    @property
    def object_is_lifted(self) -> torch.Tensor:
        return (
            self.object_position_w[:, 2] > self.table_position[:, 2] + OBJECT_LENGTH_Z
        )

    @property
    def right_fabric_palm_target_pose_w(self) -> torch.Tensor:
        # Actions are in robot frame
        # [RIGHT xyz, RIGHT euler_ZYX, LEFT xyz, LEFT euler_ZYX]
        assert self.fabric_palm_target.shape == (self.num_envs, 6 * NUM_BIMANUAL), (
            f"Fabric palm target shape: {self.fabric_palm_target.shape}"
        )
        right_pos = self.fabric_palm_target[:, :3]

        right_euler_ZYX = self.fabric_palm_target[:, 3:6]
        right_matrix = euler_angles_to_matrix(right_euler_ZYX, "ZYX")
        right_quat_wxyz = matrix_to_quat_wxyz(right_matrix)
        assert right_quat_wxyz.shape == (self.num_envs, 4), (
            f"Quat shape: {right_quat_wxyz.shape}"
        )

        # World frame
        right_pos_w = right_pos + self.scene.env_origins
        right_pose = torch.cat([right_pos_w, right_quat_wxyz], dim=-1)

        return right_pose

    @property
    def left_fabric_palm_target_pose_w(self) -> torch.Tensor:
        # Actions are in robot frame
        # [RIGHT xyz, RIGHT euler_ZYX, LEFT xyz, LEFT euler_ZYX]
        assert self.fabric_palm_target.shape == (self.num_envs, 6 * NUM_BIMANUAL), (
            f"Fabric palm target shape: {self.fabric_palm_target.shape}"
        )
        left_pos = self.fabric_palm_target[:, 6:9]

        left_euler_ZYX_np = self.fabric_palm_target[:, 9:12].detach().cpu().numpy()
        left_quat_xyzw_np = R.from_euler(
            "ZYX", left_euler_ZYX_np, degrees=False
        ).as_quat()
        left_quat_wxyz_np = np.concatenate(
            [left_quat_xyzw_np[..., 3:], left_quat_xyzw_np[..., :3]], axis=-1
        )
        assert left_quat_wxyz_np.shape == (self.num_envs, 4), (
            f"Quat shape: {left_quat_wxyz_np.shape}"
        )
        left_quat_wxyz = torch.from_numpy(left_quat_wxyz_np).to(self.device).float()

        # World frame
        left_pos_w = left_pos + self.scene.env_origins
        left_pose_w = torch.cat([left_pos_w, left_quat_wxyz], dim=-1)
        return left_pose_w

    @property
    def right_fabric_palm(self) -> torch.Tensor:
        pose_w = self.right_palm_pose_w()
        pos = pose_w[:, :3] - self.scene.env_origins
        quat_wxyz = pose_w[:, 3:]
        matrix = quat_wxyz_to_matrix(quat_wxyz)
        euler_ZYX = matrix_to_euler_angles(matrix, "ZYX")
        return torch.cat([pos, euler_ZYX], dim=-1)

    @property
    def left_fabric_palm(self) -> torch.Tensor:
        pose_w = self.left_palm_pose_w()
        pos = pose_w[:, :3] - self.scene.env_origins
        quat_wxyz = pose_w[:, 3:]
        matrix = quat_wxyz_to_matrix(quat_wxyz)
        euler_ZYX = matrix_to_euler_angles(matrix, "ZYX")
        return torch.cat([pos, euler_ZYX], dim=-1)

    def right_taskmap_helper(
        self, q: torch.Tensor, qd: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if qd is None:
            qd = torch.zeros_like(q)

        # BRITTLE: We assume that the input q is in isaacgym order
        # We need to convert it to curobo order
        q = isaaclab_to_fabric_joint_order_torch(q)
        qd = isaaclab_to_fabric_joint_order_torch(qd)

        N = q.shape[0]
        assert_equals(q.shape, (N, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS))
        assert_equals(qd.shape, (N, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS))

        x, jac = self.right_taskmap(q, None)
        n_points = len(self.right_taskmap_link_names)
        assert_equals(x.shape, (N, NUM_XYZ * n_points))
        assert_equals(
            jac.shape, (N, NUM_XYZ * n_points, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS)
        )

        # Calculate the velocity in the task space
        xd = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
        assert_equals(xd.shape, (N, NUM_XYZ * n_points))

        return (
            x.reshape(N, n_points, NUM_XYZ),
            xd.reshape(N, n_points, NUM_XYZ),
            jac.reshape(N, n_points, NUM_XYZ, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS),
        )

    def left_taskmap_helper(
        self, q: torch.Tensor, qd: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if qd is None:
            qd = torch.zeros_like(q)

        # BRITTLE: We assume that the input q is in isaacgym order
        # We need to convert it to curobo order
        q = isaaclab_to_fabric_joint_order_torch(q)
        qd = isaaclab_to_fabric_joint_order_torch(qd)

        N = q.shape[0]
        assert_equals(q.shape, (N, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS))
        assert_equals(qd.shape, (N, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS))

        x, jac = self.left_taskmap(q, None)
        n_points = len(self.left_taskmap_link_names)
        assert_equals(x.shape, (N, NUM_XYZ * n_points))
        assert_equals(
            jac.shape, (N, NUM_XYZ * n_points, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS)
        )

        # Calculate the velocity in the task space
        xd = torch.bmm(jac, qd.unsqueeze(2)).squeeze(2)
        assert_equals(xd.shape, (N, NUM_XYZ * n_points))

        return (
            x.reshape(N, n_points, NUM_XYZ),
            xd.reshape(N, n_points, NUM_XYZ),
            jac.reshape(N, n_points, NUM_XYZ, NUM_BIMANUAL * NUM_ARM_HAND_JOINTS),
        )

    def right_palm_pose_w(self, q: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos

        x, _, _ = self.right_taskmap_helper(
            q=q,
        )
        palm_pos = x[:, RIGHT_PALM_LINK_IDX]
        palm_x_pos = x[:, RIGHT_PALM_X_LINK_IDX]
        palm_y_pos = x[:, RIGHT_PALM_Y_LINK_IDX]
        palm_z_pos = x[:, RIGHT_PALM_Z_LINK_IDX]

        x_dir = torch.nn.functional.normalize(palm_x_pos - palm_pos, p=2, dim=-1)
        y_dir = torch.nn.functional.normalize(palm_y_pos - palm_pos, p=2, dim=-1)
        z_dir = torch.nn.functional.normalize(palm_z_pos - palm_pos, p=2, dim=-1)

        # Assemble the rotation matrix with columns [x̂  ŷ  ẑ]
        # Resulting shape: (B, 3, 3)
        palm_rot_matrix = torch.stack([x_dir, y_dir, z_dir], dim=-1)
        assert palm_rot_matrix.shape == (self.num_envs, 3, 3), (
            f"Palm rot matrix shape: {palm_rot_matrix.shape}"
        )

        palm_quat_wxyz = matrix_to_quat_wxyz(palm_rot_matrix)

        # World frame
        palm_pos_w = palm_pos + self.scene.env_origins
        palm_pose = torch.cat([palm_pos_w, palm_quat_wxyz], dim=-1)
        return palm_pose

    def left_palm_pose_w(self, q: Optional[torch.Tensor] = None) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos

        x, _, _ = self.left_taskmap_helper(
            q=q,
        )
        palm_pos = x[:, LEFT_PALM_LINK_IDX]
        palm_x_pos = x[:, LEFT_PALM_X_LINK_IDX]
        palm_y_pos = x[:, LEFT_PALM_Y_LINK_IDX]
        palm_z_pos = x[:, LEFT_PALM_Z_LINK_IDX]

        x_dir = torch.nn.functional.normalize(palm_x_pos - palm_pos, p=2, dim=-1)
        y_dir = torch.nn.functional.normalize(palm_y_pos - palm_pos, p=2, dim=-1)
        z_dir = torch.nn.functional.normalize(palm_z_pos - palm_pos, p=2, dim=-1)

        # Assemble the rotation matrix with columns [x̂  ŷ  ẑ]
        # Resulting shape: (B, 3, 3)
        palm_rot_matrix = torch.stack([x_dir, y_dir, z_dir], dim=-1)
        assert palm_rot_matrix.shape == (self.num_envs, 3, 3), (
            f"Palm rot matrix shape: {palm_rot_matrix.shape}"
        )

        palm_quat_wxyz = matrix_to_quat_wxyz(palm_rot_matrix)

        # World frame
        palm_pos_w = palm_pos + self.scene.env_origins
        palm_pose = torch.cat([palm_pos_w, palm_quat_wxyz], dim=-1)
        return palm_pose

    def right_fingertip_positions_w(
        self, q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos

        x, _, _ = self.right_taskmap_helper(
            q=q,
        )
        right_index_pos = x[:, RIGHT_INDEX_FINGERTIP_LINK_IDX]
        right_middle_pos = x[:, RIGHT_MIDDLE_FINGERTIP_LINK_IDX]
        right_ring_pos = x[:, RIGHT_RING_FINGERTIP_LINK_IDX]
        right_thumb_pos = x[:, RIGHT_THUMB_FINGERTIP_LINK_IDX]

        positions = torch.stack(
            [right_index_pos, right_middle_pos, right_ring_pos, right_thumb_pos],
            dim=1,
        )
        # World frame
        positions_w = positions + self.scene.env_origins.unsqueeze(dim=1)
        return positions_w

    def left_fingertip_positions_w(
        self, q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos

        x, _, _ = self.left_taskmap_helper(
            q=q,
        )
        left_index_pos = x[:, LEFT_INDEX_FINGERTIP_LINK_IDX]
        left_middle_pos = x[:, LEFT_MIDDLE_FINGERTIP_LINK_IDX]
        left_ring_pos = x[:, LEFT_RING_FINGERTIP_LINK_IDX]
        left_thumb_pos = x[:, LEFT_THUMB_FINGERTIP_LINK_IDX]

        positions = torch.stack(
            [left_index_pos, left_middle_pos, left_ring_pos, left_thumb_pos], dim=1
        )
        # World frame
        positions_w = positions + self.scene.env_origins.unsqueeze(dim=1)
        return positions_w

    def right_index_fingertip_position_w(
        self, q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.right_fingertip_positions_w(q)[:, 0]

    def left_index_fingertip_position_w(
        self, q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.left_fingertip_positions_w(q)[:, 0]

    def default_fabric_palm_target(self) -> torch.Tensor:
        # Compute palm poses at default joint positions
        default_palm_target = np.array(
            [0.7298, -0.2469, 0.5738, 2.25930292, 0.86978541, 1.86671697]
            + [0.7298, 0.2469, 0.5738, -2.25930299, 0.86978536, -1.86671716],
        )
        return (
            torch.from_numpy(default_palm_target)
            .float()
            .to(self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0)
        )

    #### TENSOR SLICE PROPERTIES END ####

    #### OTHER PROPERTIES START ####
    @property
    def include_blue_robot(self) -> bool:
        return self.cfg.debug_vis and self.num_envs < 10

    #### OTHER PROPERTIES END ####
