# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import yaml
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    CUBOID_MARKER_CFG,
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
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
from isaaclab_assets.robots.bimanual import BIMANUAL_CFG, BLUE_BIMANUAL_CFG
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
    ENV_REGEX_NS,
    NUM_QUAT,
    NUM_RPY,
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
    NUM_FABRIC_WORLD_CUBES,
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
    USE_FABRIC_CUDA_GRAPH,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.joint_order_constants import (
    ISAACLAB_JOINT_ORDER,
    VISER_JOINT_ORDER,
    change_joint_order_torch,
    fabric_to_isaaclab_joint_order_torch,
    isaaclab_to_fabric_joint_order_torch,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.object_constants import (
    NUM_OBJECT_KEYPOINTS,
    OBJECT_KEYPOINT_OFFSETS,
    OBJECT_LENGTH_Z,
    OBJECT_NUM_RIGID_BODIES,
    compute_keypoint_positions,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.robot_constants import (
    NUM_ARM_HAND_JOINTS,
    NUM_ARM_JOINTS,
    NUM_BIMANUAL,
    NUM_FINGERS,
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
    pose_to_T,
    quat_wxyz_to_matrix,
    rescale,
    sample_uniform_tensor,
    transform_points,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

FINGER_GOALS = False  # Use finger goals as the task (fingers go to goal positions)
FILTER_ARM_ACTIONS = True  # Filter arm actions over time to be smoother

USE_FABRIC = True  # Use fabric action space or direct action space

SAVE_OBS_HISTORY = False  # Store observation history over time for debugging

NUM_FUTURE_GOAL_OBS = 4  # Number of future goal observations
NUM_FUTURE_PALM_GOAL_OBS = 4  # Number of future palm goal observations

SIM_DT = 1 / 60  # Simulation time step
CONTACT_SENSOR_HISTORY_LENGTH = 1  # Number of contact sensor history steps to use

FORCE_MAG = 1.0  # Magnitude of force to apply to object

RANDOMIZE_OBJECT_SCALE = False  # NOTE: This doesn't work with collision filtering

INCLUDE_CONTACT_REWARD = False
INCLUDE_HAND_TRACKING_REWARD = False
INCLUDE_Q_OBS = True
INCLUDE_QD_OBS = True
INCLUDE_FABRIC_OBS = True
CONTACT_OBS_TYPE = "contacts"  # "forces", "contacts"
assert CONTACT_OBS_TYPE in ["forces", "contacts"], (
    f"Invalid contact obs type: {CONTACT_OBS_TYPE}"
)

OBJECT_NAME = "pitcher"  # "box", "pitcher", "basket"
OBJECT_TRAJECTORY_IDX = 0  # 0, 1, 2

if "box" in OBJECT_NAME:
    OBJECT_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/manually_created/box/usd/box.usd"
    GREEN_OBJECT_USD_PATH = (
        f"{ISAACLAB_ASSETS_DATA_DIR}/manually_created/green_box/usd/box.usd"
    )
    # OBJECT_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/white_box/usd/white_box.usd"
    # GREEN_OBJECT_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/green_white_box/usd/white_box.usd"
elif "pitcher" in OBJECT_NAME:
    OBJECT_USD_PATH = (
        f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/pitcher/usd_convex_decomp/pitcher.usd"
    )
    GREEN_OBJECT_USD_PATH = (
        f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/green_pitcher/usd/pitcher.usd"
    )
elif "basket" in OBJECT_NAME:
    OBJECT_USD_PATH = (
        f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/basket/usd_convex_decomp/basket.usd"
    )
    GREEN_OBJECT_USD_PATH = (
        f"{ISAACLAB_ASSETS_DATA_DIR}/kiri/green_basket/usd/basket.usd"
    )
else:
    raise ValueError(f"Invalid object name: {OBJECT_NAME}")

physics_material = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
)


def compute_num_actions():
    return 11 * NUM_BIMANUAL if USE_FABRIC else NUM_ARM_HAND_JOINTS * NUM_BIMANUAL


def compute_num_observations():
    return (
        (NUM_ARM_HAND_JOINTS * NUM_BIMANUAL if INCLUDE_Q_OBS else 0)  # q
        + (NUM_ARM_HAND_JOINTS * NUM_BIMANUAL if INCLUDE_QD_OBS else 0)  # qd
        + (NUM_XYZ * NUM_FINGERS * NUM_BIMANUAL)  # fingertip positions
        + ((NUM_XYZ + NUM_QUAT) * NUM_BIMANUAL)  # palm poses
        + (NUM_XYZ * NUM_OBJECT_KEYPOINTS)  # object keypoint positions
        + (NUM_XYZ * NUM_OBJECT_KEYPOINTS)  # goal object keypoint positions
        + (NUM_XYZ * NUM_BIMANUAL if FINGER_GOALS else 0)  # fingertip goal positions
        + (
            (NUM_ARM_JOINTS * NUM_BIMANUAL)
            if FILTER_ARM_ACTIONS and not USE_FABRIC
            else ((NUM_XYZ + NUM_RPY) * NUM_BIMANUAL)
            if FILTER_ARM_ACTIONS and USE_FABRIC
            else 0
        )  # filtered arm actions
        + (
            NUM_ARM_HAND_JOINTS * NUM_BIMANUAL * 2
            if USE_FABRIC and INCLUDE_FABRIC_OBS
            else 0
        )  # fabric state
        + (
            NUM_XYZ * NUM_BIMANUAL if INCLUDE_HAND_TRACKING_REWARD else 0
        )  # palm goal positions
        + (
            NUM_XYZ * NUM_BIMANUAL * NUM_FUTURE_PALM_GOAL_OBS
            if INCLUDE_HAND_TRACKING_REWARD
            else 0
        )  # future palm goal positions
        + (
            NUM_XYZ * NUM_OBJECT_KEYPOINTS * NUM_FUTURE_GOAL_OBS
        )  # future goal object keypoint positions
        + compute_num_actions()  # prev actions
        + (NUM_XYZ * NUM_BIMANUAL)  # palm linvels
        + (NUM_XYZ * NUM_FINGERS * NUM_BIMANUAL)  # fingertip linvels
        + (NUM_XYZ * 2)  # object linvel and angvel
        + 1  # smallest_this_episode_right_index_fingertip_to_object_dist
        + 1  # smallest_this_episode_left_index_fingertip_to_object_dist
        + 1  # smallest_this_episode_object_to_goal_dist
        + 1  # episode_length_buf
        + 1  # object_is_lifted
        + 1  # object_has_been_lifted_this_episode
        + (
            (17 * NUM_BIMANUAL) * (1 if CONTACT_OBS_TYPE == "contacts" else 3)
        )  # object contacts
    )


def compute_num_states():
    return compute_num_observations()


NUM_ACTIONS = compute_num_actions()
NUM_OBSERVATIONS = compute_num_observations()
NUM_STATES = compute_num_states()


def do_nothing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    pass


@configclass
class BimanualEventCfg:
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.5, 0.7),
            "dynamic_friction_range": (0.5, 0.7),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale if RANDOMIZE_OBJECT_SCALE else do_nothing,
        mode="prestartup"
        if RANDOMIZE_OBJECT_SCALE
        else "reset",  # Must be done "prestartup" to work normally, but if disable, can't be prestartup
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "scale_range": (0.8, 1.2),  # Scale all axes equally
            # "scale_range": {"x": (0.5, 1.5), "y": (0.5, 1.5), "z": (0.5, 1.5)},  # Scale axes independently
        },
    )


TABLE_CONTACT_SENSOR_RIGHT_ROBOT_LINKS = ["right_iiwa14_link_7"] + [
    f"right_{link_name}_link_{link_idx}"
    for link_name in ["index", "middle", "ring", "thumb"]
    for link_idx in range(4)
]
TABLE_CONTACT_SENSOR_LEFT_ROBOT_LINKS = [
    x.replace("right", "left") for x in TABLE_CONTACT_SENSOR_RIGHT_ROBOT_LINKS
]
TABLE_CONTACT_SENSOR_ROBOT_LINKS = (
    TABLE_CONTACT_SENSOR_RIGHT_ROBOT_LINKS + TABLE_CONTACT_SENSOR_LEFT_ROBOT_LINKS
)
OBJECT_CONTACT_SENSOR_ROBOT_LINKS = TABLE_CONTACT_SENSOR_ROBOT_LINKS


FINGERTIP_CONTACT_SENSOR_RIGHT_ROBOT_LINKS = [
    f"right_{link_name}_link_3" for link_name in ["index", "middle", "ring", "thumb"]
]
FINGERTIP_CONTACT_SENSOR_LEFT_ROBOT_LINKS = [
    x.replace("right", "left") for x in FINGERTIP_CONTACT_SENSOR_RIGHT_ROBOT_LINKS
]
FINGERTIP_CONTACT_SENSOR_ROBOT_LINKS = (
    FINGERTIP_CONTACT_SENSOR_RIGHT_ROBOT_LINKS
    + FINGERTIP_CONTACT_SENSOR_LEFT_ROBOT_LINKS
)


@configclass
class BimanualEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    arm_action_scale = 0.1
    hand_action_scale = 2.0
    debug_vis = False
    action_space = NUM_ACTIONS
    observation_space = NUM_OBSERVATIONS
    state_space = NUM_STATES

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=decimation,
        physics_material=physics_material,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
        ),
        render=sim_utils.RenderCfg(
            rendering_mode="quality",
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
        num_envs=4096,
        env_spacing=4.0,
        # replicate_physics=True,
        # replicate_physics=False,  # Should normally be True, but if randomize USDs, then must be False. But must be True for collision filtering
        replicate_physics=not RANDOMIZE_OBJECT_SCALE,
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
            activate_contact_sensors=True,
            usd_path=OBJECT_USD_PATH,
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
            usd_path=GREEN_OBJECT_USD_PATH,
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
            # Setting no collisions doesn't work, need to change usd
            # collision_props=sim_utils.CollisionPropertiesCfg(
            #     collision_enabled=False,
            # ),
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
            activate_contact_sensors=True,
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
    # contact_sensor = ContactSensorCfg(
    #     prim_path=f"{ENV_REGEX_NS}/Robot/.*",
    #     history_length=CONTACT_SENSOR_HISTORY_LENGTH,
    #     update_period=SIM_DT,
    #     debug_vis=False,
    #     force_threshold=0.01,
    # )
    table_contact_sensor = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/Table/table",
        history_length=CONTACT_SENSOR_HISTORY_LENGTH,
        update_period=SIM_DT,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=[
            f"{ENV_REGEX_NS}/Robot/{link}" for link in TABLE_CONTACT_SENSOR_ROBOT_LINKS
        ],
    )
    object_contact_sensor = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/Object/baseLink",
        history_length=CONTACT_SENSOR_HISTORY_LENGTH,
        update_period=SIM_DT,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=[
            f"{ENV_REGEX_NS}/Robot/{link}" for link in OBJECT_CONTACT_SENSOR_ROBOT_LINKS
        ],
    )

    fingertip_contact_sensor = ContactSensorCfg(
        prim_path=f"{ENV_REGEX_NS}/Robot/REPLACE",
        history_length=CONTACT_SENSOR_HISTORY_LENGTH,
        update_period=SIM_DT,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=[f"{ENV_REGEX_NS}/Object/baseLink"],
    )

    # light
    light: LightCfg = DomeLightCfg(
        intensity=500.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    )

    # events
    events: BimanualEventCfg = BimanualEventCfg()

    # Action noise
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # Observation noise
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = (
        NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
        )
    )

    # Palm pose visualizers
    right_palm_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/right_palm_pose"
    )
    right_palm_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)
    left_palm_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/left_palm_pose"
    )
    left_palm_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)

    # Palm target pose visualizers
    right_fabric_palm_target_pose_visualizer: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/right_fabric_palm_target_pose"
        )
    )
    right_fabric_palm_target_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)
    left_fabric_palm_target_pose_visualizer: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/left_fabric_palm_target_pose"
        )
    )
    left_fabric_palm_target_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)

    # Finger goal visualizers
    right_finger_goal_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/right_finger_goal"
    )
    right_finger_goal_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN_RGB)
    left_finger_goal_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/left_finger_goal"
    )
    left_finger_goal_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN_RGB)

    # Object keypoint visualizers
    object_keypoint_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/object_keypoint"
    )
    object_keypoint_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=BLUE_RGB)
    goal_object_keypoint_visualizer: VisualizationMarkersCfg = (
        SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_object_keypoint")
    )
    goal_object_keypoint_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=GREEN_RGB)

    # Fingertip visualizers
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

    # Progress visualizers
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

    # Collision sphere visualizer
    collision_sphere_visualizer: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/CollisionSphere"
    )
    collision_sphere_visualizer.markers[
        "sphere"
    ].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=RED_RGB)

    # Fabric world visualizer
    fabric_world_visualizer: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="/Visuals/FabricWorld"
    )
    fabric_world_visualizer.markers[
        "cuboid"
    ].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=BLUE_RGB, opacity=0.2
    )  # NOTE: Opacity not working
    fabric_world_visualizer.markers["cuboid"].size = (1.0, 1.0, 1.0)

    # Goal palm pose visualizers
    goal_right_palm_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_right_palm_pose"
    )
    goal_right_palm_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)
    goal_left_palm_pose_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_left_palm_pose"
    )
    goal_left_palm_pose_visualizer.markers["frame"].scale = (1.0, 1.0, 1.0)


if FINGER_GOALS:
    REWARD_NAMES = [
        "right_index_fingertip_to_goal_dist",
        "left_index_fingertip_to_goal_dist",
    ]
else:
    REWARD_NAMES = [
        "right_index_fingertip_to_object_dist",
        "left_index_fingertip_to_object_dist",
        # "object_lifted",
        # "object_to_goal_dist",
        # "object_reached_goal",
        "object_tracking_reward",
    ]
    if INCLUDE_CONTACT_REWARD:
        REWARD_NAMES.append("fingertip_contact")
    if INCLUDE_HAND_TRACKING_REWARD:
        REWARD_NAMES.append("right_hand_tracking_reward")
        REWARD_NAMES.append("left_hand_tracking_reward")


def assert_equals(a, b):
    assert a == b, f"a: {a} != b: {b}"


def check_nan_and_print_if_any(x: torch.Tensor, name: str):
    if x.isnan().any():
        print(colored("!" * 100, "red"))
        print(colored(f"{name} contains NaNs", "red"))
        env_idx = torch.where(torch.isnan(x))[0]
        print(colored(f"env_idx: {env_idx}", "red"))
        print(colored("!" * 100, "red"))
        breakpoint()


class BimanualEnv(DirectRLEnv):
    """
    Conventions:
    * _w means in world frame (each env is different, need to subtract env origin to get in env frame)
    * pose means [xyz, quat_wxyz]
    * pose_w means [xyz_w, quat_wxyz]
    * xyzZYX means [xyz, euler_ZYX]

    Goal:
    * Refers to object or hand poses that are from the human demo (usually read from a file), often used to compute rewards

    Target:
    * Refers to actions taken by the agent (e.g., joint position targets, palm pose targets)
    """

    cfg: BimanualEnvCfg

    def __init__(self, cfg: BimanualEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Plotting data
        self.plot_data = {
            "actual": [],
            "cmd": [],
            "episode_length_counter": [],
        }

        self._setup_viewer_camera()
        self._setup_keyboard()
        self._setup_robot_idxs()
        self._setup_sanity_checks()
        self._setup_demo_trajectory()
        self._setup_default_joint_pos()

        # Taskmap is needed for FK, even if not using fabric
        # Must be done before _reset_state() because it uses the taskmap
        self._setup_fabric_taskmap()

        # State
        self._reset_state(env_ids=None)

        if USE_FABRIC:
            # Must be done after _reset_state() because it uses fabric_q from that
            self._setup_fabric_action_space()

        # Logging
        self.wandb_dict = {}

        self._update_metrics(env_ids=None)

        # Debug
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_sanity_checks(self):
        pass

    def _setup_demo_trajectory(self):
        ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.parent
        DEMO_TRAJECTORY_PATH = (
            ROOT_DIR / f"2025-05-29_outputs/{OBJECT_NAME}_{OBJECT_TRAJECTORY_IDX}.pkl"
        )
        assert DEMO_TRAJECTORY_PATH.exists(), f"{DEMO_TRAJECTORY_PATH} does not exist"
        with open(DEMO_TRAJECTORY_PATH, "rb") as f:
            data = pickle.load(f)
        right_T_R_Ps = data["right_T_R_Ps"]
        left_T_R_Ps = data["left_T_R_Ps"]
        T_R_Os = data["T_R_Os"]
        NUM_GOAL_TIMESTEPS = right_T_R_Ps.shape[0]
        assert_equals(right_T_R_Ps.shape, (NUM_GOAL_TIMESTEPS, 4, 4))
        assert_equals(left_T_R_Ps.shape, (NUM_GOAL_TIMESTEPS, 4, 4))
        assert_equals(T_R_Os.shape, (NUM_GOAL_TIMESTEPS, 4, 4))
        self.goal_right_T_R_Ps = torch.from_numpy(right_T_R_Ps).to(self.device).float()
        self.goal_left_T_R_Ps = torch.from_numpy(left_T_R_Ps).to(self.device).float()
        self.goal_T_R_Os = torch.from_numpy(T_R_Os).to(self.device).float()

    def _setup_default_joint_pos(self):
        USE_ORIGINAL_DEFAULT_JOINT_POS = (
            False  # Set to True to debug using default joint pos
        )
        if USE_ORIGINAL_DEFAULT_JOINT_POS:
            self.robot_custom_default_joint_pos = (
                self.robot.data.default_joint_pos.clone()
            )
            return

        ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.parent
        DEMO_ARM_PATH = (
            ROOT_DIR
            / f"2025-05-29_outputs/{OBJECT_NAME}_{OBJECT_TRAJECTORY_IDX}_arm.pkl"
        )
        assert DEMO_ARM_PATH.exists(), f"{DEMO_ARM_PATH} does not exist"
        with open(DEMO_ARM_PATH, "rb") as f:
            data = pickle.load(f)
        right_arm_q = data["right_arm_q"]
        left_arm_q = data["left_arm_q"]
        assert right_arm_q.shape == left_arm_q.shape == (NUM_ARM_JOINTS,), (
            f"Expected right_arm_q and left_arm_q to have shape (NUM_ARM_JOINTS,), got {right_arm_q.shape} and {left_arm_q.shape}"
        )
        orig_default_q_isaaclab = self.robot.data.default_joint_pos.clone()
        assert orig_default_q_isaaclab.shape == (
            self.num_envs,
            NUM_BIMANUAL * NUM_ARM_HAND_JOINTS,
        ), (
            f"Expected default_joint_pos to have shape (NUM_BIMANUAL * NUM_ARM_HAND_JOINTS,), got {orig_default_q_isaaclab.shape}"
        )
        orig_default_q_viser = change_joint_order_torch(
            orig_default_q_isaaclab,
            from_order=ISAACLAB_JOINT_ORDER,
            to_order=VISER_JOINT_ORDER,
        )
        new_default_q_viser = orig_default_q_viser.clone()
        new_default_q_viser[:, :NUM_ARM_JOINTS] = (
            torch.from_numpy(right_arm_q).to(self.device).float().unsqueeze(dim=0)
        )
        new_default_q_viser[
            :, NUM_ARM_HAND_JOINTS : NUM_ARM_HAND_JOINTS + NUM_ARM_JOINTS
        ] = torch.from_numpy(left_arm_q).to(self.device).float().unsqueeze(dim=0)
        new_default_q_isaaclab = change_joint_order_torch(
            new_default_q_viser,
            from_order=VISER_JOINT_ORDER,
            to_order=ISAACLAB_JOINT_ORDER,
        )
        assert_equals(new_default_q_isaaclab.shape, orig_default_q_isaaclab.shape)
        self.robot_custom_default_joint_pos = new_default_q_isaaclab

    def _setup_viewer_camera(self):
        if self.viewport_camera_controller is not None:
            self.viewport_camera_controller.update_view_location(
                eye=(2.5, 0.0, 1.2),
                lookat=(0.0, 0.0, 0.2),
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
        # self._contact_link_idxs, self._contact_link_names = (
        #     self.contact_sensor.find_bodies(".*")
        # )
        # print(colored("!" * 100, "green"))
        # print(
        #     colored(
        #         f"len(self._contact_link_idxs): {len(self._contact_link_idxs)}", "green"
        #     )
        # )
        # print(colored("!" * 100, "green"))

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

        # HACK: Hardcode new params
        fabric_params["cspace_damping"]["gain"] = 10.0
        fabric_params["cspace_damping"]["hand_gain"] = 5.0

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
            timestep=self.fabric_dt,
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
                timestep=self.fabric_dt,
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
        # self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # self.scene.sensors["contact_sensor"] = self.contact_sensor

        # add table contact sensor to scene
        self.table_contact_sensor = ContactSensor(self.cfg.table_contact_sensor)
        self.scene.sensors["table_contact_sensor"] = self.table_contact_sensor

        self.object_contact_sensor = ContactSensor(self.cfg.object_contact_sensor)
        self.scene.sensors["object_contact_sensor"] = self.object_contact_sensor

        self.fingertip_contact_sensors: Dict[str, ContactSensor] = {}
        for link in FINGERTIP_CONTACT_SENSOR_ROBOT_LINKS:
            contact_sensor = ContactSensor(
                self.cfg.fingertip_contact_sensor.replace(
                    prim_path=self.cfg.fingertip_contact_sensor.prim_path.replace(
                        "REPLACE", link
                    )
                )
            )
            self.scene.sensors[f"fingertip_contact_sensor_{link}"] = contact_sensor
            self.fingertip_contact_sensors[link] = contact_sensor

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        self.cfg.light.func("/World/Light", self.cfg.light)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.clamp_(min=-1.0, max=1.0)

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

        check_nan_and_print_if_any(
            self.robot.data.joint_pos,
            "self.robot.data.joint_pos (start of pre_physics_step)",
        )
        check_nan_and_print_if_any(
            self.robot.data.joint_vel,
            "self.robot.data.joint_vel (start of pre_physics_step)",
        )
        check_nan_and_print_if_any(
            self.raw_actions, "self.raw_actions (start of pre_physics_step)"
        )

        if USE_FABRIC:
            check_nan_and_print_if_any(
                self.fabric_palm_target,
                "self.fabric_palm_target (start of pre_physics_step)",
            )
            check_nan_and_print_if_any(
                self.fabric_hand_target,
                "self.fabric_hand_target (start of pre_physics_step)",
            )

            new_fabric_palm_target, new_fabric_hand_target = (
                self._compute_fabric_actions(self.raw_actions)
            )
            check_nan_and_print_if_any(
                new_fabric_palm_target, "new_fabric_palm_target (after computing)"
            )
            check_nan_and_print_if_any(
                new_fabric_hand_target, "new_fabric_hand_target (after computing)"
            )
            if FILTER_ARM_ACTIONS:
                ALPHA = 0.1  # 1 means no filtering, 0 means never update
                new_fabric_palm_target = (
                    ALPHA * new_fabric_palm_target
                    + (1 - ALPHA) * self.fabric_palm_target
                )

            self.fabric_palm_target.copy_(new_fabric_palm_target)
            self.fabric_hand_target.copy_(new_fabric_hand_target)

            check_nan_and_print_if_any(
                self.fabric_palm_target, "self.fabric_palm_target (after copying)"
            )
            check_nan_and_print_if_any(
                self.fabric_hand_target, "self.fabric_hand_target (after copying)"
            )

        OVERWRITE_GO_TO_GOAL = False
        if OVERWRITE_GO_TO_GOAL:
            goal_right_palm_xyzZYX = self.pose_w_to_xyzZYX(
                self.goal_right_palm_pose_w()
            )
            goal_left_palm_xyzZYX = self.pose_w_to_xyzZYX(self.goal_left_palm_pose_w())

            # Actions
            self.fabric_palm_target[:, :3] = goal_right_palm_xyzZYX[:, :3]
            self.fabric_palm_target[:, 3:6] = goal_right_palm_xyzZYX[:, 3:6]
            self.fabric_palm_target[:, 6:9] = goal_left_palm_xyzZYX[:, :3]
            self.fabric_palm_target[:, 9:12] = goal_left_palm_xyzZYX[:, 3:6]

            OVERWRITE_LEFT_PALM_POSE = False
            if OVERWRITE_LEFT_PALM_POSE:
                if not hasattr(self, "first_goal_left_palm_xyzZYX"):
                    self.first_goal_left_palm_xyzZYX = goal_left_palm_xyzZYX.clone()

                # Adjust position manually
                new_goal_left_palm_xyz = goal_right_palm_xyzZYX[:, :3].clone()
                # adjusted_left_palm_xyz[:, 1] *= -1
                new_goal_left_palm_xyz[:, 1] += 0.3

                # Keep same orientation as the start
                new_goal_left_palm_ZYX = goal_left_palm_xyzZYX[:, 3:6].clone()

                self.fabric_palm_target[:, 6:9] = new_goal_left_palm_xyz
                self.fabric_palm_target[:, 9:12] = new_goal_left_palm_ZYX

        if USE_FABRIC:
            check_nan_and_print_if_any(
                self.fabric_q, "self.fabric_q (start of pre_physics_step)"
            )
            check_nan_and_print_if_any(
                self.fabric_qd, "self.fabric_qd (start of pre_physics_step)"
            )
            check_nan_and_print_if_any(
                self.fabric_qdd, "self.fabric_qdd (start of pre_physics_step)"
            )

            # NOTE: Could do this in _apply_action with some smart rounding strategy
            # That depends on sim_dt, fabric_dt, and decimation
            for i in range(self.fabric_decimation):
                # Step fabric
                check_nan_and_print_if_any(
                    self.fabric_q, f"self.fabric_q (before step {i})"
                )
                check_nan_and_print_if_any(
                    self.fabric_qd, f"self.fabric_qd (before step {i})"
                )
                check_nan_and_print_if_any(
                    self.fabric_qdd, f"self.fabric_qdd (before step {i})"
                )
                self._step_fabric_state()
                check_nan_and_print_if_any(
                    self.fabric_q, f"self.fabric_q (after step {i})"
                )
                check_nan_and_print_if_any(
                    self.fabric_qd, f"self.fabric_qd (after step {i})"
                )
                check_nan_and_print_if_any(
                    self.fabric_qdd, f"self.fabric_qdd (after step {i})"
                )

            position_targets = fabric_to_isaaclab_joint_order_torch(
                self.fabric_q.detach().clone()
            )
            check_nan_and_print_if_any(
                position_targets,
                "position_targets (after fabric_to_isaaclab_joint_order_torch)",
            )
        else:
            position_targets = self._compute_actions(self.raw_actions)
            check_nan_and_print_if_any(
                position_targets, "position_targets (after _compute_actions)"
            )

        # Clamp
        joint_pos_limits = self.robot.data.soft_joint_pos_limits.clone()
        position_targets = position_targets.clamp_(
            min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1]
        )
        check_nan_and_print_if_any(
            position_targets, "position_targets (after clamping)"
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
            self.blue_robot.set_joint_position_target(position_targets)

        self._apply_external_wrench()

        goal_T_R_Os = self.goal_T_R_Os[
            self.goal_float_idx.long().clip(max=self.goal_T_R_Os.shape[0] - 1)
        ]
        goal_object_pos = goal_T_R_Os[:, :3, 3] + self.scene.env_origins
        goal_object_quat_wxyz = matrix_to_quat_wxyz(goal_T_R_Os[:, :3, :3])
        self.goal_object.write_root_pose_to_sim(
            torch.cat([goal_object_pos, goal_object_quat_wxyz], dim=-1)
        )

    def _apply_action(self):
        pass

    def _apply_external_wrench(self):
        # NOTE: external forces are stateful and need to be reset after applying them
        external_force_o = torch.zeros(
            self.num_envs, OBJECT_NUM_RIGID_BODIES, NUM_XYZ, device=self.device
        )
        external_torque_o = torch.zeros(
            self.num_envs, OBJECT_NUM_RIGID_BODIES, NUM_XYZ, device=self.device
        )

        # Keyboard force
        if (self.keyboard_external_force_w.abs() > 0.0).any():
            # self.keyboard_external_force_w is in world frame
            # When applying to the object, we need to transform it to the object frame
            T_W_O = pose_to_T(
                torch.cat(
                    [
                        torch.zeros_like(self.object_position_w),
                        self.object_orientation,
                    ],
                    dim=-1,
                )
            )
            T_O_W = T_W_O.inverse()
            keyboard_external_force_o = transform_points(
                T=T_O_W,
                points=self.keyboard_external_force_w,
            )
            # Reset keyboard external force after applying it
            self.keyboard_external_force_w[:] = 0.0

            external_force_o += keyboard_external_force_o.unsqueeze(
                dim=1
            ).repeat_interleave(OBJECT_NUM_RIGID_BODIES, dim=1)

        # Random force
        APPLY_RANDOM_FORCE = False
        if APPLY_RANDOM_FORCE:
            random_force_o = (
                torch.randn(self.num_envs, NUM_XYZ, device=self.device) * FORCE_MAG
            )
            external_force_o += random_force_o.repeat_interleave(
                OBJECT_NUM_RIGID_BODIES, dim=1
            )

        self.object.set_external_force_and_torque(
            forces=external_force_o,
            torques=external_torque_o,
        )

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
                [
                    self.pose_w_to_xyzZYX(self.right_palm_pose_w()),
                    self.pose_w_to_xyzZYX(self.left_palm_pose_w()),
                ],
                dim=1,
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
            # arm_action_offset = self.robot.data.default_joint_pos[:, self._joint_idxs][
            arm_action_offset = self.robot_custom_default_joint_pos[
                :, self._joint_idxs
            ][:, : (NUM_ARM_JOINTS * NUM_BIMANUAL)]
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
        # hand_action_offset = self.robot.data.default_joint_pos[:, self._joint_idxs][
        hand_action_offset = self.robot_custom_default_joint_pos[:, self._joint_idxs][
            :, NUM_ARM_JOINTS * NUM_BIMANUAL :
        ]
        hand_position_targets = (
            self.cfg.hand_action_scale * raw_hand_actions + hand_action_offset
        )

        if FILTER_ARM_ACTIONS:
            ALPHA = 0.1  # 1 means no filtering, 0 means never update
            self.filtered_arm_position_targets = (
                ALPHA * arm_position_targets
                + (1 - ALPHA) * self.filtered_arm_position_targets
            )
            arm_position_targets = self.filtered_arm_position_targets

        position_targets = torch.cat(
            [arm_position_targets, hand_position_targets], dim=-1
        )
        return position_targets

    def _step_fabric_state(self):
        if USE_FABRIC_CUDA_GRAPH:
            check_nan_and_print_if_any(self.fabric_q, "self.fabric_q (before step)")
            check_nan_and_print_if_any(self.fabric_qd, "self.fabric_qd (before step)")
            check_nan_and_print_if_any(self.fabric_qdd, "self.fabric_qdd (before step)")
            self.fabric_cuda_graph.replay()
            self.fabric_q.copy_(self.fabric_q_new)
            self.fabric_qd.copy_(self.fabric_qd_new)
            self.fabric_qdd.copy_(self.fabric_qdd_new)
            check_nan_and_print_if_any(self.fabric_q, "self.fabric_q (after step)")
            check_nan_and_print_if_any(self.fabric_qd, "self.fabric_qd (after step)")
            check_nan_and_print_if_any(self.fabric_qdd, "self.fabric_qdd (after step)")
        else:
            check_nan_and_print_if_any(
                self.fabric_hand_target, "self.fabric_hand_target (before set_features)"
            )
            check_nan_and_print_if_any(
                self.fabric_palm_target, "self.fabric_palm_target (before set_features)"
            )
            check_nan_and_print_if_any(
                self.fabric_q, "self.fabric_q (before set_features)"
            )
            check_nan_and_print_if_any(
                self.fabric_qd, "self.fabric_qd (before set_features)"
            )
            check_nan_and_print_if_any(
                self.fabric_qdd, "self.fabric_qdd (before set_features)"
            )
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
            check_nan_and_print_if_any(
                self.fabric_q, "self.fabric_q (after set_features)"
            )
            check_nan_and_print_if_any(
                self.fabric_qd, "self.fabric_qd (after set_features)"
            )
            check_nan_and_print_if_any(
                self.fabric_qdd, "self.fabric_qdd (after set_features)"
            )
            prev_fabric_q = self.fabric_q.detach().clone()
            prev_fabric_qd = self.fabric_qd.detach().clone()
            prev_fabric_qdd = self.fabric_qdd.detach().clone()

            # Integrate fabrics one step producing new position and velocity.
            self.fabric_q, self.fabric_qd, self.fabric_qdd = (
                self.fabric_integrator.step(
                    prev_fabric_q,
                    prev_fabric_qd,
                    prev_fabric_qdd,
                    self.fabric_dt,
                )
            )
            check_nan_and_print_if_any(self.fabric_q, "self.fabric_q (after step)")
            check_nan_and_print_if_any(self.fabric_qd, "self.fabric_qd (after step)")
            check_nan_and_print_if_any(self.fabric_qdd, "self.fabric_qdd (after step)")

    def _compute_intermediate_values(self):
        object_goal_keypoint_dist = self.object_goal_keypoint_distance
        small_object_goal_distance_ids = (
            (object_goal_keypoint_dist < 0.25).nonzero(as_tuple=False).squeeze(-1)
        )
        self.goal_float_idx[small_object_goal_distance_ids] += 1

    def _get_observations(self) -> dict:
        right_palm_pose_w = self.right_palm_pose_w()
        left_palm_pose_w = self.left_palm_pose_w()

        table_forces = self.table_contact_sensor.data.force_matrix_w
        assert table_forces.shape == (
            self.num_envs,
            1,
            len(TABLE_CONTACT_SENSOR_ROBOT_LINKS),
            NUM_XYZ,
        ), (
            f"table_forces.shape: {table_forces.shape} != (self.num_envs, 1, len(TABLE_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ): {(self.num_envs, 1, len(TABLE_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ)}"
        )
        max_table_force = (
            table_forces.squeeze(dim=1).norm(dim=-1, p=2).max(dim=-1).values
        )
        assert max_table_force.shape == (self.num_envs,), (
            f"max_table_force.shape: {max_table_force.shape} != (self.num_envs,): {(self.num_envs,)}"
        )

        object_forces = self.object_contact_sensor.data.force_matrix_w
        assert object_forces.shape == (
            self.num_envs,
            1,
            len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS),
            NUM_XYZ,
        ), (
            f"object_forces.shape: {object_forces.shape} != (self.num_envs, 1, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ): {(self.num_envs, 1, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ)}"
        )
        object_forces = object_forces.squeeze(dim=1)
        object_force_norms = object_forces.norm(dim=-1, p=2)
        assert object_force_norms.shape == (
            self.num_envs,
            len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS),
        ), (
            f"object_force_norms.shape: {object_force_norms.shape} != (self.num_envs, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS)): {(self.num_envs, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS))}"
        )
        object_contacts = object_force_norms > 0.01

        obs_dict = {
            "q": (
                self.robot.data.joint_pos
                if INCLUDE_Q_OBS
                else torch.zeros(self.num_envs, 0, device=self.device)
            ),
            "qd": (
                self.robot.data.joint_vel
                if INCLUDE_QD_OBS
                else torch.zeros(self.num_envs, 0, device=self.device)
            ),
            "right_fingertip_positions": (
                self.right_fingertip_positions_w()
                - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, NUM_FINGERS * NUM_XYZ),
            "left_fingertip_positions": (
                self.left_fingertip_positions_w()
                - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, NUM_FINGERS * NUM_XYZ),
            "right_palm_position": right_palm_pose_w[:, :3] - self.scene.env_origins,
            "right_palm_orientation": right_palm_pose_w[:, 3:],
            "left_palm_position": left_palm_pose_w[:, :3] - self.scene.env_origins,
            "left_palm_orientation": left_palm_pose_w[:, 3:],
            "object_keypoint_positions": (
                self.object_keypoint_positions_w
                - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, NUM_OBJECT_KEYPOINTS * NUM_XYZ),
            "goal_object_keypoint_positions": (
                self.goal_object_keypoint_positions_w
                - self.scene.env_origins.unsqueeze(dim=1)
            ).reshape(self.num_envs, NUM_OBJECT_KEYPOINTS * NUM_XYZ),
            "goal_right_palm_position": (
                self.goal_right_palm_pose_w()[:, :3] - self.scene.env_origins
                if INCLUDE_HAND_TRACKING_REWARD
                else torch.zeros(self.num_envs, 0, device=self.device)
            ),
            "goal_left_palm_position": (
                self.goal_left_palm_pose_w()[:, :3] - self.scene.env_origins
                if INCLUDE_HAND_TRACKING_REWARD
                else torch.zeros(self.num_envs, 0, device=self.device)
            ),
            "future_goal_right_palm_positions": (
                self.future_goal_right_palm_poses[:, :, :3].reshape(
                    self.num_envs, NUM_FUTURE_PALM_GOAL_OBS * NUM_XYZ
                )
                if INCLUDE_HAND_TRACKING_REWARD
                else torch.zeros(self.num_envs, 0, device=self.device)
            ),
            "future_goal_left_palm_positions": (
                self.future_goal_left_palm_poses[:, :, :3].reshape(
                    self.num_envs, NUM_FUTURE_PALM_GOAL_OBS * NUM_XYZ
                )
                if INCLUDE_HAND_TRACKING_REWARD
                else torch.zeros(self.num_envs, 0, device=self.device)
            ),
            "future_goal_object_keypoint_positions": (
                self.future_goal_object_keypoint_positions.reshape(
                    self.num_envs, NUM_FUTURE_GOAL_OBS * NUM_OBJECT_KEYPOINTS * NUM_XYZ
                )
            ),
            "prev_actions": self.prev_raw_actions.reshape(self.num_envs, -1),
            "right_palm_linvel": self.right_palm_linvel(),
            "left_palm_linvel": self.left_palm_linvel(),
            "right_fingertip_linvels": self.right_fingertip_linvels().reshape(
                self.num_envs, -1
            ),
            "left_fingertip_linvels": self.left_fingertip_linvels().reshape(
                self.num_envs, -1
            ),
            "object_linvel": self.object_linvel,
            "object_angvel": self.object_angvel,
            "smallest_this_episode_right_index_fingertip_to_object_dist": self.smallest_this_episode_right_index_fingertip_to_object_dist.reshape(
                self.num_envs, -1
            ),
            "smallest_this_episode_left_index_fingertip_to_object_dist": self.smallest_this_episode_left_index_fingertip_to_object_dist.reshape(
                self.num_envs, -1
            ),
            "smallest_this_episode_object_to_goal_dist": self.smallest_this_episode_object_to_goal_dist.reshape(
                self.num_envs, -1
            ),
            "episode_length_buf": self.episode_length_buf.reshape(self.num_envs, -1),
            "object_is_lifted": self.object_is_lifted.reshape(self.num_envs, -1),
            "object_has_been_lifted_this_episode": self.object_has_been_lifted_this_episode.reshape(
                self.num_envs, -1
            ),
            "object_contacts": (
                object_contacts.float().reshape(self.num_envs, -1)
                if CONTACT_OBS_TYPE == "contacts"
                else object_forces.reshape(self.num_envs, -1)
            ),
        }
        if FINGER_GOALS:
            obs_dict["right_finger_goal_position"] = (
                self.right_finger_goal_position_w - self.scene.env_origins
            )
            obs_dict["left_finger_goal_position"] = (
                self.left_finger_goal_position_w - self.scene.env_origins
            )
        if FILTER_ARM_ACTIONS:
            if USE_FABRIC:
                obs_dict["fabric_palm_target"] = self.fabric_palm_target
            else:
                obs_dict["filtered_arm_position_targets"] = (
                    self.filtered_arm_position_targets
                )

        if USE_FABRIC and INCLUDE_FABRIC_OBS:
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
            if SAVE_OBS_HISTORY:
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
        if SAVE_OBS_HISTORY:
            batch_idx = torch.arange(self.num_envs, device=obs.device)  # shape (B,)
            time_idx = self.episode_length_buf  # shape (B,)
            self.obs_history[batch_idx, time_idx, :] = obs.detach().clone()

        ZERO_OBS = False  # Set to True to debug
        if ZERO_OBS:
            obs = torch.zeros(
                self.num_envs, self.cfg.observation_space, device=self.device
            )

        assert obs.shape == (self.num_envs, self.cfg.observation_space), (
            f"obs.shape: {obs.shape} != (self.num_envs, self.cfg.observation_space): {(self.num_envs, self.cfg.observation_space)}"
        )

        # Add critic observations
        state_dict = {
            "obs": obs,
        }
        for k, v in state_dict.items():
            if v.ndim != 2:
                print(colored(f"{k}: {v.shape} (WRONG)", "red"))

        for k, v in state_dict.items():
            if torch.isnan(v).any():
                nan_env_ids = torch.where(torch.isnan(v))[0]
                print(colored(f"{k}: {v.shape} (NAN) at {nan_env_ids}", "red"))

        state = torch.cat([state_dict[key] for key in state_dict], dim=-1)

        observations = {"policy": obs, "critic": state}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # fmt: off
        if FINGER_GOALS:
            self.individual_reward_bufs = {
                "right_index_fingertip_to_goal_dist": -(self.right_index_fingertip_position_w() - self.right_finger_goal_position_w).norm(dim=-1, p=2),
                "left_index_fingertip_to_goal_dist": -(self.left_index_fingertip_position_w() - self.left_finger_goal_position_w).norm(dim=-1, p=2),
            }
        else:
            right_index_fingertip_to_object_dist = (
                self.right_index_fingertip_position_w() - self.object_position_w
            ).norm(dim=-1, p=2)
            left_index_fingertip_to_object_dist = (
                self.left_index_fingertip_position_w() - self.object_position_w
            ).norm(dim=-1, p=2)
            right_improvement = (self.smallest_this_episode_right_index_fingertip_to_object_dist - right_index_fingertip_to_object_dist).clip(min=0.0)
            left_improvement = (self.smallest_this_episode_left_index_fingertip_to_object_dist - left_index_fingertip_to_object_dist).clip(min=0.0)
            object_goal_dist = (self.object_position_w - self.goal_object_position_w).norm(dim=-1, p=2)
            object_goal_improvement = (self.smallest_this_episode_object_to_goal_dist - object_goal_dist).clip(min=0.0)

            table_forces = self.table_contact_sensor.data.force_matrix_w
            assert table_forces.shape == (self.num_envs, 1, len(TABLE_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ), (
                f"table_forces.shape: {table_forces.shape} != (self.num_envs, 1, len(TABLE_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ): {(self.num_envs, 1, len(TABLE_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ)}"
            )
            max_table_force = table_forces.squeeze(dim=1).norm(dim=-1, p=2).max(dim=-1).values
            assert max_table_force.shape == (self.num_envs,), (
                f"max_table_force.shape: {max_table_force.shape} != (self.num_envs,): {(self.num_envs,)}"
            )

            object_forces = self.object_contact_sensor.data.force_matrix_w
            assert object_forces.shape == (self.num_envs, 1, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ), (
                f"object_forces.shape: {object_forces.shape} != (self.num_envs, 1, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ): {(self.num_envs, 1, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS), NUM_XYZ)}"
            )
            object_forces = object_forces.squeeze(dim=1)
            object_force_norms = object_forces.norm(dim=-1, p=2)
            assert object_force_norms.shape == (self.num_envs, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS)), (
                f"object_force_norms.shape: {object_force_norms.shape} != (self.num_envs, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS)): {(self.num_envs, len(OBJECT_CONTACT_SENSOR_ROBOT_LINKS))}"
            )
            object_contacts = object_force_norms > 0.01
            num_contacts = object_contacts.float().sum(dim=-1)

            # Increase reward if object is close to goal and fingertips are close to object
            object_goal_keypoint_dist = self.object_goal_keypoint_distance
            is_right_fingertips_object_close = (self.right_index_fingertip_position_w() - self.object_position_w).norm(dim=-1, p=2) < 0.3
            is_left_fingertips_object_close = (self.left_index_fingertip_position_w() - self.object_position_w).norm(dim=-1, p=2) < 0.3
            object_tracking_reward = torch.where(
                torch.logical_and(is_right_fingertips_object_close, is_left_fingertips_object_close),
                torch.exp(-object_goal_keypoint_dist * 10.0),
                torch.zeros(self.num_envs, device=self.device),
            )

            # Increase reward if object is lifted
            is_object_lifted = self.object_is_lifted
            is_goal_object_lifted = self.goal_object_is_lifted
            object_tracking_reward = torch.where(
                torch.logical_and(is_object_lifted, is_goal_object_lifted),
                5 * object_tracking_reward,
                object_tracking_reward,
            )

            # Reduce reward if object is far from goal
            object_tracking_reward = torch.where(
                object_goal_keypoint_dist < 0.25,
                object_tracking_reward,
                0.1 * object_tracking_reward,
            )

            self.individual_reward_bufs = {
                "right_index_fingertip_to_object_dist": right_improvement,
                "left_index_fingertip_to_object_dist": left_improvement,
                # "object_lifted": torch.logical_and(self.object_is_lifted, ~self.object_has_been_lifted_this_episode),
                # "object_to_goal_dist": object_goal_improvement,
                # "object_reached_goal": object_goal_dist < 0.1,
                "object_tracking_reward": object_tracking_reward,
            }
            if INCLUDE_CONTACT_REWARD:
                self.individual_reward_bufs["fingertip_contact"] = num_contacts
            if INCLUDE_HAND_TRACKING_REWARD:
                right_palm_to_target_dist = (self.right_palm_pose_w()[:, :3] - self.goal_right_palm_pose_w()[:, :3]).norm(dim=-1, p=2)
                left_palm_to_target_dist = (self.left_palm_pose_w()[:, :3] - self.goal_left_palm_pose_w()[:, :3]).norm(dim=-1, p=2)
                right_hand_tracking_reward = torch.exp(-right_palm_to_target_dist * 10.0)
                left_hand_tracking_reward = torch.exp(-left_palm_to_target_dist * 10.0)
                self.individual_reward_bufs["right_hand_tracking_reward"] = right_hand_tracking_reward
                self.individual_reward_bufs["left_hand_tracking_reward"] = left_hand_tracking_reward
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
                    "right_index_fingertip_to_object_dist": 1.0,  # max = init_dist(right, object) ~ 0.2
                    "left_index_fingertip_to_object_dist": 1.0,  # max = init_dist(left, object) ~ 0.2
                    # "object_lifted": 1.0,  # max = 1.0
                    # "object_to_goal_dist": 10.0,  # max = init_dist(object, goal) ~ 0.2
                    # "object_reached_goal": 0.1,  # max = num_steps ~ 75
                    "object_tracking_reward": 0.1,  # max = (1 or 5) * num_steps ~ 75 or 375
                }
                if INCLUDE_CONTACT_REWARD:
                    self.individual_reward_weights["fingertip_contact"] = (
                        0.002  # max = NUM_BIMANUAL * 17 * num_steps ~ 2500
                    )
                if INCLUDE_HAND_TRACKING_REWARD:
                    self.individual_reward_weights["right_hand_tracking_reward"] = (
                        0.02  # max = num_steps ~ 75
                    )
                    self.individual_reward_weights["left_hand_tracking_reward"] = (
                        0.02  # max = num_steps ~ 75
                    )

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
        # 1. _pre_physics_step() (compute actions)
        # 2. _apply_action() (apply actions)
        # 3. physics_step() (simulate)
        # 4. _compute_intermediate_values() (compute intermediate values)
        # 5. _get_dones() (compute done/time_out)
        # 6. _get_rewards() (compute rewards)
        # 7. _reset_idx() (reset envs)
        # 8. _get_observations() (compute observations)
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

        # Update smallest fingertip to object distance
        right_index_fingertip_to_object_dist = (
            self.right_index_fingertip_position_w() - self.object_position_w
        ).norm(dim=-1, p=2)
        self.smallest_this_episode_right_index_fingertip_to_object_dist = torch.where(
            right_index_fingertip_to_object_dist
            < self.smallest_this_episode_right_index_fingertip_to_object_dist,
            right_index_fingertip_to_object_dist,
            self.smallest_this_episode_right_index_fingertip_to_object_dist,
        )
        left_index_fingertip_to_object_dist = (
            self.left_index_fingertip_position_w() - self.object_position_w
        ).norm(dim=-1, p=2)
        self.smallest_this_episode_left_index_fingertip_to_object_dist = torch.where(
            left_index_fingertip_to_object_dist
            < self.smallest_this_episode_left_index_fingertip_to_object_dist,
            left_index_fingertip_to_object_dist,
            self.smallest_this_episode_left_index_fingertip_to_object_dist,
        )
        object_goal_dist = (self.object_position_w - self.goal_object_position_w).norm(
            dim=-1, p=2
        )
        self.smallest_this_episode_object_to_goal_dist = torch.where(
            object_goal_dist < self.smallest_this_episode_object_to_goal_dist,
            object_goal_dist,
            self.smallest_this_episode_object_to_goal_dist,
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

        # Set to True to debug
        DEBUG_NO_RESET = False
        if DEBUG_NO_RESET:
            return died, time_out

        died = torch.where(self.object_fallen_off_table, torch.ones_like(died), died)
        died = torch.where(
            (self.right_index_fingertip_position_w() - self.object_position_w).norm(
                dim=-1, p=2
            )
            > 0.5,
            torch.ones_like(died),
            died,
        )
        died = torch.where(
            (self.left_index_fingertip_position_w() - self.object_position_w).norm(
                dim=-1, p=2
            )
            > 0.5,
            torch.ones_like(died),
            died,
        )

        return died, time_out

    #### DONES END ####

    #### RESET START ####
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # Must be called before super()._reset_idx()
        # Since it uses episode_length_buf, which gets reset
        self._update_metrics(env_ids)

        self.robot.reset(env_ids)
        self.object.reset(env_ids)
        # self.contact_sensor.reset(env_ids)
        self.table_contact_sensor.reset(env_ids)
        self.object_contact_sensor.reset(env_ids)
        for fingertip_contact_sensor in self.fingertip_contact_sensors.values():
            fingertip_contact_sensor.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset robot
        self._reset_robot(env_ids)

        # Reset object
        self._reset_object(env_ids)

        # Must be done after _reset_robot() and _reset_object()
        # Since it uses the newly sampled initial robot and object and goal poses
        self._reset_state(env_ids)

    def _reset_robot(self, env_ids: torch.Tensor):
        # joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos = self.robot_custom_default_joint_pos[env_ids].clone()
        joint_pos *= math_utils.sample_uniform(
            *(0.95, 1.05), joint_pos.shape, joint_pos.device
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

    def _reset_object(self, env_ids: torch.Tensor):
        # object_pose = self._sample_initial_object_pose(env_ids)
        # final_object_pose = self._sample_final_object_pose(env_ids)
        goal_T_R_Os = self.goal_T_R_Os[
            self.episode_length_buf[env_ids].clip(max=self.goal_T_R_Os.shape[0] - 1)
        ]
        object_pos = goal_T_R_Os[:, :3, 3] + self.scene.env_origins[env_ids]
        object_pos[:, 2] += 0.02  # Buffer to avoid collision with table
        object_quat_wxyz = matrix_to_quat_wxyz(goal_T_R_Os[:, :3, :3])
        object_pose = torch.cat([object_pos, object_quat_wxyz], dim=-1)
        goal_object_pos = goal_T_R_Os[:, :3, 3] + self.scene.env_origins[env_ids]
        goal_object_quat_wxyz = matrix_to_quat_wxyz(goal_T_R_Os[:, :3, :3])
        goal_object_pose = torch.cat([goal_object_pos, goal_object_quat_wxyz], dim=-1)
        self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids)

        self.object.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )
        self.goal_object.write_root_pose_to_sim(goal_object_pose, env_ids=env_ids)
        # self.goal_object.write_root_velocity_to_sim(
        #     torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        # )

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

            if FILTER_ARM_ACTIONS:
                # NOTE: This actually doesn't quite work on the first run
                # Because the robot is not initialized yet
                # Probably not too big of a deal though
                self.filtered_arm_position_targets = self.robot.data.joint_pos[
                    :, : NUM_ARM_JOINTS * NUM_BIMANUAL
                ]
            self.sampled_raw_actions = sample_uniform_tensor(
                low=torch.tensor([-1.0] * self.cfg.action_space, device=self.device),
                high=torch.tensor([1.0] * self.cfg.action_space, device=self.device),
                N=self.num_envs,
            )

            if FINGER_GOALS:
                self.right_finger_goal_position_w = (
                    self._sample_right_finger_goal_position(env_ids)
                )
                self.left_finger_goal_position_w = (
                    self._sample_left_finger_goal_position(env_ids)
                )

            self.object_has_been_lifted_this_episode = torch.zeros_like(
                self.object_is_lifted
            )

            if USE_FABRIC:
                # NOTE: This actually doesn't quite work on the first run
                # Because the robot is not initialized yet
                # Probably not too big of a deal though
                self.fabric_q = isaaclab_to_fabric_joint_order_torch(
                    self.robot.data.joint_pos.clone().float()
                )
                self.fabric_qd = torch.zeros_like(self.fabric_q)
                self.fabric_qdd = torch.zeros_like(self.fabric_q)

                self.fabric_palm_target = torch.cat(
                    [
                        self.pose_w_to_xyzZYX(self.right_palm_pose_w()),
                        self.pose_w_to_xyzZYX(self.left_palm_pose_w()),
                    ],
                    dim=1,
                )
            if SAVE_OBS_HISTORY:
                self.obs_history = torch.zeros(
                    self.num_envs,
                    self.max_episode_length,
                    self.cfg.observation_space,
                    device=self.device,
                )
            self.smallest_this_episode_right_index_fingertip_to_object_dist = (
                self.right_index_fingertip_position_w() - self.object_position_w
            ).norm(dim=-1, p=2)
            self.smallest_this_episode_left_index_fingertip_to_object_dist = (
                self.left_index_fingertip_position_w() - self.object_position_w
            ).norm(dim=-1, p=2)
            self.smallest_this_episode_object_to_goal_dist = (
                self.object_position_w - self.goal_object_position_w
            ).norm(dim=-1, p=2)

            self.keyboard_external_force_w = torch.zeros(
                self.num_envs, NUM_XYZ, device=self.device
            )
            self.goal_float_idx = torch.zeros(self.num_envs, device=self.device)
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

            if FILTER_ARM_ACTIONS:
                self.filtered_arm_position_targets[env_ids] = self.robot.data.joint_pos[
                    env_ids, : NUM_ARM_JOINTS * NUM_BIMANUAL
                ]
            self.sampled_raw_actions[env_ids] = sample_uniform_tensor(
                low=torch.tensor([-1.0] * self.cfg.action_space, device=self.device),
                high=torch.tensor([1.0] * self.cfg.action_space, device=self.device),
                N=len(env_ids),
            )

            if FINGER_GOALS:
                self.right_finger_goal_position_w[env_ids] = (
                    self._sample_right_finger_goal_position(env_ids)
                )
                self.left_finger_goal_position_w[env_ids] = (
                    self._sample_left_finger_goal_position(env_ids)
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
                self.fabric_palm_target[env_ids] = torch.cat(
                    [
                        self.pose_w_to_xyzZYX(self.right_palm_pose_w()[env_ids]),
                        self.pose_w_to_xyzZYX(self.left_palm_pose_w()[env_ids]),
                    ],
                    dim=1,
                )
            if SAVE_OBS_HISTORY:
                self.obs_history[env_ids] = torch.zeros(
                    len(env_ids),
                    self.max_episode_length,
                    self.cfg.observation_space,
                    device=self.device,
                )
            self.smallest_this_episode_right_index_fingertip_to_object_dist[env_ids] = (
                self.right_index_fingertip_position_w()[env_ids]
                - self.object_position_w[env_ids]
            ).norm(dim=-1, p=2)
            self.smallest_this_episode_left_index_fingertip_to_object_dist[env_ids] = (
                self.left_index_fingertip_position_w()[env_ids]
                - self.object_position_w[env_ids]
            ).norm(dim=-1, p=2)
            self.smallest_this_episode_object_to_goal_dist[env_ids] = (
                self.object_position_w[env_ids] - self.goal_object_position_w[env_ids]
            ).norm(dim=-1, p=2)

            self.keyboard_external_force_w[env_ids] = torch.zeros(
                len(env_ids), NUM_XYZ, device=self.device
            )
            self.goal_float_idx[env_ids] = torch.zeros(len(env_ids), device=self.device)

    def _sample_right_finger_goal_position(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.2, -0.5, 0.05], device=self.device),
            high=torch.tensor([0.5, -0.1, 0.5], device=self.device),
            N=len(env_ids),
        )

    def _sample_left_finger_goal_position(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor([-0.2, 0.1, 0.05], device=self.device),
            high=torch.tensor([0.5, 0.5, 0.5], device=self.device),
            N=len(env_ids),
        )

    def _sample_initial_object_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        position = self.table_position[env_ids] + sample_uniform_tensor(
            low=torch.tensor(
                [-0.02, -0.02, OBJECT_LENGTH_Z / 2 + 0.02], device=self.device
            ),
            high=torch.tensor(
                [0.02, 0.02, OBJECT_LENGTH_Z / 2 + 0.03], device=self.device
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
                [-0.02, -0.02, OBJECT_LENGTH_Z / 2 + 0.3], device=self.device
            ),
            high=torch.tensor(
                [0.02, 0.02, OBJECT_LENGTH_Z / 2 + 0.5], device=self.device
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
            if not hasattr(self, "right_palm_pose_visualizer"):
                self.right_palm_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_palm_pose_visualizer
                )
            if not hasattr(self, "left_palm_pose_visualizer"):
                self.left_palm_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_palm_pose_visualizer
                )
            if USE_FABRIC:
                if not hasattr(self, "right_fabric_palm_target_pose_visualizer"):
                    self.right_fabric_palm_target_pose_visualizer = (
                        VisualizationMarkers(
                            self.cfg.right_fabric_palm_target_pose_visualizer
                        )
                    )
                if not hasattr(self, "left_fabric_palm_target_pose_visualizer"):
                    self.left_fabric_palm_target_pose_visualizer = VisualizationMarkers(
                        self.cfg.left_fabric_palm_target_pose_visualizer
                    )

            if FINGER_GOALS:
                if not hasattr(self, "right_finger_goal_visualizer"):
                    self.right_finger_goal_visualizer = VisualizationMarkers(
                        self.cfg.right_finger_goal_visualizer
                    )
                if not hasattr(self, "left_finger_goal_visualizer"):
                    self.left_finger_goal_visualizer = VisualizationMarkers(
                        self.cfg.left_finger_goal_visualizer
                    )

            if not hasattr(self, "object_keypoint_visualizers"):
                self.object_keypoint_visualizers = [
                    VisualizationMarkers(
                        self.cfg.object_keypoint_visualizer.replace(
                            prim_path=f"{self.cfg.object_keypoint_visualizer.prim_path}_{i}"
                        )
                    )
                    for i in range(NUM_OBJECT_KEYPOINTS)
                ]
            if not hasattr(self, "goal_object_keypoint_visualizers"):
                self.goal_object_keypoint_visualizers = [
                    VisualizationMarkers(
                        self.cfg.goal_object_keypoint_visualizer.replace(
                            prim_path=f"{self.cfg.goal_object_keypoint_visualizer.prim_path}_{i}"
                        )
                    )
                    for i in range(NUM_OBJECT_KEYPOINTS)
                ]
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
            if self.VISUALIZE_FABRIC_SPHERES:
                if not hasattr(self, "collision_sphere_visualizers"):
                    self.collision_sphere_visualizers = [
                        VisualizationMarkers(
                            self.cfg.collision_sphere_visualizer.replace(
                                prim_path=f"{self.cfg.collision_sphere_visualizer.prim_path}_{i}"
                            )
                        )
                        for i in range(NUM_FABRIC_SPHERES)
                    ]
            if self.VISUALIZE_FABRIC_WORLD:
                if not hasattr(self, "fabric_world_visualizers"):
                    self.fabric_world_visualizers = [
                        VisualizationMarkers(
                            self.cfg.fabric_world_visualizer.replace(
                                prim_path=f"{self.cfg.fabric_world_visualizer.prim_path}_{i}"
                            )
                        )
                        for i in range(NUM_FABRIC_WORLD_CUBES)
                    ]
            if not hasattr(self, "goal_right_palm_pose_visualizer"):
                self.goal_right_palm_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_right_palm_pose_visualizer
                )
            if not hasattr(self, "goal_left_palm_pose_visualizer"):
                self.goal_left_palm_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_left_palm_pose_visualizer
                )

            # set their visibility to true
            self.right_palm_pose_visualizer.set_visibility(True)
            self.left_palm_pose_visualizer.set_visibility(True)
            if USE_FABRIC:
                self.right_fabric_palm_target_pose_visualizer.set_visibility(True)
                self.left_fabric_palm_target_pose_visualizer.set_visibility(True)
            if FINGER_GOALS:
                self.right_finger_goal_visualizer.set_visibility(True)
                self.left_finger_goal_visualizer.set_visibility(True)
            if hasattr(self, "object_keypoint_visualizers"):
                for visualizer in self.object_keypoint_visualizers:
                    visualizer.set_visibility(True)
            if hasattr(self, "goal_object_keypoint_visualizers"):
                for visualizer in self.goal_object_keypoint_visualizers:
                    visualizer.set_visibility(True)
            self.right_fingertip_visualizer.set_visibility(True)
            self.left_fingertip_visualizer.set_visibility(True)
            self.progress_visualizer.set_visibility(True)
            self.progress_full_visualizer.set_visibility(True)
            if self.VISUALIZE_FABRIC_SPHERES:
                for visualizer in self.collision_sphere_visualizers:
                    visualizer.set_visibility(True)
            elif hasattr(self, "collision_sphere_visualizers"):
                for visualizer in self.collision_sphere_visualizers:
                    visualizer.set_visibility(False)
            if self.VISUALIZE_FABRIC_WORLD:
                for visualizer in self.fabric_world_visualizers:
                    visualizer.set_visibility(True)
            elif hasattr(self, "fabric_world_visualizers"):
                for visualizer in self.fabric_world_visualizers:
                    visualizer.set_visibility(False)
            self.goal_right_palm_pose_visualizer.set_visibility(True)
            self.goal_left_palm_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "right_palm_pose_visualizer"):
                self.right_palm_pose_visualizer.set_visibility(False)
            if hasattr(self, "left_palm_pose_visualizer"):
                self.left_palm_pose_visualizer.set_visibility(False)
            if USE_FABRIC:
                if hasattr(self, "right_fabric_palm_target_pose_visualizer"):
                    self.right_fabric_palm_target_pose_visualizer.set_visibility(False)
                if hasattr(self, "left_fabric_palm_target_pose_visualizer"):
                    self.left_fabric_palm_target_pose_visualizer.set_visibility(False)
            if FINGER_GOALS:
                if hasattr(self, "right_finger_goal_visualizer"):
                    self.right_finger_goal_visualizer.set_visibility(False)
                if hasattr(self, "left_finger_goal_visualizer"):
                    self.left_finger_goal_visualizer.set_visibility(False)
            if hasattr(self, "object_keypoint_visualizers"):
                for visualizer in self.object_keypoint_visualizers:
                    visualizer.set_visibility(False)
            if hasattr(self, "goal_object_keypoint_visualizers"):
                for visualizer in self.goal_object_keypoint_visualizers:
                    visualizer.set_visibility(False)
            if hasattr(self, "right_fingertip_visualizer"):
                self.right_fingertip_visualizer.set_visibility(False)
            if hasattr(self, "left_fingertip_visualizer"):
                self.left_fingertip_visualizer.set_visibility(False)
            if hasattr(self, "progress_visualizer"):
                self.progress_visualizer.set_visibility(False)
            if hasattr(self, "progress_full_visualizer"):
                self.progress_full_visualizer.set_visibility(False)
            if self.VISUALIZE_FABRIC_SPHERES:
                if hasattr(self, "collision_sphere_visualizers"):
                    for visualizer in self.collision_sphere_visualizers:
                        visualizer.set_visibility(False)
            if self.VISUALIZE_FABRIC_WORLD:
                if hasattr(self, "fabric_world_visualizers"):
                    for visualizer in self.fabric_world_visualizers:
                        visualizer.set_visibility(False)
            if hasattr(self, "goal_right_palm_pose_visualizer"):
                self.goal_right_palm_pose_visualizer.set_visibility(False)
            if hasattr(self, "goal_left_palm_pose_visualizer"):
                self.goal_left_palm_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Make sure the robot is initialized
        if not self.robot.is_initialized:
            return

        POSE_SCALE = [0.1, 0.1, 0.1]
        SPHERE_SCALE = [0.03, 0.03, 0.03]

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
            # Actions are in robot frame
            # [RIGHT xyz, RIGHT euler_ZYX, LEFT xyz, LEFT euler_ZYX]
            right_fabric_palm_target_pose_w = self.xyzZYX_to_pose_w(
                self.fabric_palm_target[:, :6]
            )
            left_fabric_palm_target_pose_w = self.xyzZYX_to_pose_w(
                self.fabric_palm_target[:, 6:12]
            )
            self.right_fabric_palm_target_pose_visualizer.visualize(
                translations=right_fabric_palm_target_pose_w[:, :3],
                orientations=right_fabric_palm_target_pose_w[:, 3:],
                scales=torch.tensor(
                    (np.array(POSE_SCALE) * 0.3).tolist(), device=self.device
                )
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
            self.left_fabric_palm_target_pose_visualizer.visualize(
                translations=left_fabric_palm_target_pose_w[:, :3],
                orientations=left_fabric_palm_target_pose_w[:, 3:],
                scales=torch.tensor(
                    (np.array(POSE_SCALE) * 0.3).tolist(), device=self.device
                )
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
        if FINGER_GOALS:
            self.right_finger_goal_visualizer.visualize(
                translations=self.right_finger_goal_position_w,
                scales=torch.tensor(SPHERE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
            self.left_finger_goal_visualizer.visualize(
                translations=self.left_finger_goal_position_w,
                scales=torch.tensor(SPHERE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )

        object_keypoint_positions = self.object_keypoint_positions_w
        for i in range(NUM_OBJECT_KEYPOINTS):
            self.object_keypoint_visualizers[i].visualize(
                translations=object_keypoint_positions[:, i, :],
                scales=torch.tensor(SPHERE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )
        goal_object_keypoint_positions = self.goal_object_keypoint_positions_w
        for i in range(NUM_OBJECT_KEYPOINTS):
            self.goal_object_keypoint_visualizers[i].visualize(
                translations=goal_object_keypoint_positions[:, i, :],
                scales=torch.tensor(SPHERE_SCALE, device=self.device)
                .unsqueeze(dim=0)
                .repeat_interleave(self.num_envs, dim=0),
            )

        VISUALIZE_FINGER_TIP = False
        if VISUALIZE_FINGER_TIP:
            fingertip_scale = SPHERE_SCALE
        else:
            fingertip_scale = np.array(SPHERE_SCALE) * 0.001
        self.right_fingertip_visualizer.visualize(
            translations=self.right_index_fingertip_position_w(),
            scales=torch.tensor(fingertip_scale, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.left_fingertip_visualizer.visualize(
            translations=self.left_index_fingertip_position_w(),
            scales=torch.tensor(fingertip_scale, device=self.device)
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

        if self.VISUALIZE_FABRIC_SPHERES:
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

        if self.VISUALIZE_FABRIC_WORLD:
            from isaaclab_tasks.direct.tyler.bimanual.utils.fabric_world import (
                transform_str_to_T,
            )

            assert len(self.fabric_world_dict) == NUM_FABRIC_WORLD_CUBES, (
                f"NUM_FABRIC_WORLD_CUBES: {NUM_FABRIC_WORLD_CUBES}, len(self.fabric_world_dict): {len(self.fabric_world_dict)}"
            )
            for i, (_name, cuboid_dict) in enumerate(self.fabric_world_dict.items()):
                T = (
                    torch.from_numpy(transform_str_to_T(cuboid_dict["transform"]))
                    .float()
                    .to(self.device)
                )
                scaling = (
                    torch.from_numpy(
                        np.array([float(x) for x in cuboid_dict["scaling"].split(" ")])
                    )
                    .float()
                    .to(self.device)
                )
                assert T.shape == (4, 4), f"T shape: {T.shape}"
                assert scaling.shape == (3,), f"scaling shape: {scaling.shape}"
                translations = self.scene.env_origins + T[:3, 3].unsqueeze(dim=0)
                quat_wxyz = matrix_to_quat_wxyz(T[:3, :3].unsqueeze(dim=0))
                self.fabric_world_visualizers[i].visualize(
                    translations=translations,
                    orientations=quat_wxyz.repeat_interleave(self.num_envs, dim=0),
                    scales=scaling.unsqueeze(dim=0).repeat_interleave(
                        self.num_envs, dim=0
                    ),
                )
        goal_right_palm_pose_w = self.goal_right_palm_pose_w()
        goal_left_palm_pose_w = self.goal_left_palm_pose_w()
        self.goal_right_palm_pose_visualizer.visualize(
            translations=goal_right_palm_pose_w[:, :3],
            orientations=goal_right_palm_pose_w[:, 3:],
            scales=torch.tensor(POSE_SCALE, device=self.device)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0),
        )
        self.goal_left_palm_pose_visualizer.visualize(
            translations=goal_left_palm_pose_w[:, :3],
            orientations=goal_left_palm_pose_w[:, 3:],
            scales=torch.tensor(POSE_SCALE, device=self.device)
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
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.W,
                        func=self._toggle_fabric_world,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.C,
                        func=self._toggle_fabric_spheres,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.D,
                        func=self._toggle_debug_vis,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.LEFT,
                        func=self._apply_external_force_neg_y,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.RIGHT,
                        func=self._apply_external_force_pos_y,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.UP,
                        func=self._apply_external_force_neg_x,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.DOWN,
                        func=self._apply_external_force_pos_x,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.PAGE_UP,
                        func=self._apply_external_force_pos_z,
                        args=[],
                    ),
                    KeyboardCommand(
                        key=carb.input.KeyboardInput.PAGE_DOWN,
                        func=self._apply_external_force_neg_z,
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

        output_filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz"
        np.savez(
            output_filename,
            plot_data=plot_data,
            joint_names=self.robot.data.joint_names,
            episode_frac=episode_frac,
        )
        print(colored(f"Saved data to {output_filename}", "green"))

    def _toggle_fabric_spheres(self):
        self.VISUALIZE_FABRIC_SPHERES = not self.VISUALIZE_FABRIC_SPHERES
        print(
            colored(
                f"Toggling fabric spheres: {self.VISUALIZE_FABRIC_SPHERES}", "green"
            )
        )
        self.set_debug_vis(self.DEBUG_VIS)

    def _toggle_fabric_world(self):
        self.VISUALIZE_FABRIC_WORLD = not self.VISUALIZE_FABRIC_WORLD
        print(colored(f"Toggling fabric world: {self.VISUALIZE_FABRIC_WORLD}", "green"))
        self.set_debug_vis(self.DEBUG_VIS)

    def _toggle_debug_vis(self):
        self.DEBUG_VIS = not self.DEBUG_VIS
        print(colored(f"Toggling debug vis: {self.DEBUG_VIS}", "green"))
        self.set_debug_vis(self.DEBUG_VIS)

    def _apply_external_force_neg_y(self):
        print(colored("In apply_external_force_neg_y", "green"))
        ENV_ID = 0
        self.keyboard_external_force_w[ENV_ID, 1] = -FORCE_MAG

    def _apply_external_force_pos_y(self):
        print(colored("In apply_external_force_pos_y", "green"))
        ENV_ID = 0
        self.keyboard_external_force_w[ENV_ID, 1] = FORCE_MAG

    def _apply_external_force_neg_x(self):
        print(colored("In apply_external_force_neg_x", "green"))
        ENV_ID = 0
        self.keyboard_external_force_w[ENV_ID, 0] = -FORCE_MAG

    def _apply_external_force_pos_x(self):
        print(colored("In apply_external_force_pos_x", "green"))
        ENV_ID = 0
        self.keyboard_external_force_w[ENV_ID, 0] = FORCE_MAG

    def _apply_external_force_pos_z(self):
        print(colored("In apply_external_force_pos_z", "green"))
        ENV_ID = 0
        self.keyboard_external_force_w[ENV_ID, 2] = FORCE_MAG

    def _apply_external_force_neg_z(self):
        print(colored("In apply_external_force_neg_z", "green"))
        ENV_ID = 0
        self.keyboard_external_force_w[ENV_ID, 2] = -FORCE_MAG

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
    def object_linvel(self) -> torch.Tensor:
        assert self.object.data.body_vel_w.shape == (self.num_envs, 1, NUM_XYZ * 2), (
            f"Object linvel shape: {self.object.data.body_vel_w.shape}"
        )
        return self.object.data.body_vel_w[:, 0, :NUM_XYZ]

    @property
    def object_angvel(self) -> torch.Tensor:
        assert self.object.data.body_vel_w.shape == (self.num_envs, 1, NUM_XYZ * 2), (
            f"Object angvel shape: {self.object.data.body_vel_w.shape}"
        )
        return self.object.data.body_vel_w[:, 0, NUM_XYZ:]

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

    #### TENSOR SLICE PROPERTIES END ####

    #### OBJECT COMPUTATIONS START ####
    @property
    def object_keypoint_positions_w(self) -> torch.Tensor:
        object_keypoint_offsets = (
            torch.tensor(
                OBJECT_KEYPOINT_OFFSETS,
                device=self.device,
                dtype=self.object_position_w.dtype,
            )
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0)
        )
        assert object_keypoint_offsets.shape == (
            self.num_envs,
            NUM_OBJECT_KEYPOINTS,
            3,
        ), (
            f"Expected object_keypoint_offsets to have shape (self.num_envs, NUM_OBJECT_KEYPOINTS, 3), got {object_keypoint_offsets.shape}"
        )
        return compute_keypoint_positions(
            pos=self.object_position_w,
            quat_xyzw=self.object_orientation[:, [1, 2, 3, 0]],
            keypoint_offsets=object_keypoint_offsets,
        )

    @property
    def goal_object_keypoint_positions_w(self) -> torch.Tensor:
        object_keypoint_offsets = (
            torch.tensor(
                OBJECT_KEYPOINT_OFFSETS,
                device=self.device,
                dtype=self.object_position_w.dtype,
            )
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0)
        )
        assert object_keypoint_offsets.shape == (
            self.num_envs,
            NUM_OBJECT_KEYPOINTS,
            3,
        ), (
            f"Expected object_keypoint_offsets to have shape (self.num_envs, NUM_OBJECT_KEYPOINTS, 3), got {object_keypoint_offsets.shape}"
        )
        return compute_keypoint_positions(
            pos=self.goal_object_position_w,
            quat_xyzw=self.goal_object_orientation[:, [1, 2, 3, 0]],
            keypoint_offsets=object_keypoint_offsets,
        )

    @property
    def object_goal_keypoint_distance(self) -> torch.Tensor:
        distance = (
            (self.object_keypoint_positions_w - self.goal_object_keypoint_positions_w)
            .norm(dim=-1)
            .mean(dim=-1)
        )
        return distance

    @property
    def object_is_lifted(self) -> torch.Tensor:
        return (
            self.object_position_w[:, 2] > self.table_position[:, 2] + OBJECT_LENGTH_Z
        )

    @property
    def goal_object_is_lifted(self) -> torch.Tensor:
        return (
            self.goal_object_position_w[:, 2]
            > self.table_position[:, 2] + OBJECT_LENGTH_Z
        )

    @property
    def object_fallen_off_table(self) -> torch.Tensor:
        return (
            self.object_position_w[:, 2] < self.table_position[:, 2] - OBJECT_LENGTH_Z
        )

    #### OBJECT COMPUTATIONS END ####

    #### GOAL COMPUTATIONS START ####
    @property
    def future_goal_object_keypoint_positions(self) -> torch.Tensor:
        object_keypoint_offsets = (
            torch.tensor(
                OBJECT_KEYPOINT_OFFSETS,
                device=self.device,
                dtype=self.object_position_w.dtype,
            )
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs * NUM_FUTURE_GOAL_OBS, dim=0)
        )
        assert object_keypoint_offsets.shape == (
            self.num_envs * NUM_FUTURE_GOAL_OBS,
            NUM_OBJECT_KEYPOINTS,
            3,
        ), (
            f"Expected object_keypoint_offsets to have shape (self.num_envs, NUM_OBJECT_KEYPOINTS, 3), got {object_keypoint_offsets.shape}"
        )

        poses = self.future_goal_object_poses
        assert poses.shape == (
            self.num_envs,
            NUM_FUTURE_GOAL_OBS,
            7,
        ), f"poses shape: {poses.shape}"
        positions = poses[:, :, :3]
        orientations = poses[:, :, 3:]

        keypoint_positions = compute_keypoint_positions(
            pos=positions.reshape(self.num_envs * NUM_FUTURE_GOAL_OBS, NUM_XYZ),
            quat_xyzw=orientations.reshape(
                self.num_envs * NUM_FUTURE_GOAL_OBS, NUM_QUAT
            )[:, [1, 2, 3, 0]],
            keypoint_offsets=object_keypoint_offsets,
        ).reshape(self.num_envs, NUM_FUTURE_GOAL_OBS, NUM_OBJECT_KEYPOINTS, 3)
        return keypoint_positions

    @property
    def future_goal_object_poses(self) -> torch.Tensor:
        # Compute future idxs we want
        TIME_BETWEEN_GOALS_SECONDS = 0.5
        CONTROL_DT = self.cfg.sim.dt * self.cfg.decimation
        IDXS_BETWEEN_GOALS = TIME_BETWEEN_GOALS_SECONDS / CONTROL_DT
        relative_idxs = (
            torch.arange(1, NUM_FUTURE_GOAL_OBS + 1, device=self.device).float()
            * IDXS_BETWEEN_GOALS
        )
        current_idx = self.goal_float_idx
        assert relative_idxs.shape == (NUM_FUTURE_GOAL_OBS,), (
            f"relative_idxs shape: {relative_idxs.shape}"
        )
        assert current_idx.shape == (self.num_envs,), (
            f"current_idx shape: {current_idx.shape}"
        )
        NUM_GOAL_TIMESTEPS = self.goal_T_R_Os.shape[0]
        new_idxs = (
            (current_idx.unsqueeze(dim=1) + relative_idxs.unsqueeze(dim=0))
            .long()
            .clip(max=NUM_GOAL_TIMESTEPS - 1)
        )
        assert new_idxs.shape == (self.num_envs, NUM_FUTURE_GOAL_OBS), (
            f"new_idxs shape: {new_idxs.shape}"
        )

        # Extract future poses
        assert self.goal_T_R_Os.shape == (NUM_GOAL_TIMESTEPS, 4, 4), (
            f"goal_T_R_Os shape: {self.goal_T_R_Os.shape}"
        )
        future_T_R_Os = self.goal_T_R_Os[new_idxs]
        assert future_T_R_Os.shape == (self.num_envs, NUM_FUTURE_GOAL_OBS, 4, 4), (
            f"future_T_R_Os shape: {future_T_R_Os.shape}"
        )

        # Convert to poses
        future_positions = future_T_R_Os[:, :, :3, 3]
        future_orientations = matrix_to_quat_wxyz(future_T_R_Os[:, :, :3, :3])
        future_poses = torch.cat([future_positions, future_orientations], dim=-1)
        assert future_poses.shape == (
            self.num_envs,
            NUM_FUTURE_GOAL_OBS,
            7,
        ), f"future_poses shape: {future_poses.shape}"
        return future_poses

    @property
    def future_goal_right_palm_poses(self) -> torch.Tensor:
        # Compute future idxs we want
        TIME_BETWEEN_GOALS_SECONDS = 0.5
        CONTROL_DT = self.cfg.sim.dt * self.cfg.decimation
        IDXS_BETWEEN_GOALS = TIME_BETWEEN_GOALS_SECONDS / CONTROL_DT
        relative_idxs = (
            torch.arange(1, NUM_FUTURE_PALM_GOAL_OBS + 1, device=self.device).float()
            * IDXS_BETWEEN_GOALS
        )
        current_idx = self.goal_float_idx
        assert relative_idxs.shape == (NUM_FUTURE_PALM_GOAL_OBS,), (
            f"relative_idxs shape: {relative_idxs.shape}"
        )
        assert current_idx.shape == (self.num_envs,), (
            f"current_idx shape: {current_idx.shape}"
        )
        NUM_GOAL_TIMESTEPS = self.goal_right_T_R_Ps.shape[0]
        new_idxs = (
            (current_idx.unsqueeze(dim=1) + relative_idxs.unsqueeze(dim=0))
            .long()
            .clip(max=NUM_GOAL_TIMESTEPS - 1)
        )
        assert new_idxs.shape == (self.num_envs, NUM_FUTURE_PALM_GOAL_OBS), (
            f"new_idxs shape: {new_idxs.shape}"
        )

        # Extract future poses
        assert self.goal_right_T_R_Ps.shape == (NUM_GOAL_TIMESTEPS, 4, 4), (
            f"goal_right_T_R_Ps shape: {self.goal_right_T_R_Ps.shape}"
        )
        future_right_T_R_Ps = self.goal_right_T_R_Ps[new_idxs]
        assert future_right_T_R_Ps.shape == (
            self.num_envs,
            NUM_FUTURE_PALM_GOAL_OBS,
            4,
            4,
        ), f"future_right_T_R_Ps shape: {future_right_T_R_Ps.shape}"

        # Convert to poses
        future_positions = future_right_T_R_Ps[:, :, :3, 3]
        future_orientations = matrix_to_quat_wxyz(future_right_T_R_Ps[:, :, :3, :3])
        future_poses = torch.cat([future_positions, future_orientations], dim=-1)
        assert future_poses.shape == (
            self.num_envs,
            NUM_FUTURE_PALM_GOAL_OBS,
            7,
        ), f"future_poses shape: {future_poses.shape}"
        return future_poses

    @property
    def future_goal_left_palm_poses(self) -> torch.Tensor:
        # Compute future idxs we want
        TIME_BETWEEN_GOALS_SECONDS = 0.5
        CONTROL_DT = self.cfg.sim.dt * self.cfg.decimation
        IDXS_BETWEEN_GOALS = TIME_BETWEEN_GOALS_SECONDS / CONTROL_DT

        relative_idxs = (
            torch.arange(1, NUM_FUTURE_PALM_GOAL_OBS + 1, device=self.device).float()
            * IDXS_BETWEEN_GOALS
        )
        current_idx = self.goal_float_idx
        assert relative_idxs.shape == (NUM_FUTURE_PALM_GOAL_OBS,), (
            f"relative_idxs shape: {relative_idxs.shape}"
        )
        assert current_idx.shape == (self.num_envs,), (
            f"current_idx shape: {current_idx.shape}"
        )
        NUM_GOAL_TIMESTEPS = self.goal_left_T_R_Ps.shape[0]
        new_idxs = (
            (current_idx.unsqueeze(dim=1) + relative_idxs.unsqueeze(dim=0))
            .long()
            .clip(max=NUM_GOAL_TIMESTEPS - 1)
        )
        assert new_idxs.shape == (self.num_envs, NUM_FUTURE_PALM_GOAL_OBS), (
            f"new_idxs shape: {new_idxs.shape}"
        )

        # Extract future poses
        assert self.goal_left_T_R_Ps.shape == (NUM_GOAL_TIMESTEPS, 4, 4), (
            f"goal_left_T_R_Ps shape: {self.goal_left_T_R_Ps.shape}"
        )
        future_left_T_R_Ps = self.goal_left_T_R_Ps[new_idxs]
        assert future_left_T_R_Ps.shape == (
            self.num_envs,
            NUM_FUTURE_PALM_GOAL_OBS,
            4,
            4,
        ), f"future_left_T_R_Ps shape: {future_left_T_R_Ps.shape}"

        # Convert to poses
        future_positions = future_left_T_R_Ps[:, :, :3, 3]
        future_orientations = matrix_to_quat_wxyz(future_left_T_R_Ps[:, :, :3, :3])
        future_poses = torch.cat([future_positions, future_orientations], dim=-1)
        assert future_poses.shape == (
            self.num_envs,
            NUM_FUTURE_PALM_GOAL_OBS,
            7,
        ), f"future_left_palm_poses shape: {future_poses.shape}"
        return future_poses

    def goal_right_palm_pose_w(self) -> torch.Tensor:
        right_T_R_P = self.goal_right_T_R_Ps[
            self.goal_float_idx.long().clip(max=self.goal_right_T_R_Ps.shape[0] - 1)
        ]
        assert right_T_R_P.shape == (self.num_envs, 4, 4), (
            f"right_T_R_P shape: {right_T_R_P.shape}"
        )

        pos = right_T_R_P[:, :3, 3]
        rot_matrix = right_T_R_P[:, :3, :3]
        quat_wxyz = matrix_to_quat_wxyz(rot_matrix)

        # World frame
        pos_w = pos + self.scene.env_origins
        pose_w = torch.cat([pos_w, quat_wxyz], dim=-1)
        return pose_w

    def goal_left_palm_pose_w(self) -> torch.Tensor:
        left_T_R_P = self.goal_left_T_R_Ps[
            self.goal_float_idx.long().clip(max=self.goal_left_T_R_Ps.shape[0] - 1)
        ]
        assert left_T_R_P.shape == (self.num_envs, 4, 4), (
            f"left_T_R_P shape: {left_T_R_P.shape}"
        )

        pos = left_T_R_P[:, :3, 3]
        rot_matrix = left_T_R_P[:, :3, :3]
        quat_wxyz = matrix_to_quat_wxyz(rot_matrix)

        # World frame
        pos_w = pos + self.scene.env_origins
        pose_w = torch.cat([pos_w, quat_wxyz], dim=-1)
        return pose_w

    #### GOAL COMPUTATIONS END ####

    #### FABRIC TASKMAP FORWARD KINEMATICS START ####
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

    def right_palm_linvel(
        self, q: Optional[torch.Tensor] = None, qd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos
        if qd is None:
            qd = self.robot.data.joint_vel

        _, xd, _ = self.right_taskmap_helper(
            q=q,
            qd=qd,
        )
        palm_linvel = xd[:, RIGHT_PALM_LINK_IDX]
        return palm_linvel

    def left_palm_linvel(
        self, q: Optional[torch.Tensor] = None, qd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos
        if qd is None:
            qd = self.robot.data.joint_vel

        _, xd, _ = self.left_taskmap_helper(
            q=q,
            qd=qd,
        )
        palm_linvel = xd[:, LEFT_PALM_LINK_IDX]
        return palm_linvel

    def right_fingertip_positions_w(
        self, q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos

        x, _, _ = self.right_taskmap_helper(
            q=q,
        )
        index_pos = x[:, RIGHT_INDEX_FINGERTIP_LINK_IDX]
        middle_pos = x[:, RIGHT_MIDDLE_FINGERTIP_LINK_IDX]
        ring_pos = x[:, RIGHT_RING_FINGERTIP_LINK_IDX]
        thumb_pos = x[:, RIGHT_THUMB_FINGERTIP_LINK_IDX]

        positions = torch.stack(
            [index_pos, middle_pos, ring_pos, thumb_pos],
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
        index_pos = x[:, LEFT_INDEX_FINGERTIP_LINK_IDX]
        middle_pos = x[:, LEFT_MIDDLE_FINGERTIP_LINK_IDX]
        ring_pos = x[:, LEFT_RING_FINGERTIP_LINK_IDX]
        thumb_pos = x[:, LEFT_THUMB_FINGERTIP_LINK_IDX]

        positions = torch.stack(
            [index_pos, middle_pos, ring_pos, thumb_pos],
            dim=1,
        )
        # World frame
        positions_w = positions + self.scene.env_origins.unsqueeze(dim=1)
        return positions_w

    def right_fingertip_linvels(
        self, q: Optional[torch.Tensor] = None, qd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos
        if qd is None:
            qd = self.robot.data.joint_vel

        _, xd, _ = self.right_taskmap_helper(
            q=q,
            qd=qd,
        )
        index_linvel = xd[:, RIGHT_INDEX_FINGERTIP_LINK_IDX]
        middle_linvel = xd[:, RIGHT_MIDDLE_FINGERTIP_LINK_IDX]
        ring_linvel = xd[:, RIGHT_RING_FINGERTIP_LINK_IDX]
        thumb_linvel = xd[:, RIGHT_THUMB_FINGERTIP_LINK_IDX]

        linvels = torch.stack(
            [
                index_linvel,
                middle_linvel,
                ring_linvel,
                thumb_linvel,
            ],
            dim=1,
        )
        return linvels

    def left_fingertip_linvels(
        self, q: Optional[torch.Tensor] = None, qd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if q is None:
            q = self.robot.data.joint_pos
        if qd is None:
            qd = self.robot.data.joint_vel

        _, xd, _ = self.left_taskmap_helper(
            q=q,
            qd=qd,
        )
        index_linvel = xd[:, LEFT_INDEX_FINGERTIP_LINK_IDX]
        middle_linvel = xd[:, LEFT_MIDDLE_FINGERTIP_LINK_IDX]
        ring_linvel = xd[:, LEFT_RING_FINGERTIP_LINK_IDX]
        thumb_linvel = xd[:, LEFT_THUMB_FINGERTIP_LINK_IDX]

        linvels = torch.stack(
            [
                index_linvel,
                middle_linvel,
                ring_linvel,
                thumb_linvel,
            ],
            dim=1,
        )
        return linvels

    def right_index_fingertip_position_w(
        self, q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.right_fingertip_positions_w(q)[:, 0]

    def left_index_fingertip_position_w(
        self, q: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.left_fingertip_positions_w(q)[:, 0]

    #### FABRIC TASKMAP FORWARD KINEMATICS END ####

    #### CONVERSION FUNCTIONS START ####
    def xyzZYX_to_pose_w(self, xyzZYX: torch.Tensor) -> torch.Tensor:
        N = xyzZYX.shape[0]
        assert xyzZYX.shape == (N, 6), f"xyzZYX shape: {xyzZYX.shape}"
        pos = xyzZYX[:, :3]
        pos_w = pos + self.scene.env_origins
        euler_ZYX = xyzZYX[:, 3:]
        matrix = euler_angles_to_matrix(euler_ZYX, "ZYX")
        quat_wxyz = matrix_to_quat_wxyz(matrix)
        pose_w = torch.cat([pos_w, quat_wxyz], dim=1)
        assert pose_w.shape == (N, 7), f"Pose shape: {pose_w.shape}"
        return pose_w

    def pose_w_to_xyzZYX(self, pose_w: torch.Tensor) -> torch.Tensor:
        N = pose_w.shape[0]
        assert pose_w.shape == (N, 7), f"Pose shape: {pose_w.shape}"
        xyz = pose_w[:, :3] - self.scene.env_origins
        quat_wxyz = pose_w[:, 3:]
        matrix = quat_wxyz_to_matrix(quat_wxyz)
        euler_ZYX = matrix_to_euler_angles(matrix, "ZYX")
        xyzZYX = torch.cat([xyz, euler_ZYX], dim=1)
        assert xyzZYX.shape == (N, 6), f"xyzZYX shape: {xyzZYX.shape}"
        return xyzZYX

    #### CONVERSION FUNCTIONS END ####

    #### MODIFIABLE PROPERTIES START ####
    @property
    def VISUALIZE_FABRIC_SPHERES(self) -> bool:
        if not hasattr(self, "_VISUALIZE_FABRIC_SPHERES"):
            self._VISUALIZE_FABRIC_SPHERES = False
        return self._VISUALIZE_FABRIC_SPHERES

    @VISUALIZE_FABRIC_SPHERES.setter
    def VISUALIZE_FABRIC_SPHERES(self, value: bool):
        self._VISUALIZE_FABRIC_SPHERES = value

    @property
    def VISUALIZE_FABRIC_WORLD(self) -> bool:
        if not hasattr(self, "_VISUALIZE_FABRIC_WORLD"):
            self._VISUALIZE_FABRIC_WORLD = False
        return self._VISUALIZE_FABRIC_WORLD

    @VISUALIZE_FABRIC_WORLD.setter
    def VISUALIZE_FABRIC_WORLD(self, value: bool):
        self._VISUALIZE_FABRIC_WORLD = value

    @property
    def DEBUG_VIS(self) -> bool:
        if not hasattr(self, "_DEBUG_VIS"):
            self._DEBUG_VIS = self.cfg.debug_vis
        return self._DEBUG_VIS

    @DEBUG_VIS.setter
    def DEBUG_VIS(self, value: bool):
        self._DEBUG_VIS = value

    #### MODIFIABLE PROPERTIES END ####

    #### CONSTANT PROPERTIES START ####
    @property
    def include_blue_robot(self) -> bool:
        return self.cfg.debug_vis and self.num_envs < 10

    @property
    def fabric_dt(self) -> float:
        # Use same as simulator for now
        return self.cfg.sim.dt

    @property
    def fabric_decimation(self) -> int:
        # Use same as simulator for now
        return self.cfg.decimation

    #### CONSTANT PROPERTIES END ####
