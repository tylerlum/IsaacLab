# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Bimanual robots.

The following configurations are available:

* :obj:`BIMANUAL_CFG`: Bimanual robot consisting of Kuka arms and Allegro hands
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
import numpy as np

##
# Configuration
##

DEFAULT_KUKA_DOF_POS = np.deg2rad([0, 0, 0, -90, 0, 90, 0]).tolist()

DEFAULT_ALLEGRO_DOF_POS = [
    0.0,
    0.3,
    0.3,
    0.3,
    0.0,
    0.3,
    0.3,
    0.3,
    0.0,
    0.3,
    0.3,
    0.3,
    1.2,
    0.6,
    0.3,
    0.6,
]

BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/bimanual_kuka_allegro_v18/usd/bimanual_kuka_allegro.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # NOTE: Save on compute, fabrics handle this
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,  # NOTE: This isn't actually needed if the USD already has a fixed base
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Right arm
            "right_iiwa14_joint_1": DEFAULT_KUKA_DOF_POS[0],
            "right_iiwa14_joint_2": DEFAULT_KUKA_DOF_POS[1],
            "right_iiwa14_joint_3": DEFAULT_KUKA_DOF_POS[2],
            "right_iiwa14_joint_4": DEFAULT_KUKA_DOF_POS[3],
            "right_iiwa14_joint_5": DEFAULT_KUKA_DOF_POS[4],
            "right_iiwa14_joint_6": DEFAULT_KUKA_DOF_POS[5],
            "right_iiwa14_joint_7": DEFAULT_KUKA_DOF_POS[6],
            # Right hand
            "right_index_joint_0": DEFAULT_ALLEGRO_DOF_POS[0],
            "right_index_joint_1": DEFAULT_ALLEGRO_DOF_POS[1],
            "right_index_joint_2": DEFAULT_ALLEGRO_DOF_POS[2],
            "right_index_joint_3": DEFAULT_ALLEGRO_DOF_POS[3],
            "right_middle_joint_0": DEFAULT_ALLEGRO_DOF_POS[4],
            "right_middle_joint_1": DEFAULT_ALLEGRO_DOF_POS[5],
            "right_middle_joint_2": DEFAULT_ALLEGRO_DOF_POS[6],
            "right_middle_joint_3": DEFAULT_ALLEGRO_DOF_POS[7],
            "right_ring_joint_0": DEFAULT_ALLEGRO_DOF_POS[8],
            "right_ring_joint_1": DEFAULT_ALLEGRO_DOF_POS[9],
            "right_ring_joint_2": DEFAULT_ALLEGRO_DOF_POS[10],
            "right_ring_joint_3": DEFAULT_ALLEGRO_DOF_POS[11],
            "right_thumb_joint_0": DEFAULT_ALLEGRO_DOF_POS[12],
            "right_thumb_joint_1": DEFAULT_ALLEGRO_DOF_POS[13],
            "right_thumb_joint_2": DEFAULT_ALLEGRO_DOF_POS[14],
            "right_thumb_joint_3": DEFAULT_ALLEGRO_DOF_POS[15],
            # Left arm
            "left_iiwa14_joint_1": DEFAULT_KUKA_DOF_POS[0],
            "left_iiwa14_joint_2": DEFAULT_KUKA_DOF_POS[1],
            "left_iiwa14_joint_3": DEFAULT_KUKA_DOF_POS[2],
            "left_iiwa14_joint_4": DEFAULT_KUKA_DOF_POS[3],
            "left_iiwa14_joint_5": DEFAULT_KUKA_DOF_POS[4],
            "left_iiwa14_joint_6": DEFAULT_KUKA_DOF_POS[5],
            "left_iiwa14_joint_7": DEFAULT_KUKA_DOF_POS[6],
            # Left hand
            "left_index_joint_0": DEFAULT_ALLEGRO_DOF_POS[0],
            "left_index_joint_1": DEFAULT_ALLEGRO_DOF_POS[1],
            "left_index_joint_2": DEFAULT_ALLEGRO_DOF_POS[2],
            "left_index_joint_3": DEFAULT_ALLEGRO_DOF_POS[3],
            "left_middle_joint_0": DEFAULT_ALLEGRO_DOF_POS[4],
            "left_middle_joint_1": DEFAULT_ALLEGRO_DOF_POS[5],
            "left_middle_joint_2": DEFAULT_ALLEGRO_DOF_POS[6],
            "left_middle_joint_3": DEFAULT_ALLEGRO_DOF_POS[7],
            "left_ring_joint_0": DEFAULT_ALLEGRO_DOF_POS[8],
            "left_ring_joint_1": DEFAULT_ALLEGRO_DOF_POS[9],
            "left_ring_joint_2": DEFAULT_ALLEGRO_DOF_POS[10],
            "left_ring_joint_3": DEFAULT_ALLEGRO_DOF_POS[11],
            "left_thumb_joint_0": DEFAULT_ALLEGRO_DOF_POS[12],
            "left_thumb_joint_1": DEFAULT_ALLEGRO_DOF_POS[13],
            "left_thumb_joint_2": DEFAULT_ALLEGRO_DOF_POS[14],
            "left_thumb_joint_3": DEFAULT_ALLEGRO_DOF_POS[15],
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_iiwa14_joint_[1-7]",
            ],
            # effort_limit_sim=300,
            effort_limit_sim={
                "right_iiwa14_joint_1": 176,
                "right_iiwa14_joint_2": 176,
                "right_iiwa14_joint_3": 110,
                "right_iiwa14_joint_4": 110,
                "right_iiwa14_joint_5": 110,
                "right_iiwa14_joint_6": 40,
                "right_iiwa14_joint_7": 40,
            },
            velocity_limit_sim=10,
            stiffness={
                "right_iiwa14_joint_1": 600,
                "right_iiwa14_joint_2": 600,
                "right_iiwa14_joint_3": 500,
                "right_iiwa14_joint_4": 400,
                "right_iiwa14_joint_5": 200,
                "right_iiwa14_joint_6": 200,
                "right_iiwa14_joint_7": 200,
            },
            damping={
                "right_iiwa14_joint_1": 70,
                "right_iiwa14_joint_2": 70,
                "right_iiwa14_joint_3": 70,
                "right_iiwa14_joint_4": 70,
                "right_iiwa14_joint_5": 40,
                "right_iiwa14_joint_6": 30,
                "right_iiwa14_joint_7": 30,
            },
            armature=0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_index_joint_[0-3]",
                "right_middle_joint_[0-3]",
                "right_ring_joint_[0-3]",
                "right_thumb_joint_[0-3]",
            ],
            effort_limit_sim=0.5,
            velocity_limit_sim=7,
            stiffness=0.5,
            damping=0.1,
            armature=0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_iiwa14_joint_[1-7]",
            ],
            # effort_limit_sim=300,
            effort_limit_sim={
                "left_iiwa14_joint_1": 176,
                "left_iiwa14_joint_2": 176,
                "left_iiwa14_joint_3": 110,
                "left_iiwa14_joint_4": 110,
                "left_iiwa14_joint_5": 110,
                "left_iiwa14_joint_6": 40,
                "left_iiwa14_joint_7": 40,
            },
            velocity_limit_sim=10,
            stiffness={
                "left_iiwa14_joint_1": 600,
                "left_iiwa14_joint_2": 600,
                "left_iiwa14_joint_3": 500,
                "left_iiwa14_joint_4": 400,
                "left_iiwa14_joint_5": 200,
                "left_iiwa14_joint_6": 200,
                "left_iiwa14_joint_7": 200,
            },
            damping={
                "left_iiwa14_joint_1": 70,
                "left_iiwa14_joint_2": 70,
                "left_iiwa14_joint_3": 70,
                "left_iiwa14_joint_4": 70,
                "left_iiwa14_joint_5": 40,
                "left_iiwa14_joint_6": 30,
                "left_iiwa14_joint_7": 30,
            },
            armature=0,
        ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_index_joint_[0-3]",
                "left_middle_joint_[0-3]",
                "left_ring_joint_[0-3]",
                "left_thumb_joint_[0-3]",
            ],
            effort_limit_sim=0.5,
            velocity_limit_sim=7,
            stiffness=0.5,
            damping=0.1,
            armature=0,
        ),
    },
)

BLUE_BIMANUAL_CFG = BIMANUAL_CFG.copy()
BLUE_BIMANUAL_CFG.spawn.usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/bimanual_kuka_allegro_v9/usd_blue/bimanual_kuka_allegro.usd"
BLUE_BIMANUAL_CFG.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
    # collision_enabled=False,
)  # No collision

"""Configuration for the Bimanual robot consisting of Kuka arms and Allegro hands."""
