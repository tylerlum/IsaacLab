# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd"
    ),
    actuators={
        "wheel_acts": ImplicitActuatorCfg(
            joint_names_expr=[".*"], damping=None, stiffness=None
        )
    },
)

BIMANUAL_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/bimanual_kuka_allegro_v4/usd/bimanual_kuka_allegro.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            # Right arm
            "right_iiwa14_joint_1": 0.0,
            "right_iiwa14_joint_2": 0.0,
            "right_iiwa14_joint_3": 0.0,
            "right_iiwa14_joint_4": 0.0,
            "right_iiwa14_joint_5": 0.0,
            "right_iiwa14_joint_6": 0.0,
            "right_iiwa14_joint_7": 0.0,
            # Right hand
            "right_index_joint_0": 0.0,
            "right_index_joint_1": 0.0,
            "right_index_joint_2": 0.0,
            "right_index_joint_3": 0.0,
            "right_middle_joint_0": 0.0,
            "right_middle_joint_1": 0.0,
            "right_middle_joint_2": 0.0,
            "right_middle_joint_3": 0.0,
            "right_ring_joint_0": 0.0,
            "right_ring_joint_1": 0.0,
            "right_ring_joint_2": 0.0,
            "right_ring_joint_3": 0.0,
            "right_thumb_joint_0": 0.5,
            "right_thumb_joint_1": 0.0,
            "right_thumb_joint_2": 0.0,
            "right_thumb_joint_3": 0.0,
            # Left arm
            "left_iiwa14_joint_1": 0.0,
            "left_iiwa14_joint_2": 0.0,
            "left_iiwa14_joint_3": 0.0,
            "left_iiwa14_joint_4": 0.0,
            "left_iiwa14_joint_5": 0.0,
            "left_iiwa14_joint_6": 0.0,
            "left_iiwa14_joint_7": 0.0,
            # Left hand
            "left_index_joint_0": 0.0,
            "left_index_joint_1": 0.0,
            "left_index_joint_2": 0.0,
            "left_index_joint_3": 0.0,
            "left_middle_joint_0": 0.0,
            "left_middle_joint_1": 0.0,
            "left_middle_joint_2": 0.0,
            "left_middle_joint_3": 0.0,
            "left_ring_joint_0": 0.0,
            "left_ring_joint_1": 0.0,
            "left_ring_joint_2": 0.0,
            "left_ring_joint_3": 0.0,
            "left_thumb_joint_0": 0.5,
            "left_thumb_joint_1": 0.0,
            "left_thumb_joint_2": 0.0,
            "left_thumb_joint_3": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_iiwa14_joint_[1-7]",
            ],
            # effort_limit=300,
            effort_limit={
                "right_iiwa14_joint_1": 176,
                "right_iiwa14_joint_2": 176,
                "right_iiwa14_joint_3": 110,
                "right_iiwa14_joint_4": 110,
                "right_iiwa14_joint_5": 110,
                "right_iiwa14_joint_6": 40,
                "right_iiwa14_joint_7": 40,
            },
            velocity_limit=10,
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
            effort_limit=0.5,
            velocity_limit=7,
            stiffness=0.5,
            damping=0.1,
            armature=0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_iiwa14_joint_[1-7]",
            ],
            # effort_limit=300,
            effort_limit={
                "left_iiwa14_joint_1": 176,
                "left_iiwa14_joint_2": 176,
                "left_iiwa14_joint_3": 110,
                "left_iiwa14_joint_4": 110,
                "left_iiwa14_joint_5": 110,
                "left_iiwa14_joint_6": 40,
                "left_iiwa14_joint_7": 40,
            },
            velocity_limit=10,
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
            effort_limit=0.5,
            velocity_limit=7,
            stiffness=0.5,
            damping=0.1,
            armature=0,
        ),
    },
)

DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    Bimanual = BIMANUAL_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Bimanual")
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            root_jetbot_state[:, :3] += scene.env_origins
            root_bimanual_state = scene["Bimanual"].data.default_root_state.clone()
            root_bimanual_state[:, :3] += scene.env_origins
            root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            root_dofbot_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
            scene["Bimanual"].write_root_pose_to_sim(root_bimanual_state[:, :7])
            scene["Bimanual"].write_root_velocity_to_sim(root_bimanual_state[:, 7:])
            scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone(),
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            joint_pos, joint_vel = (
                scene["Bimanual"].data.default_joint_pos.clone(),
                scene["Bimanual"].data.default_joint_vel.clone(),
            )
            scene["Bimanual"].write_joint_state_to_sim(joint_pos, joint_vel)
            joint_pos, joint_vel = (
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone(),
            )
            scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Jetbot, Bimanual, and Dofbot state...")

        # drive around
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])

        scene["Jetbot"].set_joint_velocity_target(action)

        straight_action = torch.Tensor([[0.25 * np.sin(2 * np.pi * 0.5 * sim_time)] * 46])
        scene["Bimanual"].set_joint_position_target(straight_action)

        # wave
        wave_action = scene["Dofbot"].data.default_joint_pos
        wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["Dofbot"].set_joint_position_target(wave_action)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
