# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare pinch-style sustained contact across PhysX, Newton rigid, and Newton hydroelastic."""

import argparse

from isaaclab.app import AppLauncher

from physics_compare_utils import add_physics_mode_arg, build_sim_cfg, make_hydro_shapes, validate_backend

parser = argparse.ArgumentParser(description="Compare pinch-grasp contact behavior across physics modes.")
parser.add_argument("--num_steps", type=int, default=240, help="Number of simulation steps to run.")
parser.add_argument("--object", type=str, choices=("cube", "pen"), default="cube", help="Object to pinch.")
AppLauncher.add_app_launcher_args(parser)
add_physics_mode_arg(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


def _make_grasped_object_cfg() -> RigidObjectCfg:
    rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
    collision_props = sim_utils.CollisionPropertiesCfg()
    mass_props = sim_utils.MassPropertiesCfg(mass=0.3)
    if args_cli.object == "pen":
        spawn = sim_utils.CapsuleCfg(
            radius=0.03,
            height=0.28,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.75, 0.2), metallic=0.1),
        )
    else:
        spawn = sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.14),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.98, 0.9, 0.1), emissive_color=(0.08, 0.08, 0.0)),
        )
    return RigidObjectCfg(
        prim_path="/World/Object",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def design_scene() -> dict[str, RigidObject]:
    """Create two kinematic pads and a central grasped object."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.82, 0.82))
    light_cfg.func("/World/Light", light_cfg)

    table_cfg = RigidObjectCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.8, 0.16),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.28, 0.32)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.08)),
    )

    pad_spawn = sim_utils.CuboidCfg(
        size=(0.08, 0.22, 0.22),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.65, 0.9), metallic=0.1),
    )
    left_pad_cfg = RigidObjectCfg(
        prim_path="/World/LeftPad",
        spawn=pad_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.24, 0.0, 0.32)),
    )
    right_pad_cfg = RigidObjectCfg(
        prim_path="/World/RightPad",
        spawn=pad_spawn.replace(visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.45, 0.25), metallic=0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.24, 0.0, 0.32)),
    )
    return {
        "table": RigidObject(cfg=table_cfg),
        "left_pad": RigidObject(cfg=left_pad_cfg),
        "right_pad": RigidObject(cfg=right_pad_cfg),
        "object": RigidObject(cfg=_make_grasped_object_cfg()),
    }


def configure_initial_state(entities: dict[str, RigidObject]) -> None:
    """Reset pads and object to a deterministic pre-grasp setup."""
    left_pad = entities["left_pad"]
    right_pad = entities["right_pad"]
    grasped = entities["object"]

    table_pose = wp.to_torch(entities["table"].data.default_root_pose).clone()
    left_pose = wp.to_torch(left_pad.data.default_root_pose).clone()
    right_pose = wp.to_torch(right_pad.data.default_root_pose).clone()
    object_pose = wp.to_torch(grasped.data.default_root_pose).clone()
    zero_vel = torch.zeros((1, 6), device=grasped.device)

    table_pose[:, :3] = torch.tensor([[0.0, 0.0, 0.08]], device=entities["table"].device)
    left_pose[:, :3] = torch.tensor([[-0.24, 0.0, 0.32]], device=left_pad.device)
    right_pose[:, :3] = torch.tensor([[0.24, 0.0, 0.32]], device=right_pad.device)
    object_pose[:, :3] = torch.tensor([[0.0, 0.01, 0.32]], device=grasped.device)
    object_pose[:, 3:] = torch.tensor([[0.0, 0.0, 0.08, 0.9968]], device=grasped.device)

    entities["table"].write_root_pose_to_sim_index(root_pose=table_pose)
    entities["table"].write_root_velocity_to_sim_index(root_velocity=zero_vel)
    left_pad.write_root_pose_to_sim_index(root_pose=left_pose)
    right_pad.write_root_pose_to_sim_index(root_pose=right_pose)
    left_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)
    right_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)
    grasped.write_root_pose_to_sim_index(root_pose=object_pose)
    grasped.write_root_velocity_to_sim_index(root_velocity=zero_vel)

    for entity in entities.values():
        entity.reset()

    object_pos = wp.to_torch(grasped.data.root_pos_w)[0].cpu().tolist()
    print(f"[INFO]: Center object initialized at {object_pos} as '{args_cli.object}'.")


def run_simulator(sim: SimulationContext, entities: dict[str, RigidObject]) -> None:
    """Run the pinch comparison with kinematic pad motion."""
    left_pad = entities["left_pad"]
    right_pad = entities["right_pad"]
    grasped = entities["object"]
    sim_dt = sim.get_physics_dt()

    configure_initial_state(entities)
    validate_backend(args_cli.physics, min_hydro_shapes=3)

    left_pose = wp.to_torch(left_pad.data.root_pose_w).clone()
    right_pose = wp.to_torch(right_pad.data.root_pose_w).clone()
    zero_vel = torch.zeros((1, 6), device=left_pad.device)

    close_steps = 120
    hold_steps = 120
    lift_steps = max(args_cli.num_steps - close_steps - hold_steps, 0)

    for step in range(args_cli.num_steps):
        if step < close_steps:
            left_pose[:, 0] += 0.0015
            right_pose[:, 0] -= 0.0015
            left_pad.write_root_pose_to_sim_index(root_pose=left_pose)
            right_pad.write_root_pose_to_sim_index(root_pose=right_pose)
            left_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)
            right_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)
        elif step < close_steps + hold_steps:
            left_pad.write_root_pose_to_sim_index(root_pose=left_pose)
            right_pad.write_root_pose_to_sim_index(root_pose=right_pose)
            left_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)
            right_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)
        elif lift_steps > 0:
            left_pose[:, 2] += 0.0015
            right_pose[:, 2] += 0.0015
            left_pad.write_root_pose_to_sim_index(root_pose=left_pose)
            right_pad.write_root_pose_to_sim_index(root_pose=right_pose)
            left_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)
            right_pad.write_root_velocity_to_sim_index(root_velocity=zero_vel)

        for entity in entities.values():
            entity.write_data_to_sim()
        sim.step()
        for entity in entities.values():
            entity.update(sim_dt)

        if step % 30 == 0 or step == args_cli.num_steps - 1:
            object_pose = wp.to_torch(grasped.data.root_pose_w)[0]
            gap = float(wp.to_torch(right_pad.data.root_pos_w)[0, 0] - wp.to_torch(left_pad.data.root_pos_w)[0, 0])
            pad_height = float(wp.to_torch(left_pad.data.root_pos_w)[0, 2])
            print(
                f"[INFO]: step={step:03d} pad_gap={gap:.4f} pad_height={pad_height:.4f} "
                f"object_pos={object_pose[:3].cpu().tolist()} object_quat={object_pose[3:].cpu().tolist()}"
            )


def main() -> None:
    hydro_shapes = make_hydro_shapes(["/World/LeftPad", "/World/RightPad", "/World/Object"])
    sim_cfg = build_sim_cfg(
        args_cli.physics,
        device=args_cli.device,
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, 0.0),
        hydroelastic_shapes=hydro_shapes,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.45, 1.0, 0.95], target=[0.0, 0.0, 0.30])

    entities = design_scene()
    sim.reset()
    print(f"[INFO]: Pinch-grasp scene ready. object={args_cli.object}")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
