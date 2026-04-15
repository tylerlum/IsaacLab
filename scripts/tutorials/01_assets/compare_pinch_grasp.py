# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare pinch-style sustained contact across PhysX, Newton rigid, and Newton hydroelastic."""

import argparse

from isaaclab.app import AppLauncher

from physics_compare_utils import add_physics_mode_arg, build_sim_cfg, make_hydro_shapes, validate_backend

parser = argparse.ArgumentParser(description="Compare pinch-grasp contact behavior across physics modes.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=480,
    help="Number of simulation steps per cycle. Ignored when --loop is active unless used with --max_cycles.",
)
parser.add_argument("--object", type=str, choices=("cube", "pen"), default="cube", help="Object to pinch.")
parser.add_argument("--loop", action="store_true", help="Repeat the pinch-close-hold-lift cycle until the viewer closes.")
parser.add_argument("--max_cycles", type=int, default=0, help="Optional cap on repeated cycles when --loop is active.")
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
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        linear_damping=0.2,
        angular_damping=0.2,
        max_depenetration_velocity=2.0,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    )
    collision_props = sim_utils.CollisionPropertiesCfg()
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        static_friction=1.4,
        dynamic_friction=1.2,
        restitution=0.0,
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=0.3)
    if args_cli.object == "pen":
        spawn = sim_utils.CapsuleCfg(
            radius=0.03,
            height=0.28,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            physics_material=material_cfg,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.75, 0.2), metallic=0.1),
        )
    else:
        spawn = sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.14),
            rigid_props=rigid_props,
            collision_props=collision_props,
            physics_material=material_cfg,
            mass_props=mass_props,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.98, 0.9, 0.1), emissive_color=(0.08, 0.08, 0.0)),
        )
    return RigidObjectCfg(
        prim_path="/World/Object",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def design_scene() -> dict[str, RigidObject]:
    """Create two force-driven pads and a central grasped object."""

    light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.82, 0.82))
    light_cfg.func("/World/Light", light_cfg)

    material_cfg = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        static_friction=1.5,
        dynamic_friction=1.3,
        restitution=0.0,
    )

    table_cfg = RigidObjectCfg(
        prim_path="/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.8, 0.16),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=material_cfg,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.28, 0.32)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.08)),
    )

    pad_spawn = sim_utils.CuboidCfg(
        size=(0.08, 0.22, 0.22),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=4.0,
            angular_damping=10.0,
            max_depenetration_velocity=2.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=material_cfg,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.65, 0.9), metallic=0.1),
    )
    left_pad_cfg = RigidObjectCfg(
        prim_path="/World/LeftPad",
        spawn=pad_spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.28, 0.0, 0.30)),
    )
    right_pad_cfg = RigidObjectCfg(
        prim_path="/World/RightPad",
        spawn=pad_spawn.replace(visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.45, 0.25), metallic=0.1)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.28, 0.0, 0.30)),
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
    left_pose[:, :3] = torch.tensor([[-0.28, 0.0, 0.30]], device=left_pad.device)
    right_pose[:, :3] = torch.tensor([[0.28, 0.0, 0.30]], device=right_pad.device)
    object_pose[:, :3] = torch.tensor([[0.0, 0.0, 0.235]], device=grasped.device)
    object_pose[:, 3:] = torch.tensor([[0.0, 0.0, 0.08, 0.9968]], device=grasped.device)

    entities["table"].write_root_pose_to_sim_index(root_pose=table_pose)
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

    validate_backend(args_cli.physics, min_hydro_shapes=3)
    settle_steps = 120
    close_steps = 180
    hold_steps = 120
    lift_steps = max(args_cli.num_steps - settle_steps - close_steps - hold_steps, 0)
    cycle_count = 0
    running = True

    kp_xy = 1800.0
    kd_xy = 160.0
    kp_z = 2200.0
    kd_z = 180.0
    max_force = 240.0

    while running and simulation_app.is_running():
        configure_initial_state(entities)
        print(f"[INFO]: Starting cycle {cycle_count}")

        for step in range(args_cli.num_steps):
            if step < settle_steps:
                left_target = torch.tensor([[-0.28, 0.0, 0.30]], device=left_pad.device)
                right_target = torch.tensor([[0.28, 0.0, 0.30]], device=right_pad.device)
            elif step < settle_steps + close_steps:
                alpha = (step - settle_steps + 1) / close_steps
                left_target = torch.tensor([[-0.28 + 0.17 * alpha, 0.0, 0.30]], device=left_pad.device)
                right_target = torch.tensor([[0.28 - 0.17 * alpha, 0.0, 0.30]], device=right_pad.device)
            elif step < settle_steps + close_steps + hold_steps:
                left_target = torch.tensor([[-0.095, 0.0, 0.30]], device=left_pad.device)
                right_target = torch.tensor([[0.095, 0.0, 0.30]], device=right_pad.device)
            else:
                alpha = (step - settle_steps - close_steps - hold_steps + 1) / max(lift_steps, 1)
                lift_height = 0.12 * min(alpha, 1.0)
                left_target = torch.tensor([[-0.095, 0.0, 0.30 + lift_height]], device=left_pad.device)
                right_target = torch.tensor([[0.095, 0.0, 0.30 + lift_height]], device=right_pad.device)

            left_pos = wp.to_torch(left_pad.data.root_pos_w)
            right_pos = wp.to_torch(right_pad.data.root_pos_w)
            left_vel = wp.to_torch(left_pad.data.root_lin_vel_w)
            right_vel = wp.to_torch(right_pad.data.root_lin_vel_w)

            left_force = torch.zeros((1, 1, 3), device=left_pad.device)
            right_force = torch.zeros((1, 1, 3), device=right_pad.device)
            left_force[..., 0] = kp_xy * (left_target[:, 0:1] - left_pos[:, 0:1]) - kd_xy * left_vel[:, 0:1]
            right_force[..., 0] = kp_xy * (right_target[:, 0:1] - right_pos[:, 0:1]) - kd_xy * right_vel[:, 0:1]
            left_force[..., 2] = kp_z * (left_target[:, 2:3] - left_pos[:, 2:3]) - kd_z * left_vel[:, 2:3]
            right_force[..., 2] = kp_z * (right_target[:, 2:3] - right_pos[:, 2:3]) - kd_z * right_vel[:, 2:3]
            left_force = torch.clamp(left_force, min=-max_force, max=max_force)
            right_force = torch.clamp(right_force, min=-max_force, max=max_force)

            zero_torque = torch.zeros_like(left_force)
            left_pad.instantaneous_wrench_composer.set_forces_and_torques(left_force, zero_torque)
            right_pad.instantaneous_wrench_composer.set_forces_and_torques(right_force, zero_torque)

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
                    f"[INFO]: cycle={cycle_count} step={step:03d} pad_gap={gap:.4f} pad_height={pad_height:.4f} "
                    f"object_pos={object_pose[:3].cpu().tolist()} object_quat={object_pose[3:].cpu().tolist()}"
                )

            if not simulation_app.is_running():
                running = False
                break

        cycle_count += 1
        if not args_cli.loop:
            running = False
        elif args_cli.max_cycles > 0 and cycle_count >= args_cli.max_cycles:
            running = False


def main() -> None:
    hydro_shapes = make_hydro_shapes(["/World/LeftPad", "/World/RightPad", "/World/Object"])
    sim_cfg = build_sim_cfg(
        args_cli.physics,
        device=args_cli.device,
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
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
