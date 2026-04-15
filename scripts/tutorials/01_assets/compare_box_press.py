# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare sustained box contact across PhysX, Newton rigid, and Newton hydroelastic."""

import argparse

from isaaclab.app import AppLauncher

from physics_compare_utils import add_physics_mode_arg, build_sim_cfg, make_hydro_shapes, validate_backend

parser = argparse.ArgumentParser(description="Compare box-press contact behavior across physics modes.")
parser.add_argument("--num_steps", type=int, default=180, help="Number of simulation steps to run.")
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


def design_scene() -> dict[str, RigidObject]:
    """Create a heavy support block and a tilted dynamic top box."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    support_cfg = RigidObjectCfg(
        prim_path="/World/Support",
        spawn=sim_utils.CuboidCfg(
            size=(1.1, 1.1, 0.18),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.09)),
    )
    top_cfg = RigidObjectCfg(
        prim_path="/World/TopBox",
        spawn=sim_utils.CuboidCfg(
            size=(0.32, 0.32, 0.16),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.65, 0.9), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.37)),
    )
    return {
        "support": RigidObject(cfg=support_cfg),
        "top_box": RigidObject(cfg=top_cfg),
    }


def configure_initial_state(entities: dict[str, RigidObject]) -> None:
    """Reset support and top box to a deterministic edge-biased contact setup."""
    support = entities["support"]
    top_box = entities["top_box"]

    support_pose = wp.to_torch(support.data.default_root_pose).clone()
    support_vel = wp.to_torch(support.data.default_root_vel).clone()
    top_pose = wp.to_torch(top_box.data.default_root_pose).clone()
    top_vel = wp.to_torch(top_box.data.default_root_vel).clone()

    support_pose[:, :3] = torch.tensor([[0.0, 0.0, 0.09]], device=support.device)
    support_vel.zero_()

    tilt_quat = torch.tensor([[0.085, 0.03, 0.0, 0.9959]], device=top_box.device)
    top_pose[:, :3] = torch.tensor([[0.05, 0.0, 0.37]], device=top_box.device)
    top_pose[:, 3:] = tilt_quat
    top_vel.zero_()

    support.write_root_pose_to_sim_index(root_pose=support_pose)
    support.write_root_velocity_to_sim_index(root_velocity=support_vel)
    top_box.write_root_pose_to_sim_index(root_pose=top_pose)
    top_box.write_root_velocity_to_sim_index(root_velocity=top_vel)
    support.reset()
    top_box.reset()


def run_simulator(sim: SimulationContext, entities: dict[str, RigidObject]) -> None:
    """Run the comparison and print top-box settling diagnostics."""
    top_box = entities["top_box"]
    sim_dt = sim.get_physics_dt()

    configure_initial_state(entities)
    validate_backend(args_cli.physics, min_hydro_shapes=2)

    for step in range(args_cli.num_steps):
        for entity in entities.values():
            entity.write_data_to_sim()
        sim.step()
        for entity in entities.values():
            entity.update(sim_dt)

        if step % 30 == 0 or step == args_cli.num_steps - 1:
            pose = wp.to_torch(top_box.data.root_pose_w)[0]
            print(f"[INFO]: step={step:03d} top_pos={pose[:3].cpu().tolist()} top_quat={pose[3:].cpu().tolist()}")


def main() -> None:
    hydro_shapes = make_hydro_shapes(["/World/Support", "/World/TopBox"])
    sim_cfg = build_sim_cfg(
        args_cli.physics,
        device=args_cli.device,
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
        hydroelastic_shapes=hydro_shapes,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, 1.25, 1.1], target=[0.0, 0.0, 0.2])

    entities = design_scene()
    sim.reset()
    print("[INFO]: Box-press scene ready.")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
