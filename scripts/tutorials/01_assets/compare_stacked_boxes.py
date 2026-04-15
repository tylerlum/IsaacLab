# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare stack-settling behavior across PhysX, Newton rigid, and Newton hydroelastic."""

import argparse

from isaaclab.app import AppLauncher

from physics_compare_utils import add_physics_mode_arg, build_sim_cfg, make_hydro_shapes, validate_backend

parser = argparse.ArgumentParser(description="Compare stacked-box settling across physics modes.")
parser.add_argument("--num_steps", type=int, default=240, help="Number of simulation steps to run.")
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
    """Create a small stack of boxes on a static base."""
    light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.84, 0.84, 0.84))
    light_cfg.func("/World/Light", light_cfg)

    entities: dict[str, RigidObject] = {}
    entities["base"] = RigidObject(
        cfg=RigidObjectCfg(
            prim_path="/World/Base",
            spawn=sim_utils.CuboidCfg(
                size=(0.6, 0.6, 0.12),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.4)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06)),
        )
    )

    for idx in range(4):
        entities[f"box_{idx}"] = RigidObject(
            cfg=RigidObjectCfg(
                prim_path=f"/World/Box{idx}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.14, 0.14, 0.14),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2 + 0.15 * idx, 0.6 - 0.08 * idx, 0.85 - 0.1 * idx)
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2 + 0.16 * idx)),
            )
        )
    return entities


def configure_initial_state(entities: dict[str, RigidObject]) -> None:
    """Reset the stack with slight offsets to provoke visible settling differences."""
    zero_vel = torch.zeros((1, 6), device=next(iter(entities.values())).device)
    base_pose = wp.to_torch(entities["base"].data.default_root_pose).clone()
    base_pose[:, :3] = torch.tensor([[0.0, 0.0, 0.06]], device=entities["base"].device)
    entities["base"].write_root_pose_to_sim_index(root_pose=base_pose)
    entities["base"].write_root_velocity_to_sim_index(root_velocity=zero_vel)

    offsets = [(0.0, 0.0), (0.018, -0.012), (-0.014, 0.01), (0.012, 0.016)]
    for idx in range(4):
        box = entities[f"box_{idx}"]
        pose = wp.to_torch(box.data.default_root_pose).clone()
        pose[:, :3] = torch.tensor([[offsets[idx][0], offsets[idx][1], 0.19 + 0.145 * idx]], device=box.device)
        box.write_root_pose_to_sim_index(root_pose=pose)
        box.write_root_velocity_to_sim_index(root_velocity=zero_vel)

    for entity in entities.values():
        entity.reset()


def run_simulator(sim: SimulationContext, entities: dict[str, RigidObject]) -> None:
    """Run the settling comparison and print top-box drift."""
    sim_dt = sim.get_physics_dt()
    top_box = entities["box_3"]

    configure_initial_state(entities)
    validate_backend(args_cli.physics, min_hydro_shapes=5)

    for step in range(args_cli.num_steps):
        for entity in entities.values():
            entity.write_data_to_sim()
        sim.step()
        for entity in entities.values():
            entity.update(sim_dt)

        if step % 40 == 0 or step == args_cli.num_steps - 1:
            top_pose = wp.to_torch(top_box.data.root_pose_w)[0]
            lateral_drift = float(torch.linalg.vector_norm(top_pose[:2]).cpu())
            print(
                f"[INFO]: step={step:03d} top_pos={top_pose[:3].cpu().tolist()} "
                f"top_quat={top_pose[3:].cpu().tolist()} lateral_drift={lateral_drift:.4f}"
            )


def main() -> None:
    hydro_shapes = make_hydro_shapes(
        ["/World/Base", "/World/Box0", "/World/Box1", "/World/Box2", "/World/Box3"]
    )
    sim_cfg = build_sim_cfg(
        args_cli.physics,
        device=args_cli.device,
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
        hydroelastic_shapes=hydro_shapes,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.6, 1.1, 1.3], target=[0.0, 0.0, 0.35])

    entities = design_scene()
    sim.reset()
    print("[INFO]: Stacked-boxes scene ready.")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
