# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the Isaac Lab Newton hydroelastic config path on simple rigid boxes.

It copies the structure of ``run_rigid_object.py`` but switches the simulation backend to Newton,
enables the hydroelastic collision pipeline, and marks two imported rigid objects hydroelastic by path.

.. code-block:: bash

    ./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object_hydroelastic.py --headless

"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on running Newton hydroelastic rigid object contacts.")
parser.add_argument("--num_steps", type=int, default=120, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_newton.physics import HydroelasticCfg, HydroelasticShapeCfg, MJWarpSolverCfg, NewtonCfg, NewtonManager


def design_scene():
    """Create two boxes that collide under Newton hydroelastic contact."""
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    moving_box_cfg = RigidObjectCfg(
        prim_path="/World/MovingBox",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.9), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.45, 0.0, 0.0)),
    )
    target_box_cfg = RigidObjectCfg(
        prim_path="/World/TargetBox",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.5, 0.2), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    return {
        "moving_box": RigidObject(cfg=moving_box_cfg),
        "target_box": RigidObject(cfg=target_box_cfg),
    }


def configure_initial_state(moving_box: RigidObject, target_box: RigidObject) -> None:
    """Reset poses and send the moving box toward the target box."""
    moving_pose = wp.to_torch(moving_box.data.default_root_pose).clone()
    moving_vel = wp.to_torch(moving_box.data.default_root_vel).clone()
    target_pose = wp.to_torch(target_box.data.default_root_pose).clone()
    target_vel = wp.to_torch(target_box.data.default_root_vel).clone()

    moving_pose[:, :3] = torch.tensor([[-0.45, 0.0, 0.0]], device=moving_box.device)
    target_pose[:, :3] = torch.tensor([[0.0, 0.0, 0.0]], device=target_box.device)

    moving_vel.zero_()
    target_vel.zero_()
    moving_vel[:, 0] = 1.25

    moving_box.write_root_pose_to_sim_index(root_pose=moving_pose)
    moving_box.write_root_velocity_to_sim_index(root_velocity=moving_vel)
    target_box.write_root_pose_to_sim_index(root_pose=target_pose)
    target_box.write_root_velocity_to_sim_index(root_velocity=target_vel)
    moving_box.reset()
    target_box.reset()


def validate_hydroelastic_setup() -> None:
    """Ensure the finalized Newton model contains hydroelastic collision shapes."""
    from newton import ShapeFlags

    model = NewtonManager.get_model()
    shape_flags = model.shape_flags.numpy()
    hydro_indices = [i for i, flags in enumerate(shape_flags) if flags & int(ShapeFlags.HYDROELASTIC)]
    if len(hydro_indices) < 2:
        raise RuntimeError(
            f"Expected at least 2 hydroelastic shapes in the Newton model, found {len(hydro_indices)}."
        )
    print(f"[INFO]: Hydroelastic-enabled shapes in finalized model: {len(hydro_indices)}")


def run_simulator(sim: SimulationContext, entities: dict[str, RigidObject]):
    """Run the short hydroelastic collision demo."""
    moving_box = entities["moving_box"]
    target_box = entities["target_box"]
    sim_dt = sim.get_physics_dt()

    configure_initial_state(moving_box, target_box)
    validate_hydroelastic_setup()

    for step in range(args_cli.num_steps):
        moving_box.write_data_to_sim()
        target_box.write_data_to_sim()
        sim.step()
        moving_box.update(sim_dt)
        target_box.update(sim_dt)
        if step % 30 == 0 or step == args_cli.num_steps - 1:
            moving_pos = wp.to_torch(moving_box.data.root_pos_w)[0].cpu().tolist()
            target_pos = wp.to_torch(target_box.data.root_pos_w)[0].cpu().tolist()
            print(f"[INFO]: step={step:03d} moving={moving_pos} target={target_pos}")


def main():
    """Main function."""
    solver_cfg = MJWarpSolverCfg(
        solver="newton",
        integrator="implicitfast",
        cone="elliptic",
        use_mujoco_contacts=False,
        ls_parallel=False,
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
        hydroelastic_cfg=HydroelasticCfg(
            grid_size=48,
            buffer_fraction=0.25,
            contact_buffer_fraction=0.25,
        ),
        hydroelastic_shapes=[
            HydroelasticShapeCfg(
                prim_path="/World/MovingBox*",
                kh=5.0e8,
                sdf_max_resolution=32,
            ),
            HydroelasticShapeCfg(
                prim_path="/World/TargetBox*",
                kh=5.0e8,
                sdf_max_resolution=32,
            ),
        ],
    )

    sim_cfg = SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, 0.0),
        physics=newton_cfg,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 0.0, 0.8], target=[0.0, 0.0, 0.0])

    scene_entities = design_scene()

    sim.reset()
    print("[INFO]: Hydroelastic Newton scene setup complete...")
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
