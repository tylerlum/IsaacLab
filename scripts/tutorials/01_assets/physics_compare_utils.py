# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for physics comparison tutorial scripts."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import HydroelasticShapeCfg

PhysicsMode = Literal["physx", "newton", "newton_hydro"]


def add_physics_mode_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common physics selection CLI flag."""
    parser.add_argument(
        "--physics",
        type=str,
        choices=("physx", "newton", "newton_hydro"),
        default="newton_hydro",
        help="Physics backend mode to compare.",
    )


def build_sim_cfg(
    mode: PhysicsMode,
    *,
    device: str,
    dt: float,
    gravity: tuple[float, float, float],
    hydroelastic_shapes: list["HydroelasticShapeCfg"] | None = None,
) -> "SimulationCfg":
    """Build a simulation config for the requested comparison mode."""
    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import HydroelasticCfg, MJWarpSolverCfg, NewtonCfg
    from isaaclab_physx.physics import PhysxCfg

    if mode == "physx":
        physics = PhysxCfg(enable_ccd=False, enable_stabilization=False)
    else:
        solver_cfg = MJWarpSolverCfg(
            njmax=512,
            nconmax=512,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            use_mujoco_contacts=False,
            ls_parallel=False,
        )
        physics = NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )
        if mode == "newton_hydro":
            physics.hydroelastic_cfg = HydroelasticCfg(
                grid_size=48,
                buffer_fraction=0.5,
                buffer_mult_iso=16,
                buffer_mult_contact=16,
                contact_buffer_fraction=0.5,
            )
            physics.hydroelastic_shapes = list(hydroelastic_shapes or [])

    return SimulationCfg(
        device=device,
        dt=dt,
        gravity=gravity,
        physics=physics,
    )


def make_hydro_shapes(
    prim_paths: list[str],
    *,
    kh: float = 5.0e8,
    sdf_max_resolution: int = 32,
    sdf_narrow_band_range: tuple[float, float] = (-0.05, 0.05),
) -> list["HydroelasticShapeCfg"]:
    """Create one hydroelastic shape override per prim path prefix."""
    from isaaclab_newton.physics import HydroelasticShapeCfg

    return [
        HydroelasticShapeCfg(
            prim_path=f"{prim_path}*",
            kh=kh,
            sdf_max_resolution=sdf_max_resolution,
            sdf_narrow_band_range=sdf_narrow_band_range,
        )
        for prim_path in prim_paths
    ]


def validate_backend(mode: PhysicsMode, *, min_hydro_shapes: int = 0) -> None:
    """Print backend diagnostics and validate hydroelastic shape activation when requested."""
    print(f"[INFO]: Physics mode = {mode}")
    if mode == "physx":
        print("[INFO]: PhysX mode active. Newton hydroelastic validation skipped.")
        return

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    if model is None:
        raise RuntimeError("Newton model was not initialized.")

    shape_labels = getattr(model, "shape_label", None) or getattr(model, "shape_key", None)
    shape_count = 0 if shape_labels is None else len(shape_labels)
    print(f"[INFO]: Newton mode active. Finalized model shape_count={shape_count}")

    if mode == "newton_hydro":
        from newton import ShapeFlags

        shape_flags = model.shape_flags.numpy()
        hydro_count = sum(1 for flags in shape_flags if flags & int(ShapeFlags.HYDROELASTIC))
        print(f"[INFO]: Hydroelastic-enabled shapes in finalized model: {hydro_count}")
        if hydro_count < min_hydro_shapes:
            raise RuntimeError(
                f"Expected at least {min_hydro_shapes} hydroelastic shapes in Newton model, found {hydro_count}."
            )
    else:
        print("[INFO]: Newton rigid mode active. Hydroelastic shape validation skipped.")
