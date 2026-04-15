# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper utilities for Newton hydroelastic configuration."""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

from newton import ModelBuilder, ShapeFlags
from newton.geometry import HydroelasticSDF

if TYPE_CHECKING:
    from .newton_manager_cfg import HydroelasticCfg, HydroelasticShapeCfg


def normalize_glob_expr(expr: str) -> str:
    """Convert Isaac Lab regex-style wildcards into fnmatch globs."""
    return expr.replace(".*", "*")


def iter_glob_exprs(expr: str | list[str]) -> list[str]:
    """Normalize one-or-many path expressions into fnmatch globs."""
    if isinstance(expr, str):
        return [normalize_glob_expr(expr)]
    return [normalize_glob_expr(pattern) for pattern in expr]


def build_hydroelastic_sdf_config(cfg: HydroelasticCfg) -> HydroelasticSDF.Config:
    """Create a Newton hydroelastic config from Isaac Lab config."""
    return HydroelasticSDF.Config(
        reduce_contacts=cfg.reduce_contacts,
        pre_prune_contacts=cfg.pre_prune_contacts,
        buffer_fraction=cfg.buffer_fraction,
        buffer_mult_broad=cfg.buffer_mult_broad,
        buffer_mult_iso=cfg.buffer_mult_iso,
        buffer_mult_contact=cfg.buffer_mult_contact,
        contact_buffer_fraction=cfg.contact_buffer_fraction,
        grid_size=cfg.grid_size,
        output_contact_surface=cfg.output_contact_surface,
        normal_matching=cfg.normal_matching,
        anchor_contact=cfg.anchor_contact,
        margin_contact_area=cfg.margin_contact_area,
        pre_prune_accumulate_all_penetrating_aggregates=cfg.pre_prune_accumulate_all_penetrating_aggregates,
    )


def validate_hydroelastic_compatibility(
    solver_type: str,
    use_mujoco_contacts: bool,
    hydroelastic_cfg: HydroelasticCfg | None,
) -> None:
    """Validate that the requested hydroelastic mode is compatible with the solver pipeline."""
    if hydroelastic_cfg is None:
        return
    if solver_type == "mujoco_warp" and use_mujoco_contacts:
        raise ValueError(
            "Newton hydroelastic contact requires Newton's external collision pipeline. "
            "Set MJWarpSolverCfg.use_mujoco_contacts=False when hydroelastic_cfg is enabled."
        )


def validate_hydroelastic_shape_cfg(shape_cfg: HydroelasticShapeCfg) -> None:
    """Validate one pattern-based hydroelastic shape override."""
    if not shape_cfg.prim_path:
        raise ValueError("HydroelasticShapeCfg.prim_path must not be empty.")
    if shape_cfg.sdf_max_resolution is not None and shape_cfg.sdf_target_voxel_size is not None:
        raise ValueError(
            "HydroelasticShapeCfg accepts either sdf_max_resolution or sdf_target_voxel_size, not both."
        )
    if shape_cfg.sdf_max_resolution is not None and shape_cfg.sdf_max_resolution % 8 != 0:
        raise ValueError(
            "HydroelasticShapeCfg.sdf_max_resolution must be divisible by 8 to match Newton's builder requirements."
        )


def apply_hydroelastic_shape_configs(builder: ModelBuilder, shape_cfgs: list[HydroelasticShapeCfg]) -> int:
    """Apply pattern-based hydroelastic overrides to an imported Newton builder."""
    if not shape_cfgs:
        return 0

    shape_labels = list(getattr(builder, "shape_label", []) or [])
    if not shape_labels:
        raise ValueError(
            "Newton builder has no shape_label entries, so hydroelastic shape matching cannot be applied."
        )

    hydroelastic_count = 0
    for shape_cfg in shape_cfgs:
        validate_hydroelastic_shape_cfg(shape_cfg)
        patterns = iter_glob_exprs(shape_cfg.prim_path)
        matched = [
            idx for idx, label in enumerate(shape_labels) if any(fnmatch.fnmatch(label, pattern) for pattern in patterns)
        ]
        if not matched and shape_cfg.require_match:
            raise ValueError(
                "HydroelasticShapeCfg did not match any imported shapes for "
                f"prim_path={shape_cfg.prim_path!r}. Available sample labels: {shape_labels[:10]!r}"
            )

        for idx in matched:
            if shape_cfg.is_hydroelastic:
                builder.shape_flags[idx] |= int(ShapeFlags.HYDROELASTIC)
                hydroelastic_count += 1
            else:
                builder.shape_flags[idx] &= ~int(ShapeFlags.HYDROELASTIC)
            builder.shape_material_kh[idx] = float(shape_cfg.kh)
            builder.shape_sdf_narrow_band_range[idx] = shape_cfg.sdf_narrow_band_range
            builder.shape_sdf_target_voxel_size[idx] = shape_cfg.sdf_target_voxel_size
            builder.shape_sdf_max_resolution[idx] = shape_cfg.sdf_max_resolution

    return hydroelastic_count
