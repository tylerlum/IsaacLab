# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton hydroelastic Isaac Lab config plumbing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from newton import ModelBuilder, ShapeFlags

from isaaclab_newton.physics import HydroelasticCfg, HydroelasticShapeCfg
from isaaclab_newton.physics.hydroelastic_utils import (
    apply_hydroelastic_shape_configs,
    build_hydroelastic_sdf_config,
    validate_hydroelastic_compatibility,
)


def _make_builder() -> ModelBuilder:
    builder = ModelBuilder(up_axis="Z")
    body = builder.add_body(label="/World/Env_0/Robot")
    builder.add_shape_sphere(
        body,
        radius=0.05,
        label="/World/Env_0/Robot/finger_left/collisions/shape",
    )
    builder.add_shape_box(
        body,
        hx=0.03,
        hy=0.04,
        hz=0.05,
        label="/World/Env_0/Object/collisions/shape",
    )
    return builder


def test_build_hydroelastic_sdf_config():
    cfg = HydroelasticCfg(
        reduce_contacts=False,
        buffer_fraction=0.5,
        grid_size=64,
        output_contact_surface=True,
        normal_matching=False,
    )

    hydro_cfg = build_hydroelastic_sdf_config(cfg)

    assert hydro_cfg.reduce_contacts is False
    assert hydro_cfg.buffer_fraction == pytest.approx(0.5)
    assert hydro_cfg.grid_size == 64
    assert hydro_cfg.output_contact_surface is True
    assert hydro_cfg.normal_matching is False


def test_apply_hydroelastic_shape_configs_matches_patterns():
    builder = _make_builder()

    modified_count = apply_hydroelastic_shape_configs(
        builder,
        [
            HydroelasticShapeCfg(
                prim_path="/World/Env_.*/Robot/finger_left/*",
                kh=2.5e8,
                sdf_max_resolution=32,
            ),
            HydroelasticShapeCfg(
                prim_path="/World/Env_0/Object/*",
                kh=4.0e8,
                sdf_target_voxel_size=0.01,
            ),
        ],
    )

    assert modified_count == 2
    assert builder.shape_flags[0] & int(ShapeFlags.HYDROELASTIC)
    assert builder.shape_flags[1] & int(ShapeFlags.HYDROELASTIC)
    assert builder.shape_material_kh[0] == pytest.approx(2.5e8)
    assert builder.shape_material_kh[1] == pytest.approx(4.0e8)
    assert builder.shape_sdf_max_resolution[0] == 32
    assert builder.shape_sdf_target_voxel_size[0] is None
    assert builder.shape_sdf_target_voxel_size[1] == pytest.approx(0.01)


def test_apply_hydroelastic_shape_configs_requires_match():
    builder = _make_builder()

    with pytest.raises(ValueError, match="did not match any imported shapes"):
        apply_hydroelastic_shape_configs(
            builder,
            [
                HydroelasticShapeCfg(
                    prim_path="/World/Env_0/DoesNotExist/*",
                    sdf_max_resolution=32,
                )
            ],
        )


def test_validate_hydroelastic_requires_external_collision_pipeline():
    with pytest.raises(ValueError, match="use_mujoco_contacts=False"):
        validate_hydroelastic_compatibility(
            solver_type="mujoco_warp",
            use_mujoco_contacts=True,
            hydroelastic_cfg=HydroelasticCfg(),
        )


def test_validate_hydroelastic_allows_external_collision_pipeline():
    validate_hydroelastic_compatibility(
        solver_type="mujoco_warp",
        use_mujoco_contacts=False,
        hydroelastic_cfg=HydroelasticCfg(),
    )
