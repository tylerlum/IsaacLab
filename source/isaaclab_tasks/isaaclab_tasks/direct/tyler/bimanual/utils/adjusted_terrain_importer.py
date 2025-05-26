
from __future__ import annotations

import math
from typing import List

import yaml
from pathlib import Path
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import torch.nn as nn
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    SPHERE_MARKER_CFG,
    FRAME_MARKER_CFG,
    CYLINDER_MARKER_CFG,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, LightCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.bimanual import BIMANUAL_CFG, BLUE_BIMANUAL_CFG
from isaaclab_tasks.direct.tyler.bimanual.utils.average_meter import AverageMeter
from isaaclab_tasks.direct.tyler.bimanual.utils.torch_utils import (
    sample_uniform_tensor,
    rescale,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.joint_order_constants import (
    isaaclab_to_fabric_joint_order_torch,
    fabric_to_isaaclab_joint_order_torch,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.constants import NUM_XYZ, NUM_QUAT
from isaaclab_tasks.direct.tyler.bimanual.utils.color_constants import (
    RED_RGB,
    GREEN_RGB,
    BLUE_RGB,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.robot_constants import (
    INDEX_FINGERTIP_IDX,
    MIDDLE_FINGERTIP_IDX,
    RING_FINGERTIP_IDX,
    THUMB_FINGERTIP_IDX,
    RIGHT_FINGERTIP_LINK_NAMES,
    LEFT_FINGERTIP_LINK_NAMES,
    RIGHT_PALM_LINK_NAME,
    LEFT_PALM_LINK_NAME,
)
from isaaclab_tasks.direct.tyler.bimanual.utils.table_constants import (
    TABLE_X,
    TABLE_Y,
    TABLE_Z,
    TABLE_QX,
    TABLE_QY,
    TABLE_QZ,
    TABLE_QW,
    TABLE_LENGTH_Z,
)
import wandb


from isaaclab.terrains.terrain_importer import TerrainImporter



class AdjustedTerrainImporter(TerrainImporter):
    def import_ground_plane(
        self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)
    ):
        """
        NOTE: This is identical to the import_ground_plane method in TerrainImporter, but it is
        modified to change the position of the ground plane.

        Add a plane to the terrain importer.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            size: The size of the plane. Defaults to (2.0e6, 2.0e6).

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # obtain ground plane color from the configured visual material
        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            # defaults to the `GroundPlaneCfg` color if diffuse color attribute is not found
            if "diffuse_color" in material:
                color = material["diffuse_color"]
            else:
                pass
                # omni.log.warn(
                #     "Visual material specified for ground plane but no diffuse color found."
                #     " Using default color: (0.0, 0.0, 0.0)"
                # )

        # get the mesh
        ground_plane_cfg = sim_utils.GroundPlaneCfg(
            physics_material=self.cfg.physics_material, size=size, color=color
        )
        ground_plane_cfg.func(prim_path, ground_plane_cfg, translation=(0.0, 0.0, -1.0))

