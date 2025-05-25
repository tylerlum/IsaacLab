from __future__ import annotations

import math
from typing import List
from live_plotter import FastLivePlotter

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


class AverageMeter(nn.Module):
    def __init__(self, in_shape: int = 1, max_size: int = 1000) -> None:
        super().__init__()
        self.max_size = max_size

        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values: torch.Tensor) -> None:
        assert len(values.shape) == 1, f"values.shape: {values.shape}"
        size = values.size()[0]
        if size == 0:
            return

        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self) -> None:
        self.current_size = 0
        self.mean.fill_(0.0)

    def __len__(self) -> int:
        return self.current_size

    def get_mean(self) -> np.ndarray:
        return self.mean.squeeze(0).cpu().numpy()

