from __future__ import annotations

import isaaclab.sim as sim_utils
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
        ground_plane_cfg.func(prim_path, ground_plane_cfg, translation=(0.0, 0.0, -0.5))
