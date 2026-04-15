# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SimToolReal: Kuka IIWA-14 + SHARPA hand dexterous manipulation environment.

Ported from IsaacGym (depthbasedRL/isaacgymenvs/tasks/simtoolreal/env.py).
Physics alignment settings from depthbasedRL/isaacsim_conversion/isaacsim_env.py.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse, sample_uniform

if TYPE_CHECKING:
    from .sim_tool_real_env_cfg import SimToolRealEnvCfg

# ---------------------------------------------------------------------------
# Joint ordering: Isaac Gym uses depth-first alphabetical traversal which
# differs from Isaac Lab's breadth-first order. We store the expected order
# and compute a permutation at init time to reindex tensors.
# ---------------------------------------------------------------------------
JOINT_NAMES_ISAACGYM = [
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
]

# Per-joint PD gains — exact values used during Isaac Gym training.
# ImplicitActuatorCfg writes these directly to PhysX without any unit conversion.
JOINT_STIFFNESSES = {
    "iiwa14_joint_1": 600, "iiwa14_joint_2": 600, "iiwa14_joint_3": 500,
    "iiwa14_joint_4": 400, "iiwa14_joint_5": 200, "iiwa14_joint_6": 200,
    "iiwa14_joint_7": 200,
    "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2,
    "left_thumb_MCP_FE": 4.76, "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
    "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
    "left_index_PIP": 0.9, "left_index_DIP": 0.9,
    "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
    "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
    "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
    "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
    "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76,
    "left_pinky_MCP_AA": 6.62, "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
}

JOINT_DAMPINGS = {
    "iiwa14_joint_1": 27.027, "iiwa14_joint_2": 27.027, "iiwa14_joint_3": 24.672,
    "iiwa14_joint_4": 22.067, "iiwa14_joint_5": 9.753, "iiwa14_joint_6": 9.148,
    "iiwa14_joint_7": 9.148,
    "left_1_thumb_CMC_FE": 0.28677, "left_thumb_CMC_AA": 0.40845,
    "left_thumb_MCP_FE": 0.20394, "left_thumb_MCP_AA": 0.24044, "left_thumb_IP": 0.04191,
    "left_2_index_MCP_FE": 0.20859, "left_index_MCP_AA": 0.24596,
    "left_index_PIP": 0.04243, "left_index_DIP": 0.03504,
    "left_3_middle_MCP_FE": 0.20859, "left_middle_MCP_AA": 0.24596,
    "left_middle_PIP": 0.04243, "left_middle_DIP": 0.03504,
    "left_4_ring_MCP_FE": 0.20859, "left_ring_MCP_AA": 0.24596,
    "left_ring_PIP": 0.04243, "left_ring_DIP": 0.03504,
    "left_5_pinky_CMC": 0.02782, "left_pinky_MCP_FE": 0.20859,
    "left_pinky_MCP_AA": 0.24596, "left_pinky_PIP": 0.04243, "left_pinky_DIP": 0.03504,
}


class SimToolRealEnv(DirectRLEnv):
    """Dexterous manipulation environment: grasp, lift, and reorient tools.

    Observation space (policy): joint_pos + joint_vel + prev_actions + palm_pos +
        palm_rot + object_rot + fingertip_pos_rel_palm + keypoints_rel_palm +
        keypoints_rel_goal + object_scales.

    Action space: 11-dim (7 arm + 4 hand), scaled to joint position targets.
    """

    cfg: SimToolRealEnvCfg

    def __init__(self, cfg: SimToolRealEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Compute joint ordering permutation (Isaac Gym vs Isaac Lab ordering)
        sim_names = list(self._robot.joint_names)
        if sim_names != JOINT_NAMES_ISAACGYM:
            self._joint_permutation = torch.tensor(
                [sim_names.index(name) for name in JOINT_NAMES_ISAACGYM],
                device=self.device, dtype=torch.long,
            )
        else:
            self._joint_permutation = None

        # Find the 11 policy-controlled joint indices in Isaac Lab ordering
        self._arm_joint_ids, _ = self._robot.find_joints(
            ["iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
             "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7"]
        )
        # Policy controls thumb_CMC_FE, index_MCP_FE, middle_MCP_FE, ring_MCP_FE
        self._hand_joint_ids, _ = self._robot.find_joints(
            ["left_1_thumb_CMC_FE", "left_2_index_MCP_FE",
             "left_3_middle_MCP_FE", "left_4_ring_MCP_FE"]
        )
        self._policy_joint_ids = self._arm_joint_ids + self._hand_joint_ids  # 11 total

        # Previous joint position targets (for EMA smoothing and obs)
        num_total_joints = self._robot.num_joints
        self._prev_targets = torch.zeros(self.num_envs, num_total_joints, device=self.device)

        # Object goal poses: (num_envs, 7) [pos_xyz + quat_wxyz]
        self._goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_quat[:, 0] = 1.0  # identity quaternion (w=1)

        # Keypoint offsets in object local frame: (num_keypoints, 3)
        # These are axis-aligned offsets scaled by object size
        self._keypoints_local = self._build_keypoints_local()

        # Cache palm body index (iiwa14_link_7 = wrist, used as palm reference)
        palm_ids, _ = self._robot.find_bodies(["iiwa14_link_7"])
        self._palm_body_idx = palm_ids[0]

        # Cache fingertip body indices (one per controlled finger)
        fingertip_names = [
            "left_thumb_fingertip", "left_index_fingertip",
            "left_middle_fingertip", "left_ring_fingertip",
        ]
        tip_ids = []
        for name in fingertip_names:
            try:
                ids, _ = self._robot.find_bodies([name])
                tip_ids.append(ids[0])
            except Exception:
                tip_ids.append(-1)  # body not found; fingertip obs will be zero
        self._fingertip_body_ids = tip_ids

        # Track whether object has been lifted
        self._lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._consecutive_successes = torch.zeros(self.num_envs, device=self.device)

        # Fixed goal cycling index (eval mode)
        self._goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Object initial z on table (set at reset)
        self._object_table_z = torch.full((self.num_envs,), self.cfg.table_reset_z + self.cfg.table_object_z_offset, device=self.device)

        # Backend-adaptive write methods: Newton uses mask-based, PhysX uses indexed.
        use_mask = "newton" in self.sim.physics_manager.__name__.lower()
        if use_mask:
            self._set_joint_pos_target = self._robot.set_joint_position_target
            self._write_robot_joint_pos = self._robot.write_joint_position_to_sim
            self._write_robot_joint_vel = self._robot.write_joint_velocity_to_sim
            self._write_robot_root_pose = self._robot.write_root_pose_to_sim
            self._write_robot_root_vel = self._robot.write_root_velocity_to_sim
            self._write_obj_root_pose = self._object.write_root_pose_to_sim
            self._write_obj_root_vel = self._object.write_root_velocity_to_sim
        else:
            self._set_joint_pos_target = self._robot.set_joint_position_target_index
            self._write_robot_joint_pos = self._robot.write_joint_position_to_sim_index
            self._write_robot_joint_vel = self._robot.write_joint_velocity_to_sim_index
            self._write_robot_root_pose = self._robot.write_root_pose_to_sim_index
            self._write_robot_root_vel = self._robot.write_root_velocity_to_sim_index
            self._write_obj_root_pose = self._object.write_root_pose_to_sim_index
            self._write_obj_root_vel = self._object.write_root_velocity_to_sim_index

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # Convert URDFs to USD (cached after first run)
        robot_usd = self._convert_urdf(self.cfg.robot_urdf_path, cache_subdir="robot", fix_base=True)
        table_usd = self._convert_urdf(self.cfg.table_urdf_path, cache_subdir="table", fix_base=True)
        object_usd = self._convert_urdf(self.cfg.object_urdf_path, cache_subdir="object", fix_base=False)

        # Robot articulation
        # Physics settings from isaacsim_conversion/isaacsim_env.py:
        #   disable_gravity=True, self_collision=False
        # Note: contact_offset=0.002 cannot be set via UsdFileCfg on instanced prims
        # (the URDF merge_fixed_joints=True produces instanced geometry). The default
        # PhysX contact offset is used instead.
        robot_cfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=robot_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=1000.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.cfg.robot_translation,
                joint_pos={
                    "iiwa14_joint_1": self.cfg.default_arm_joint_pos[0],
                    "iiwa14_joint_2": self.cfg.default_arm_joint_pos[1],
                    "iiwa14_joint_3": self.cfg.default_arm_joint_pos[2],
                    "iiwa14_joint_4": self.cfg.default_arm_joint_pos[3],
                    "iiwa14_joint_5": self.cfg.default_arm_joint_pos[4],
                    "iiwa14_joint_6": self.cfg.default_arm_joint_pos[5],
                    "iiwa14_joint_7": self.cfg.default_arm_joint_pos[6],
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["iiwa14_joint_.*"],
                    stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("iiwa14")},
                    damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("iiwa14")},
                ),
                "hand": ImplicitActuatorCfg(
                    joint_names_expr=["left_.*"],
                    stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("left")},
                    damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("left")},
                ),
            },
        )
        self._robot = Articulation(robot_cfg)

        # Table (kinematic static body — URDF converter adds ArticulationRoot which
        # must be disabled, then kinematic_enabled keeps it fixed in place)
        table_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=table_usd,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=self.cfg.table_translation),
        )
        self._table = RigidObject(table_cfg)

        # Object (dynamic rigid body)
        object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=object_usd,
            ),
        )
        self._object = RigidObject(object_cfg)

        # Ground plane and lighting
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Register with scene for cloning across envs
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["table"] = self._table
        self.scene.rigid_objects["object"] = self._object

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert [-1, 1] policy actions to joint position targets with EMA smoothing."""
        self._actions = actions.clone()

        joint_pos = wp.to_torch(self._robot.data.joint_pos)  # (N, num_joints)

        # Arm: absolute position targets with EMA
        arm_pos = joint_pos[:, self._arm_joint_ids]
        arm_targets = arm_pos + (
            self.cfg.dof_speed_scale * self.physics_dt * actions[:, :7]
        )
        arm_targets = torch.clamp(arm_targets, -2 * math.pi, 2 * math.pi)

        # Hand: scale from [-1,1] to joint limits
        hand_pos = joint_pos[:, self._hand_joint_ids]
        hand_targets = hand_pos + (
            self.cfg.dof_speed_scale * self.physics_dt * actions[:, 7:]
        )

        # EMA smoothing
        prev_arm = self._prev_targets[:, self._arm_joint_ids]
        prev_hand = self._prev_targets[:, self._hand_joint_ids]
        arm_alpha = self.cfg.arm_moving_average
        hand_alpha = self.cfg.hand_moving_average
        arm_targets = arm_alpha * arm_targets + (1.0 - arm_alpha) * prev_arm
        hand_targets = hand_alpha * hand_targets + (1.0 - hand_alpha) * prev_hand

        self._prev_targets[:, self._arm_joint_ids] = arm_targets
        self._prev_targets[:, self._hand_joint_ids] = hand_targets

    def _apply_action(self) -> None:
        self._set_joint_pos_target(
            target=self._prev_targets[:, self._policy_joint_ids],
            joint_ids=self._policy_joint_ids,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        joint_pos = wp.to_torch(self._robot.data.joint_pos)   # (N, num_joints)
        joint_vel = wp.to_torch(self._robot.data.joint_vel)   # (N, num_joints)

        # Policy-controlled joints only (11-dim)
        policy_joint_pos = joint_pos[:, self._policy_joint_ids]
        policy_joint_vel = joint_vel[:, self._policy_joint_ids]
        prev_actions = self._prev_targets[:, self._policy_joint_ids]

        # Palm pose (iiwa14_link_7 = wrist, used as palm reference)
        body_pos_w = wp.to_torch(self._robot.data.body_pos_w)   # (N, num_bodies, 3)
        body_quat_w = wp.to_torch(self._robot.data.body_quat_w)  # (N, num_bodies, 4)
        palm_pos_w = body_pos_w[:, self._palm_body_idx, :]   # (N, 3)
        palm_quat_w = body_quat_w[:, self._palm_body_idx, :]  # (N, 4) wxyz

        # Object pose
        obj_pos_w = wp.to_torch(self._object.data.root_pos_w)   # (N, 3)
        obj_quat_w = wp.to_torch(self._object.data.root_quat_w)  # (N, 4) wxyz

        # Fingertip positions relative to palm (4 fingertips × 3 = 12)
        fingertip_pos_rel_palm = torch.zeros(self.num_envs, 12, device=self.device)
        for i, tip_idx in enumerate(self._fingertip_body_ids):
            if tip_idx >= 0:
                tip_pos = body_pos_w[:, tip_idx, :]  # (N, 3)
                fingertip_pos_rel_palm[:, i * 3 : i * 3 + 3] = tip_pos - palm_pos_w

        # Keypoints: object keypoints in palm frame and goal frame
        keypoints_world = self._compute_keypoints_world(obj_pos_w, obj_quat_w)  # (N, K, 3)
        goal_keypoints_world = self._compute_keypoints_world(self._goal_pos, self._goal_quat)
        keypoints_rel_palm = self._to_palm_frame(keypoints_world, palm_pos_w, palm_quat_w)  # (N, K*3)
        keypoints_rel_goal = (keypoints_world - goal_keypoints_world).reshape(self.num_envs, -1)

        # Object scales (fixed per object, 1.0 for now)
        object_scales = torch.ones(self.num_envs, 3, device=self.device)

        obs = torch.cat([
            policy_joint_pos,           # 11
            policy_joint_vel,           # 11
            prev_actions,               # 11
            palm_pos_w,                 # 3
            palm_quat_w,                # 4
            obj_quat_w,                 # 4
            fingertip_pos_rel_palm,     # 12
            keypoints_rel_palm,         # K*3
            keypoints_rel_goal,         # K*3
            object_scales,              # 3
        ], dim=-1)

        obs = torch.clamp(obs, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        joint_vel = wp.to_torch(self._robot.data.joint_vel)
        obj_pos_w = wp.to_torch(self._object.data.root_pos_w)
        obj_quat_w = wp.to_torch(self._object.data.root_quat_w)

        # Lifting reward
        obj_z = obj_pos_w[:, 2]
        table_z = self._object_table_z
        lift_height = obj_z - table_z
        newly_lifted = (~self._lifted_object) & (lift_height > self.cfg.lifting_bonus_threshold)
        self._lifted_object |= newly_lifted
        lifting_rew = torch.clamp(lift_height, 0.0, 0.5) * self.cfg.lifting_rew_scale
        lifting_rew += newly_lifted.float() * self.cfg.lifting_bonus

        # Keypoint reward (after lifting)
        goal_kpts = self._compute_keypoints_world(self._goal_pos, self._goal_quat)
        obj_kpts = self._compute_keypoints_world(obj_pos_w, obj_quat_w)
        kpt_dist = torch.norm(obj_kpts - goal_kpts, dim=-1).max(dim=-1).values  # (N,)
        keypoint_rew = -kpt_dist * self.cfg.keypoint_rew_scale * self._lifted_object.float()

        # Success bonus
        success = (kpt_dist < self.cfg.success_tolerance) & self._lifted_object
        self._consecutive_successes += success.float()
        self._consecutive_successes *= success.float()  # reset if not success
        success_bonus = success.float() * (self.cfg.reach_goal_bonus / self.cfg.consecutive_success_steps)

        # Action penalties
        arm_vel = joint_vel[:, self._arm_joint_ids]
        hand_vel = joint_vel[:, self._hand_joint_ids]
        kuka_penalty = -torch.sum(torch.abs(arm_vel), dim=-1) * self.cfg.kuka_actions_penalty_scale
        hand_penalty = -torch.sum(torch.abs(hand_vel), dim=-1) * self.cfg.hand_actions_penalty_scale

        return lifting_rew + keypoint_rew + success_bonus + kuka_penalty + hand_penalty

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos_w = wp.to_torch(self._object.data.root_pos_w)
        obj_z = obj_pos_w[:, 2]

        # Object fell off table
        fell = obj_z < self.cfg.object_fall_z_threshold

        # Too many consecutive successes (reset on goal achieved)
        goal_reached = self._consecutive_successes >= self.cfg.consecutive_success_steps

        terminated = fell | goal_reached
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # Reset tracking state
        self._lifted_object[env_ids] = False
        self._consecutive_successes[env_ids] = 0.0

        # ---- Robot joint positions ----
        default_pos = wp.to_torch(self._robot.data.default_joint_pos)[env_ids].clone()
        if not self.cfg.eval_mode:
            noise = sample_uniform(
                -self.cfg.reset_dof_pos_noise_arm,
                self.cfg.reset_dof_pos_noise_arm,
                (n, len(self._arm_joint_ids)),
                device=self.device,
            )
            default_pos[:, self._arm_joint_ids] += noise
            noise_hand = sample_uniform(
                -self.cfg.reset_dof_pos_noise_fingers,
                self.cfg.reset_dof_pos_noise_fingers,
                (n, len(self._hand_joint_ids)),
                device=self.device,
            )
            default_pos[:, self._hand_joint_ids] += noise_hand

        default_vel = torch.zeros_like(default_pos)
        if not self.cfg.eval_mode:
            vel_noise = sample_uniform(
                -self.cfg.reset_dof_vel_noise,
                self.cfg.reset_dof_vel_noise,
                default_vel.shape,
                device=self.device,
            )
            default_vel += vel_noise

        root_pose = wp.to_torch(self._robot.data.default_root_pose)[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_vel = wp.to_torch(self._robot.data.default_root_vel)[env_ids].clone()

        self._write_robot_root_pose(root_pose=root_pose, env_ids=env_ids)
        self._write_robot_root_vel(root_velocity=root_vel, env_ids=env_ids)
        self._write_robot_joint_pos(position=default_pos, env_ids=env_ids)
        self._write_robot_joint_vel(velocity=default_vel, env_ids=env_ids)

        # Reset prev targets to default position
        self._prev_targets[env_ids] = default_pos

        # ---- Object pose ----
        if self.cfg.use_fixed_object_pose:
            # Fixed pose from eval config: [x, y, z, qx, qy, qz, qw] (xyzw)
            fp = self.cfg.fixed_object_pose
            obj_pos = torch.tensor([fp[0], fp[1], fp[2]], device=self.device).unsqueeze(0).expand(n, 3).clone()
            # Convert xyzw → wxyz for IsaacLab
            obj_quat = torch.tensor([fp[6], fp[3], fp[4], fp[5]], device=self.device).unsqueeze(0).expand(n, 4).clone()
        else:
            obj_pos = torch.zeros(n, 3, device=self.device)
            noise_x = 0.0 if self.cfg.eval_mode else self.cfg.reset_position_noise_x
            noise_y = 0.0 if self.cfg.eval_mode else self.cfg.reset_position_noise_y
            noise_z = 0.0 if self.cfg.eval_mode else self.cfg.reset_position_noise_z
            obj_pos[:, 0] += sample_uniform(-noise_x, noise_x, (n,), device=self.device)
            obj_pos[:, 1] += sample_uniform(-noise_y, noise_y, (n,), device=self.device)
            obj_pos[:, 2] = self._object_table_z[env_ids] + sample_uniform(-noise_z, noise_z, (n,), device=self.device)

            # Random object rotation
            if self.cfg.randomize_object_rotation and not self.cfg.eval_mode:
                obj_quat = torch.randn(n, 4, device=self.device)
                obj_quat = obj_quat / obj_quat.norm(dim=-1, keepdim=True)
            else:
                obj_quat = torch.zeros(n, 4, device=self.device)
                obj_quat[:, 0] = 1.0  # wxyz identity

        obj_pos = obj_pos + self.scene.env_origins[env_ids]
        obj_pose = torch.cat([obj_pos, obj_quat], dim=-1)
        self._write_obj_root_pose(root_pose=obj_pose, env_ids=env_ids)
        obj_vel = torch.zeros(n, 6, device=self.device)
        self._write_obj_root_vel(root_velocity=obj_vel, env_ids=env_ids)

        # ---- Sample new goal pose ----
        self._sample_goal(env_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_urdf(self, urdf_path: str, cache_subdir: str, fix_base: bool) -> str:
        """Convert a URDF to USD and return the USD path (cached).

        Gains are NOT baked into the USD here — they are applied at runtime via
        :class:`ImplicitActuatorCfg` in :meth:`_setup_scene`, which writes
        directly to PhysX without any unit conversion.
        """
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=f"{self.cfg.usd_cache_dir}/{cache_subdir}",
            force_usd_conversion=False,  # use cache if available
            fix_base=fix_base,
            merge_fixed_joints=True,  # matches Isaac Gym collapse_fixed_joints=True
            self_collision=False,
        )
        converter = UrdfConverter(cfg)
        return converter.usd_path

    def _build_keypoints_local(self) -> torch.Tensor:
        """Build axis-aligned keypoint offsets in object local frame."""
        K = self.cfg.num_keypoints
        scale = self.cfg.keypoint_scale * self.cfg.object_base_size
        # Distribute keypoints on unit cube corners / axes
        offsets = torch.zeros(K, 3, device=self.device)
        for i in range(K):
            offsets[i, i % 3] = scale * (1.0 if i < K // 2 else -1.0)
        return offsets  # (K, 3)

    def _compute_keypoints_world(
        self, pos: torch.Tensor, quat_wxyz: torch.Tensor
    ) -> torch.Tensor:
        """Transform local keypoints to world frame. Returns (N, K, 3)."""
        N = pos.shape[0]
        K = self._keypoints_local.shape[0]
        kpts = self._keypoints_local.unsqueeze(0).expand(N, K, 3)  # (N, K, 3)
        # Rotate each keypoint by object quaternion
        kpts_flat = kpts.reshape(N * K, 3)
        quat_exp = quat_wxyz.unsqueeze(1).expand(N, K, 4).reshape(N * K, 4)
        kpts_world = quat_rotate(quat_exp, kpts_flat).reshape(N, K, 3)
        kpts_world += pos.unsqueeze(1)
        return kpts_world

    def _to_palm_frame(
        self,
        kpts_world: torch.Tensor,
        palm_pos: torch.Tensor,
        palm_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Transform world-frame keypoints into palm frame. Returns (N, K*3)."""
        N, K, _ = kpts_world.shape
        rel = kpts_world - palm_pos.unsqueeze(1)  # (N, K, 3)
        rel_flat = rel.reshape(N * K, 3)
        quat_exp = palm_quat.unsqueeze(1).expand(N, K, 4).reshape(N * K, 4)
        in_palm = quat_rotate_inverse(quat_exp, rel_flat).reshape(N, K * 3)
        return in_palm

    def _sample_goal(self, env_ids: torch.Tensor):
        """Sample goal poses for the given envs.

        When :attr:`~SimToolRealEnvCfg.use_fixed_goals` is True, cycles through
        :attr:`~SimToolRealEnvCfg.fixed_goal_poses` in sequence (each env
        independently).  Otherwise samples a random orientation above the table.
        """
        n = len(env_ids)

        if self.cfg.use_fixed_goals and len(self.cfg.fixed_goal_poses) > 0:
            goals = self.cfg.fixed_goal_poses
            num_goals = len(goals)
            goal_pos = torch.zeros(n, 3, device=self.device)
            goal_quat = torch.zeros(n, 4, device=self.device)
            for i, env_id in enumerate(env_ids):
                idx = int(self._goal_idx[env_id].item()) % num_goals
                gp = goals[idx]
                goal_pos[i] = torch.tensor([gp[0], gp[1], gp[2]], device=self.device)
                # xyzw → wxyz
                goal_quat[i] = torch.tensor([gp[6], gp[3], gp[4], gp[5]], device=self.device)
                self._goal_idx[env_id] = (idx + 1) % num_goals
            goal_pos = goal_pos + self.scene.env_origins[env_ids]
        else:
            goal_pos = torch.zeros(n, 3, device=self.device)
            goal_pos[:, 2] = self._object_table_z[env_ids] + self.cfg.lifting_bonus_threshold + 0.05
            goal_pos = goal_pos + self.scene.env_origins[env_ids]
            goal_quat = torch.randn(n, 4, device=self.device)
            goal_quat = goal_quat / goal_quat.norm(dim=-1, keepdim=True)

        self._goal_pos[env_ids] = goal_pos
        self._goal_quat[env_ids] = goal_quat
