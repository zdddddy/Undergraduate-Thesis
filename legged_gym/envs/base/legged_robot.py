# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        # Optional runtime diagnostics for observation scale / NaN checks.
        self._obs_debug_steps = max(int(os.getenv("OBS_DEBUG_STEPS", "0")), 0)
        self._obs_debug_split = max(int(os.getenv("OBS_DEBUG_SPLIT", "48")), 0)
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._init_warp()
        self._init_nsr_runtime()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.cfg.env.use_warp:
            self._update_warp_sensor_pose()
            if self.cfg.sensor.add_depth:
                if self.depth_image_update_decimation <= 1 or (self.depth_image_update_counter % self.depth_image_update_decimation) == 0:
                    self.update_depth_images()
                self.depth_image_update_counter += 1

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8)
        min_base_height = float(getattr(self.cfg.rewards, "min_base_height", -1.0))
        if min_base_height > 0.0:
            self.reset_buf |= self.root_states[:, 2] < min_base_height
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.feet_air_time_clearance[env_ids] = 0.
        self.last_contacts_clearance[env_ids] = False
        if getattr(self, "nsr_enabled", False):
            self._reset_nsr_state(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            measured_heights = self.measured_heights
            if not torch.is_tensor(measured_heights):
                measured_heights = self._get_heights_nsr() if getattr(self, "nsr_enabled", False) else self._get_heights()
                self.measured_heights = measured_heights
            height_center = float(getattr(self.cfg.terrain, "height_measurements_center", 0.5))
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - height_center - measured_heights,
                -1,
                1.,
            ) * self.obs_scales.height_measurements
            # Optional explicit height-map noise in meters (for RL robustness to NSR prediction errors).
            h_noise_min_m = float(getattr(self.cfg.noise, "height_measurements_noise_m_min", 0.0))
            h_noise_max_m = float(getattr(self.cfg.noise, "height_measurements_noise_m_max", 0.0))
            if h_noise_max_m > max(h_noise_min_m, 0.0):
                h_noise_min_m = max(h_noise_min_m, 0.0)
                amp_m = torch.empty((self.num_envs, 1), device=self.device, dtype=heights.dtype).uniform_(
                    h_noise_min_m, h_noise_max_m
                )
                heights += (2.0 * torch.rand_like(heights) - 1.0) * amp_m * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self._maybe_debug_observations()

    def _maybe_debug_observations(self):
        if self._obs_debug_steps <= 0:
            return
        if int(self.common_step_counter) >= self._obs_debug_steps:
            return

        obs = self.obs_buf
        split = int(min(max(self._obs_debug_split, 0), obs.shape[1]))

        def _stats(x: torch.Tensor):
            finite = torch.isfinite(x)
            invalid = int((~finite).sum().item())
            if not torch.any(finite):
                return float("nan"), float("nan"), float("nan"), float("nan"), invalid
            vals = x[finite]
            return (
                float(vals.mean().item()),
                float(vals.std(unbiased=False).item()),
                float(vals.min().item()),
                float(vals.max().item()),
                invalid,
            )

        all_mean, all_std, all_min, all_max, all_invalid = _stats(obs)
        msg = (
            f"[obs_debug] step={int(self.common_step_counter)} dim={obs.shape[1]} "
            f"all(mean={all_mean:.4g}, std={all_std:.4g}, min={all_min:.4g}, max={all_max:.4g}, invalid={all_invalid})"
        )
        if split > 0:
            p_mean, p_std, p_min, p_max, p_invalid = _stats(obs[:, :split])
            msg += (
                f" | part0[:{split}](mean={p_mean:.4g}, std={p_std:.4g}, "
                f"min={p_min:.4g}, max={p_max:.4g}, invalid={p_invalid})"
            )
        if split < obs.shape[1]:
            h_mean, h_std, h_min, h_max, h_invalid = _stats(obs[:, split:])
            msg += (
                f" | part1[{split}:](mean={h_mean:.4g}, std={h_std:.4g}, "
                f"min={h_min:.4g}, max={h_max:.4g}, invalid={h_invalid})"
            )
        print(msg)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _init_warp(self):
        if not self.cfg.env.use_warp:
            return
        if not self.cfg.sensor.add_depth:
            raise ValueError("Depth sensor is required for warp. Set cfg.sensor.add_depth=True.")
        try:
            import warp as wp  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError as exc:
            raise ImportError("Warp camera requires 'warp' and 'trimesh' packages.") from exc
        from legged_gym.warp.warp_cam import WarpCam

        wp.init()
        self._create_warp_envs()
        self._create_warp_tensors()
        self.warp_sensor = WarpCam(self.warp_tensor_dict, self.num_envs, self.cfg.sensor, self.mesh_ids, self.device)
        pixels = self.warp_sensor.update()
        self.depth_images[:, :, 0] = pixels  # pixels: [num_envs, num_sensors, H, W]

    def _create_warp_envs(self):
        import warp as wp
        import trimesh

        if self.cfg.terrain.mesh_type == "trimesh":
            vertices = self.terrain.vertices
            triangles = self.terrain.triangles
        elif self.cfg.terrain.mesh_type == "heightfield":
            from isaacgym import terrain_utils
            vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.terrain.height_field_raw,
                self.cfg.terrain.horizontal_scale,
                self.cfg.terrain.vertical_scale,
                self.cfg.terrain.slope_treshold,
            )
        else:
            raise ValueError("Warp camera requires terrain mesh_type 'trimesh' or 'heightfield'.")

        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

        # align mesh with Isaac Gym terrain transform (border offset)
        transform = np.zeros((3,), dtype=np.float32)
        transform[0] = -self.cfg.terrain.border_size
        transform[1] = -self.cfg.terrain.border_size
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)

        vertex_tensor = torch.tensor(
            terrain_mesh.vertices,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        vertex_vec3_array = wp.from_torch(vertex_tensor, dtype=wp.vec3)
        faces_wp_int32_array = wp.from_numpy(
            terrain_mesh.faces.flatten(),
            dtype=wp.int32,
            device=self.device,
        )

        self.wp_meshes = wp.Mesh(points=vertex_vec3_array, indices=faces_wp_int32_array)
        self.mesh_ids = wp.array([self.wp_meshes.id], dtype=wp.uint64)

    def _create_warp_tensors(self):
        self.warp_tensor_dict = {}
        pointcloud_dims = 3 * (self.cfg.sensor.depth_camera_config.return_pointcloud == True)
        if pointcloud_dims > 0:
            self.depth_image_tensor_warp = torch.zeros(
                (
                    self.num_envs,
                    self.cfg.sensor.num_sensors,
                    self.cfg.sensor.depth_camera_config.resolution[1],  # height
                    self.cfg.sensor.depth_camera_config.resolution[0],  # width
                    pointcloud_dims,  # xyz
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.depth_image_tensor_warp = torch.zeros(
                (
                    self.num_envs,
                    self.cfg.sensor.num_sensors,
                    self.cfg.sensor.depth_camera_config.resolution[1],  # height
                    self.cfg.sensor.depth_camera_config.resolution[0],  # width
                ),
                dtype=torch.float32,
                device=self.device,
            )

        self.sensor_pos_tensor = torch.zeros((self.num_envs, self.cfg.sensor.num_sensors, 3), device=self.device)
        self.sensor_quat_tensor = torch.zeros((self.num_envs, self.cfg.sensor.num_sensors, 4), device=self.device)

        pos_cfg = self.cfg.sensor.depth_camera_config.pos
        euler_cfg = self.cfg.sensor.depth_camera_config.euler
        if isinstance(pos_cfg[0], (list, tuple)):
            if len(pos_cfg) != self.cfg.sensor.num_sensors:
                raise ValueError("depth_camera_config.pos length must match sensor.num_sensors")
            pos_list = pos_cfg
        else:
            pos_list = [pos_cfg] * self.cfg.sensor.num_sensors
        if isinstance(euler_cfg[0], (list, tuple)):
            if len(euler_cfg) != self.cfg.sensor.num_sensors:
                raise ValueError("depth_camera_config.euler length must match sensor.num_sensors")
            euler_list = euler_cfg
        else:
            euler_list = [euler_cfg] * self.cfg.sensor.num_sensors

        self.sensor_offset_pos = torch.tensor(pos_list, device=self.device)  # (S,3)
        rpy_offset = torch.tensor(euler_list, device=self.device)  # (S,3)
        self.sensor_offset_quat = quat_from_euler_xyz(
            rpy_offset[:, 0], rpy_offset[:, 1], rpy_offset[:, 2]
        )  # (S,4)

        self.warp_tensor_dict["depth_image_tensor"] = self.depth_image_tensor_warp
        self.warp_tensor_dict["device"] = self.device
        self.warp_tensor_dict["num_envs"] = self.num_envs
        self.warp_tensor_dict["num_sensors"] = self.cfg.sensor.num_sensors
        self.warp_tensor_dict["sensor_pos_tensor"] = self.sensor_pos_tensor
        self.warp_tensor_dict["sensor_quat_tensor"] = self.sensor_quat_tensor
        self.warp_tensor_dict["mesh_ids"] = self.mesh_ids

    def _update_warp_sensor_pose(self):
        base_quat = self.base_quat[:, None, :].expand(-1, self.cfg.sensor.num_sensors, -1)
        base_pos = self.root_states[:, None, 0:3].expand(-1, self.cfg.sensor.num_sensors, -1)
        offset_quat = self.sensor_offset_quat[None, :, :].expand(self.num_envs, -1, -1)
        offset_pos = self.sensor_offset_pos[None, :, :].expand(self.num_envs, -1, -1)
        sensor_quat = quat_mul(base_quat, offset_quat)
        sensor_pos = base_pos + quat_apply(base_quat, offset_pos)
        self.sensor_pos_tensor[:, :, :] = sensor_pos
        self.sensor_quat_tensor[:, :, :] = sensor_quat

    def update_depth_images(self):
        """ Update depth images from the warp camera sensor. """
        if not self.cfg.env.use_warp:
            raise NotImplementedError("Depth image update not implemented without warp.")
        pixels = self.warp_sensor.update()
        self.depth_images[:, :, 0] = pixels
        if self.cfg.sensor.depth_camera_config.calculate_depth and not self.cfg.sensor.depth_camera_config.return_pointcloud:
            near_clip = self.cfg.sensor.depth_camera_config.near_clip
            far_clip = self.cfg.sensor.depth_camera_config.far_clip
            self.depth_images = torch.clip(self.depth_images, near_clip, far_clip)
            self.depth_images = (self.depth_images - near_clip) / (far_clip - near_clip) - 0.5

    def draw_debug_depth_images(self):
        """Draw warp pointcloud in the Isaac Gym viewer (env 0)."""
        if not self.viewer:
            return
        if not (self.cfg.env.use_warp and self.cfg.sensor.add_depth):
            return
        if not self.cfg.sensor.depth_camera_config.return_pointcloud:
            return
        far_plane = self.cfg.sensor.depth_camera_config.far_plane

        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(0, 1, 1))

        for sensor_id in range(self.cfg.sensor.num_sensors):
            pointcloud = self.depth_images[0, sensor_id, 0]
            pointcloud_np = pointcloud.detach().cpu().numpy().astype(np.float32)
            cam_pos = self.sensor_pos_tensor[0, sensor_id].detach().cpu().numpy()

            # simple range filter to skip "no hit" points
            dist = np.linalg.norm(pointcloud_np - cam_pos[None, None, :], axis=2)
            mask = dist < (far_plane - 1e-3)

            for i in range(pointcloud_np.shape[0]):
                for j in range(pointcloud_np.shape[1]):
                    if not mask[i, j]:
                        continue
                    x, y, z = pointcloud_np[i, j, :]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)

    def _nsr_quat_to_rotmat_xyzw_torch(self, q):
        n = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)
        qn = q / n
        x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=1)
        row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=1)
        row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    def _nsr_quat_to_yaw_rotmat_xyzw_torch(self, q):
        n = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)
        qn = q / n
        x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        zeros = torch.zeros_like(cy)
        ones = torch.ones_like(cy)
        row0 = torch.stack([cy, -sy, zeros], dim=1)
        row1 = torch.stack([sy, cy, zeros], dim=1)
        row2 = torch.stack([zeros, zeros, ones], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    def _nsr_warp_prev_to_current(self, prev_map, prev_pose7, cur_pose7, map_size, prev_valid=None, gravity_aligned=True):
        bsz, _, h, w = prev_map.shape
        if prev_valid is None:
            prev_valid = torch.ones_like(prev_map)
        if h < 2 or w < 2:
            return prev_map, prev_valid

        device = prev_map.device
        dtype = prev_map.dtype
        half = 0.5 * float(map_size)
        res_x = float(map_size) / float(h)
        res_y = float(map_size) / float(w)

        x_centers = (torch.arange(h, device=device, dtype=dtype) + 0.5) * res_x - half
        y_centers = (torch.arange(w, device=device, dtype=dtype) + 0.5) * res_y - half
        x_grid = x_centers[:, None].expand(h, w)
        y_grid = y_centers[None, :].expand(h, w)
        p_cur = torch.stack([x_grid, y_grid, torch.zeros_like(x_grid)], dim=-1)
        p_cur = p_cur.unsqueeze(0).expand(bsz, h, w, 3)

        pos_prev = prev_pose7[:, :3].to(dtype=dtype)
        pos_cur = cur_pose7[:, :3].to(dtype=dtype)
        if gravity_aligned:
            rot_prev = self._nsr_quat_to_yaw_rotmat_xyzw_torch(prev_pose7[:, 3:7].to(dtype=dtype))
            rot_cur = self._nsr_quat_to_yaw_rotmat_xyzw_torch(cur_pose7[:, 3:7].to(dtype=dtype))
        else:
            rot_prev = self._nsr_quat_to_rotmat_xyzw_torch(prev_pose7[:, 3:7].to(dtype=dtype))
            rot_cur = self._nsr_quat_to_rotmat_xyzw_torch(cur_pose7[:, 3:7].to(dtype=dtype))

        p_world = torch.einsum("bij,bhwj->bhwi", rot_cur, p_cur) + pos_cur[:, None, None, :]
        p_prev = torch.einsum("bij,bhwj->bhwi", rot_prev.transpose(1, 2), p_world - pos_prev[:, None, None, :])

        x_prev = p_prev[..., 0]
        y_prev = p_prev[..., 1]
        i_prev = (x_prev + half) / res_x - 0.5
        j_prev = (y_prev + half) / res_y - 0.5

        grid_x = 2.0 * (j_prev / max(w - 1, 1)) - 1.0
        grid_y = 2.0 * (i_prev / max(h - 1, 1)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        warped_h_prev_raw = F.grid_sample(
            prev_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        warped_valid = F.grid_sample(
            prev_valid,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).clamp_(0.0, 1.0)

        valid_eps = 5e-2
        warped_h_prev = torch.where(
            warped_valid > valid_eps,
            warped_h_prev_raw / warped_valid.clamp_min(valid_eps),
            torch.zeros_like(warped_h_prev_raw),
        )

        p_prev_obj = torch.stack([x_prev, y_prev, warped_h_prev[:, 0]], dim=-1)
        p_world_obj = torch.einsum("bij,bhwj->bhwi", rot_prev, p_prev_obj) + pos_prev[:, None, None, :]
        p_cur_obj = torch.einsum(
            "bij,bhwj->bhwi",
            rot_cur.transpose(1, 2),
            p_world_obj - pos_cur[:, None, None, :],
        )
        warped_h_cur = p_cur_obj[..., 2].unsqueeze(1)
        warped_h_cur = torch.where(
            warped_valid > valid_eps,
            warped_h_cur,
            torch.zeros_like(warped_h_cur),
        )
        return warped_h_cur, warped_valid

    def _nsr_build_model_input(self, meas_h, meas_m, prev_h, prev_valid, in_channels):
        if in_channels == 4:
            return torch.cat([meas_h, meas_m, prev_h, prev_valid], dim=1)
        if in_channels == 3:
            return torch.cat([meas_h, meas_m, prev_h], dim=1)
        raise ValueError(f"Unsupported NSR in_channels={in_channels}; expected 3 or 4.")

    def _nsr_unpack_outputs(self, model_out):
        if isinstance(model_out, (tuple, list)):
            h = model_out[0]
            edge_logits = model_out[1] if len(model_out) > 1 else None
            return h, edge_logits
        if isinstance(model_out, dict):
            return model_out.get("height", None), model_out.get("edge_logits", None)
        return model_out, None

    def _nsr_binarize_valid(self, mask, threshold):
        if threshold <= 0.0:
            return mask.clamp(0.0, 1.0)
        if threshold >= 1.0:
            return (mask >= 1.0).float()
        return (mask > float(threshold)).float()

    def _init_nsr_runtime(self):
        self.nsr_enabled = bool(getattr(getattr(self.cfg, "nsr", None), "enable", False))
        if not self.nsr_enabled:
            return

        if not self.cfg.terrain.measure_heights:
            raise ValueError("NSR runtime requires terrain.measure_heights=True.")
        if not self.cfg.env.use_warp:
            raise ValueError("NSR runtime requires env.use_warp=True.")
        if not self.cfg.sensor.add_depth:
            raise ValueError("NSR runtime requires sensor.add_depth=True.")
        if not self.cfg.sensor.depth_camera_config.return_pointcloud:
            raise ValueError("NSR runtime requires depth pointclouds (return_pointcloud=True).")
        if not self.cfg.sensor.depth_camera_config.pointcloud_in_world_frame:
            raise ValueError("NSR runtime expects pointcloud_in_world_frame=True.")

        nsr_cfg = self.cfg.nsr
        ckpt_path = str(getattr(nsr_cfg, "ckpt", "")).strip()
        if not ckpt_path:
            raise ValueError("NSR runtime enabled but cfg.nsr.ckpt is empty.")
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(LEGGED_GYM_ROOT_DIR, ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"NSR checkpoint not found: {ckpt_path}")

        from nsr_height.model import HeightRecurrentUNet

        ckpt = torch.load(ckpt_path, map_location="cpu")
        train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        self.nsr_map_size = float(train_args.get("map_size", getattr(nsr_cfg, "map_size", 3.2)))
        self.nsr_resolution = float(train_args.get("resolution", getattr(nsr_cfg, "resolution", 0.05)))
        self.nsr_fill_value = float(train_args.get("fill_value", getattr(nsr_cfg, "fill_value", 0.0)))
        self.nsr_in_channels = int(train_args.get("in_channels", getattr(nsr_cfg, "in_channels", 4)))
        self.nsr_base_channels = int(train_args.get("base_channels", getattr(nsr_cfg, "base_channels", 32)))
        self.nsr_use_edge_head = bool(train_args.get("use_edge_head", False))
        self.nsr_norm_type = str(train_args.get("norm_type", getattr(nsr_cfg, "norm_type", "group")))
        self.nsr_group_norm_groups = int(train_args.get("group_norm_groups", getattr(nsr_cfg, "group_norm_groups", 8)))
        self.nsr_gravity_aligned = bool(train_args.get("gravity_aligned", getattr(nsr_cfg, "gravity_aligned", True)))
        self.nsr_align_prev = bool(train_args.get("align_prev", getattr(nsr_cfg, "align_prev", True)))
        self.nsr_disable_prev = bool(train_args.get("disable_prev", getattr(nsr_cfg, "disable_prev", False)))
        self.nsr_prev_valid_threshold = float(
            train_args.get("prev_valid_threshold", getattr(nsr_cfg, "prev_valid_threshold", 0.5))
        )
        self.nsr_memory_meas_override = bool(
            train_args.get("memory_meas_override", getattr(nsr_cfg, "memory_meas_override", True))
        )
        self.nsr_residual_from_base = bool(
            train_args.get("residual_from_base", getattr(nsr_cfg, "residual_from_base", False))
        )
        self.nsr_residual_scale = float(train_args.get("residual_scale", getattr(nsr_cfg, "residual_scale", 0.2)))
        self.nsr_residual_tanh = bool(train_args.get("residual_tanh", getattr(nsr_cfg, "residual_tanh", True)))
        self.nsr_far_plane = float(getattr(nsr_cfg, "max_valid_depth", self.cfg.sensor.depth_camera_config.far_plane))

        self.nsr_grid_size = int(round(self.nsr_map_size / self.nsr_resolution))
        if self.nsr_grid_size <= 1:
            raise ValueError(f"Invalid NSR map shape: map_size={self.nsr_map_size}, resolution={self.nsr_resolution}")

        self.nsr_model = HeightRecurrentUNet(
            in_channels=self.nsr_in_channels,
            base_channels=self.nsr_base_channels,
            out_channels=1,
            use_edge_head=self.nsr_use_edge_head,
            norm_type=self.nsr_norm_type,
            group_norm_groups=self.nsr_group_norm_groups,
        ).to(self.device)
        self.nsr_model.load_state_dict(state_dict, strict=False)
        self.nsr_model.eval()

        gs = self.nsr_grid_size
        self.nsr_prev_pred = torch.full(
            (self.num_envs, 1, gs, gs),
            float(self.nsr_fill_value),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.nsr_prev_valid = torch.zeros(
            (self.num_envs, 1, gs, gs),
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.nsr_prev_pose7 = self.root_states[:, :7].clone().detach()

        half = 0.5 * self.nsr_map_size
        gx = self.height_points[0, :, 0]
        gy = self.height_points[0, :, 1]
        i = (gx + half) / self.nsr_resolution - 0.5
        j = (gy + half) / self.nsr_resolution - 0.5
        grid_x = 2.0 * (j / max(gs - 1, 1)) - 1.0
        grid_y = 2.0 * (i / max(gs - 1, 1)) - 1.0
        self.nsr_sample_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2).to(self.device)

    def _reset_nsr_state(self, env_ids):
        if len(env_ids) == 0:
            return
        self.nsr_prev_pred[env_ids] = float(self.nsr_fill_value)
        self.nsr_prev_valid[env_ids] = 0.0
        self.nsr_prev_pose7[env_ids] = self.root_states[env_ids, :7]

    def _build_local_measurement_heightmaps(self):
        points_world = self.depth_images[:, :, 0]
        num_envs, num_sensors, h, w, _ = points_world.shape
        points = points_world.view(num_envs, num_sensors, h * w, 3)
        cam_pos = self.sensor_pos_tensor[:, :, None, :]
        dist = torch.norm(points - cam_pos, dim=-1)
        valid = torch.isfinite(points).all(dim=-1) & (dist < (self.nsr_far_plane - 1e-3))

        points = points.view(num_envs, num_sensors * h * w, 3)
        valid = valid.view(num_envs, num_sensors * h * w)
        rel = points - self.root_states[:, None, :3]

        if self.nsr_gravity_aligned:
            x = self.base_quat[:, 0]
            y = self.base_quat[:, 1]
            z = self.base_quat[:, 2]
            wq = self.base_quat[:, 3]
            yaw = torch.atan2(2.0 * (wq * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cy = torch.cos(yaw)[:, None]
            sy = torch.sin(yaw)[:, None]
            lx = cy * rel[..., 0] + sy * rel[..., 1]
            ly = -sy * rel[..., 0] + cy * rel[..., 1]
            lz = rel[..., 2]
        else:
            rel_flat = rel.reshape(-1, 3)
            quat_rep = self.base_quat[:, None, :].expand(-1, rel.shape[1], -1).reshape(-1, 4)
            local_flat = quat_rotate_inverse(quat_rep, rel_flat)
            local = local_flat.view(num_envs, -1, 3)
            lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]

        gs = self.nsr_grid_size
        half = 0.5 * self.nsr_map_size
        ix = torch.floor((lx + half) / self.nsr_resolution).to(torch.long)
        iy = torch.floor((ly + half) / self.nsr_resolution).to(torch.long)
        inside = valid & (ix >= 0) & (ix < gs) & (iy >= 0) & (iy < gs)

        meas_h = torch.full(
            (num_envs, 1, gs, gs),
            float(self.nsr_fill_value),
            device=self.device,
            dtype=torch.float32,
        )
        meas_m = torch.zeros((num_envs, 1, gs, gs), device=self.device, dtype=torch.float32)

        for env_i in range(num_envs):
            use = inside[env_i]
            if not torch.any(use):
                continue
            lin = ix[env_i, use] * gs + iy[env_i, use]
            zvals = lz[env_i, use].to(torch.float32)

            flat_h = torch.full((gs * gs,), -1e9, device=self.device, dtype=torch.float32)
            if hasattr(flat_h, "scatter_reduce_"):
                flat_h.scatter_reduce_(0, lin, zvals, reduce="amax", include_self=True)
            else:
                for idx in torch.unique(lin):
                    flat_h[idx] = torch.max(zvals[lin == idx])
            flat_m = torch.zeros((gs * gs,), device=self.device, dtype=torch.float32)
            flat_m[lin] = 1.0

            out_h = torch.where(flat_m > 0.5, flat_h, torch.full_like(flat_h, float(self.nsr_fill_value)))
            meas_h[env_i, 0] = out_h.view(gs, gs)
            meas_m[env_i, 0] = flat_m.view(gs, gs)
        return meas_h, meas_m

    def _sample_local_map_at_height_points(self, local_map):
        sample_grid = self.nsr_sample_grid.expand(local_map.shape[0], -1, -1, -1)
        sampled = F.grid_sample(
            local_map,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled[:, 0, 0, :]

    def _local_height_to_world_height(self, local_h):
        if self.nsr_gravity_aligned:
            return local_h + self.root_states[:, 2:3]

        pts_local = self.height_points.clone()
        pts_local[:, :, 2] = local_h
        pcount = pts_local.shape[1]
        pts_flat = pts_local.reshape(-1, 3)
        quat_rep = self.base_quat[:, None, :].expand(-1, pcount, -1).reshape(-1, 4)
        world_flat = quat_apply(quat_rep, pts_flat)
        world = world_flat.view(self.num_envs, pcount, 3) + self.root_states[:, None, :3]
        return world[:, :, 2]

    def _get_heights_nsr(self):
        meas_h, meas_m = self._build_local_measurement_heightmaps()
        cur_pose7 = self.root_states[:, :7]

        if self.nsr_disable_prev:
            prev_in = meas_h
            prev_in_valid = meas_m
        else:
            if self.nsr_align_prev:
                prev_in, prev_in_valid = self._nsr_warp_prev_to_current(
                    self.nsr_prev_pred,
                    self.nsr_prev_pose7,
                    cur_pose7,
                    self.nsr_map_size,
                    prev_valid=self.nsr_prev_valid,
                    gravity_aligned=self.nsr_gravity_aligned,
                )
            else:
                prev_in = self.nsr_prev_pred
                prev_in_valid = self.nsr_prev_valid

        prev_in_valid = self._nsr_binarize_valid(prev_in_valid, self.nsr_prev_valid_threshold)
        model_in = self._nsr_build_model_input(meas_h, meas_m, prev_in, prev_in_valid, self.nsr_in_channels)

        with torch.no_grad():
            model_out = self.nsr_model(model_in)
            pred_core, _ = self._nsr_unpack_outputs(model_out)
            if self.nsr_residual_from_base:
                prev_base = torch.where(
                    prev_in_valid > 0.5,
                    prev_in,
                    torch.full_like(prev_in, float(self.nsr_fill_value)),
                )
                base_h = torch.where(meas_m > 0.5, meas_h, prev_base)
                residual = pred_core
                if self.nsr_residual_tanh:
                    residual = self.nsr_residual_scale * torch.tanh(residual)
                else:
                    residual = self.nsr_residual_scale * residual
                pred_h = base_h + residual
            else:
                pred_h = pred_core
            pred_h = torch.nan_to_num(pred_h, nan=float(self.nsr_fill_value), posinf=1e3, neginf=-1e3)

        pred_for_obs = torch.where(meas_m > 0.5, meas_h, pred_h)
        if self.nsr_memory_meas_override:
            self.nsr_prev_pred = pred_for_obs.detach()
        else:
            self.nsr_prev_pred = pred_h.detach()
        self.nsr_prev_valid = torch.maximum(meas_m, prev_in_valid).detach()
        self.nsr_prev_pose7 = cur_pose7.detach().clone()

        sampled_local = self._sample_local_map_at_height_points(pred_for_obs)
        return self._local_height_to_world_height(sampled_local)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            if getattr(self, "nsr_enabled", False):
                self.measured_heights = self._get_heights_nsr()
            else:
                self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            if not self.cfg.terrain.curriculum:
                rows = torch.randint(0, self.terrain.cfg.num_rows, (len(env_ids),), device=self.device)
                cols = torch.randint(0, self.terrain.cfg.num_cols, (len(env_ids),), device=self.device)
                self.terrain_levels[env_ids] = rows
                self.terrain_types[env_ids] = cols
                self.env_origins[env_ids] = self.terrain_origins[rows, cols]
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terrains
        move_up_dist = float(getattr(self.cfg.terrain, "curriculum_move_up_dist", -1.0))
        if move_up_dist > 0.0:
            move_up = distance > move_up_dist
        else:
            move_up = distance > self.terrain.env_length / 2
        # robots that advanced too little go to simpler terrains
        move_down_dist = float(getattr(self.cfg.terrain, "curriculum_move_down_dist", -1.0))
        if move_down_dist > 0.0:
            move_down = (distance < move_down_dist) * ~move_up
        else:
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are recycled, optionally inside a hard-row band.
        recycle_min = int(getattr(self.cfg.terrain, "curriculum_recycle_min_level", 0))
        recycle_min = max(0, min(recycle_min, self.max_terrain_level - 1))
        recycle_span = self.max_terrain_level - recycle_min
        recycle_levels = torch.randint_like(self.terrain_levels[env_ids], recycle_span) + recycle_min
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   recycle_levels,
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            num_h = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
            noise_vec[48:48 + num_h] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        # Keep a dedicated state for dense clearance reward so it can be combined
        # with feet_air_time without double-updating shared buffers.
        self.feet_air_time_clearance = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts_clearance = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # depth images (warp camera)
        if self.cfg.sensor.add_depth:
            pointcloud_dims = 3 * (self.cfg.sensor.depth_camera_config.return_pointcloud == True)
            if pointcloud_dims > 0:
                self.depth_images = torch.zeros(
                    (
                        self.num_envs,
                        self.cfg.sensor.num_sensors,
                        self.cfg.sensor.depth_camera_config.num_history,
                        self.cfg.sensor.depth_camera_config.resolution[1],
                        self.cfg.sensor.depth_camera_config.resolution[0],
                        pointcloud_dims,
                    ),
                    device=self.device,
                    dtype=torch.float,
                )
            else:
                self.depth_images = torch.zeros(
                    (
                        self.num_envs,
                        self.cfg.sensor.num_sensors,
                        self.cfg.sensor.depth_camera_config.num_history,
                        self.cfg.sensor.depth_camera_config.resolution[1],
                        self.cfg.sensor.depth_camera_config.resolution[0],
                    ),
                    device=self.device,
                    dtype=torch.float,
                )
            self.depth_image_update_decimation = max(1, int(self.cfg.sensor.depth_camera_config.decimation))
            self.depth_image_update_counter = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        self.trot_contact_order = None
        if len(feet_names) == 4:
            def _find_foot_index(primary_tag, secondary_tags):
                for idx, foot_name in enumerate(feet_names):
                    up = foot_name.upper()
                    if primary_tag in up:
                        return idx
                for idx, foot_name in enumerate(feet_names):
                    up = foot_name.upper()
                    if all(tag in up for tag in secondary_tags):
                        return idx
                return None

            fl_idx = _find_foot_index("FL", ("FRONT", "LEFT"))
            fr_idx = _find_foot_index("FR", ("FRONT", "RIGHT"))
            rl_idx = _find_foot_index("RL", ("REAR", "LEFT"))
            rr_idx = _find_foot_index("RR", ("REAR", "RIGHT"))
            if None not in (fl_idx, fr_idx, rl_idx, rr_idx):
                self.trot_contact_order = torch.tensor(
                    [fl_idx, fr_idx, rl_idx, rr_idx],
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            min_init_level = int(getattr(self.cfg.terrain, "min_init_terrain_level", 0))
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                min_init_level = 0
                max_init_level = self.cfg.terrain.num_rows - 1
            min_init_level = max(0, min(min_init_level, self.cfg.terrain.num_rows - 1))
            max_init_level = max(0, min(int(max_init_level), self.cfg.terrain.num_rows - 1))
            if min_init_level > max_init_level:
                min_init_level, max_init_level = max_init_level, min_init_level
            self.terrain_levels = torch.randint(
                min_init_level,
                max_init_level + 1,
                (self.num_envs,),
                device=self.device,
            )
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        if bool(getattr(self.cfg.rewards, "base_height_use_terrain", False)) and self.cfg.terrain.measure_heights:
            measured_heights = self.measured_heights
            if not torch.is_tensor(measured_heights):
                measured_heights = self._get_heights_nsr() if getattr(self, "nsr_enabled", False) else self._get_heights()
                self.measured_heights = measured_heights
            base_height = base_height - torch.mean(measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_leg_motion_balance(self):
        # Penalize one leg becoming under-used compared to the others.
        # Assumes 12 actions arranged as 4 legs x 3 joints for quadrupeds.
        if self.actions.shape[1] != 12:
            return torch.zeros(self.num_envs, device=self.device, dtype=self.actions.dtype)
        leg_actions = self.actions.view(self.num_envs, 4, 3)
        leg_mag = torch.mean(torch.abs(leg_actions), dim=2)
        return torch.var(leg_mag, dim=1)

    def _reward_pronk(self):
        # Penalize fully-synchronous gait states (all feet down or all feet in air).
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        all_contact = torch.all(contact, dim=1).float()
        all_air = torch.all(~contact, dim=1).float()
        return all_contact + all_air

    def _reward_non_alternating(self):
        # Penalize non-alternating contact phases (pace/bound-like synchrony).
        # For quadrupeds with identifiable FL/FR/RL/RR feet, prefer trot-like phase:
        # FL with RR, FR with RL, and opposing diagonal groups out of phase.
        if self.trot_contact_order is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=self.actions.dtype)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        ordered = contact[:, self.trot_contact_order]
        fl, fr, rl, rr = ordered.unbind(dim=1)

        diag_mismatch = torch.logical_xor(fl, rr).float() + torch.logical_xor(fr, rl).float()
        same_pair = torch.logical_not(torch.logical_xor(fl, fr)).float() + torch.logical_not(torch.logical_xor(rl, rr)).float()
        penalty = 0.25 * (diag_mismatch + same_pair)

        cmd_thresh = float(getattr(self.cfg.rewards, "cmd_nonzero_thresh", 0.15))
        active = torch.norm(self.commands[:, :2], dim=1) > cmd_thresh
        return penalty * active.float()
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        target_air_time = float(getattr(self.cfg.rewards, "feet_air_time_target", 0.5))
        rew_airTime = torch.sum((self.feet_air_time - target_air_time) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_foot_clearance(self):
        # Dense proxy for clearance:
        # reward feet that stay in swing longer than target (instead of sparse first-contact only).
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts_clearance)
        self.last_contacts_clearance = contact
        self.feet_air_time_clearance += self.dt

        target_air_time = float(getattr(self.cfg.rewards, "feet_air_time_target", 0.2))
        cap_above_target = float(getattr(self.cfg.rewards, "foot_clearance_cap", 0.25))
        swing = ~contact_filt
        clearance_bonus = torch.clamp(self.feet_air_time_clearance - target_air_time, min=0.0, max=cap_above_target)
        rew_clearance = torch.sum(clearance_bonus * swing, dim=1)
        rew_clearance *= torch.norm(self.commands[:, :2], dim=1) > 0.1

        self.feet_air_time_clearance *= ~contact_filt
        return rew_clearance
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_stumble(self):
        # Backward-compatible alias for reward scale key "feet_stumble".
        return self._reward_stumble()
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_cmd_nonzero_but_slow(self):
        # Penalize "commanded to move but barely moving" behavior.
        cmd_norm = torch.norm(self.commands[:, :2], dim=1)
        vel_norm = torch.norm(self.base_lin_vel[:, :2], dim=1)
        cmd_thresh = float(getattr(self.cfg.rewards, "cmd_nonzero_thresh", 0.2))
        slow_thresh = float(getattr(self.cfg.rewards, "slow_speed_thresh", 0.2))
        active = cmd_norm > cmd_thresh
        shortfall = torch.clamp(slow_thresh - vel_norm, min=0.0)
        penalty = shortfall / max(slow_thresh, 1.0e-6)
        return penalty * active.float()

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
