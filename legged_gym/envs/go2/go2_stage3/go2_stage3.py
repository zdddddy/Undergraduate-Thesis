from __future__ import annotations

import os
import time
from typing import Tuple

import torch
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go2.go2_stage2.go2_stage2 import Go2Stage2
from legged_gym.utils.math_utils import quat_apply, quat_rotate_inverse


class Go2Stage3(Go2Stage2):
    """代码 stage3：对应论文越障阶段二，actor 侧地形改为在线 NSR 预测。"""

    def __init__(self, cfg, sim_params, sim_device, headless):
        super().__init__(cfg, sim_params, sim_device, headless)
        self._init_stage3_phase_state()
        self._init_nsr_runtime()
        self._init_stage3_metric_buffers()

    def _init_stage3_phase_state(self):
        self._stage3_phase = "hard"
        self._stage3_warmup_iters = int(getattr(self.cfg.stage3, "warmup_iters", 0))
        self._stage3_warmup_max_terrain_level = int(
            getattr(self.cfg.stage3, "warmup_max_terrain_level", 9)
        )
        self._stage3_warmup_disable_terrain_curriculum = bool(
            getattr(self.cfg.stage3, "warmup_disable_terrain_curriculum", False)
        )
        self._stage3_warmup_disable_command_curriculum = bool(
            getattr(self.cfg.stage3, "warmup_disable_command_curriculum", False)
        )
        self._stage3_warmup_disable_push_robots = bool(
            getattr(self.cfg.stage3, "warmup_disable_push_robots", False)
        )
        self._stage3_warmup_max_command_x = float(
            getattr(self.cfg.stage3, "warmup_max_command_x", self.command_ranges["lin_vel_x"][1])
        )
        self._stage3_warmup_max_command_y = float(
            getattr(self.cfg.stage3, "warmup_max_command_y", self.command_ranges["lin_vel_y"][1])
        )
        self._stage3_warmup_max_ang_vel_yaw = float(
            getattr(self.cfg.stage3, "warmup_max_ang_vel_yaw", self.command_ranges["ang_vel_yaw"][1])
        )
        self._stage3_warned_static_dr_runtime = False
        self._stage3_is_isaacgym = "isaacgym" in type(self.simulator).__name__.lower()

        # 保存目标分布，warmup 结束后恢复。
        self._stage3_hard_domain_rand = {
            "friction_range": list(self.cfg.domain_rand.friction_range),
            "added_mass_range": list(self.cfg.domain_rand.added_mass_range),
            "push_interval_s": self.cfg.domain_rand.push_interval_s,
            "max_push_vel_xy": self.cfg.domain_rand.max_push_vel_xy,
            "com_pos_x_range": list(self.cfg.domain_rand.com_pos_x_range),
            "com_pos_y_range": list(self.cfg.domain_rand.com_pos_y_range),
            "com_pos_z_range": list(self.cfg.domain_rand.com_pos_z_range),
            "kp_range": list(self.cfg.domain_rand.kp_range),
            "kd_range": list(self.cfg.domain_rand.kd_range),
        }
        self._stage3_hard_push_robots = bool(self.cfg.domain_rand.push_robots)
        self._stage3_hard_terrain_curriculum = bool(self.cfg.terrain.curriculum)
        self._stage3_hard_command_curriculum = bool(self.cfg.commands.curriculum)
        self._stage3_hard_command_lin_vel_x = list(self.command_ranges["lin_vel_x"])
        self._stage3_hard_command_lin_vel_y = list(self.command_ranges["lin_vel_y"])
        self._stage3_hard_command_ang_vel_yaw = list(self.command_ranges["ang_vel_yaw"])
        self._stage3_hard_cfg_ang_vel_yaw = list(self.cfg.commands.ranges.ang_vel_yaw)
        self._stage3_hard_max_terrain_level = int(
            getattr(self.simulator, "_max_terrain_level", self.cfg.terrain.num_rows)
        )

        if self._stage3_warmup_iters > 0:
            self._apply_stage3_phase("warmup")
        else:
            self._apply_stage3_phase("hard")

    def _init_stage3_metric_buffers(self):
        self._track_runtime_metrics = bool(getattr(self.cfg.stage3, "track_runtime_metrics", True))
        self._nsr_valid_ratio_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._nsr_runtime_ms_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._nsr_metric_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._nsr_last_valid_ratio = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self._nsr_last_runtime_ms = 0.0

    def set_training_iteration(self, it: int):
        if self._stage3_phase == "warmup" and it >= self._stage3_warmup_iters:
            self._apply_stage3_phase("hard")

    def _apply_stage3_phase(self, phase: str):
        if phase == self._stage3_phase:
            return
        self._stage3_phase = phase
        if phase == "warmup":
            # IsaacGym 中摩擦/质量/COM 创建后不能实时改，只调整可运行时生效的项。
            if not self._stage3_is_isaacgym:
                self.cfg.domain_rand.friction_range = list(self.cfg.stage3.warmup_friction_range)
                self.cfg.domain_rand.added_mass_range = list(self.cfg.stage3.warmup_added_mass_range)
                self.cfg.domain_rand.com_pos_x_range = list(self.cfg.stage3.warmup_com_pos_x_range)
                self.cfg.domain_rand.com_pos_y_range = list(self.cfg.stage3.warmup_com_pos_y_range)
                self.cfg.domain_rand.com_pos_z_range = list(self.cfg.stage3.warmup_com_pos_z_range)
            elif not self._stage3_warned_static_dr_runtime:
                print(
                    "[Go2Stage3] Warmup DR note: in IsaacGym, friction/mass/COM are static after env creation; "
                    "warmup runtime controls mainly affect command ranges, push, terrain curriculum, and PD gain."
                )
                self._stage3_warned_static_dr_runtime = True

            self.cfg.domain_rand.push_interval_s = self.cfg.stage3.warmup_push_interval_s
            self.cfg.domain_rand.max_push_vel_xy = self.cfg.stage3.warmup_max_push_vel_xy
            self.cfg.domain_rand.kp_range = list(self.cfg.stage3.warmup_kp_range)
            self.cfg.domain_rand.kd_range = list(self.cfg.stage3.warmup_kd_range)
            # 切换阶段时同步 push_interval。
            self.cfg.domain_rand.push_interval = int(
                torch.ceil(torch.tensor(self.cfg.domain_rand.push_interval_s / self.dt)).item()
            )
            if self._stage3_warmup_disable_push_robots:
                self.cfg.domain_rand.push_robots = False
                if hasattr(self.simulator, "_rand_push_vels"):
                    self.simulator._rand_push_vels[:, :2] = 0.0

            if self._stage3_warmup_disable_terrain_curriculum:
                self.cfg.terrain.curriculum = False
            if self._stage3_warmup_disable_command_curriculum:
                self.cfg.commands.curriculum = False
                warmup_cmd_x = abs(self._stage3_warmup_max_command_x)
                self.command_ranges["lin_vel_x"][0] = -warmup_cmd_x
                self.command_ranges["lin_vel_x"][1] = warmup_cmd_x
                warmup_cmd_y = abs(self._stage3_warmup_max_command_y)
                self.command_ranges["lin_vel_y"][0] = -warmup_cmd_y
                self.command_ranges["lin_vel_y"][1] = warmup_cmd_y
                warmup_cmd_yaw = abs(self._stage3_warmup_max_ang_vel_yaw)
                self.command_ranges["ang_vel_yaw"][0] = -warmup_cmd_yaw
                self.command_ranges["ang_vel_yaw"][1] = warmup_cmd_yaw
                # heading 模式也要同步 yaw 限幅。
                self.cfg.commands.ranges.ang_vel_yaw = [-warmup_cmd_yaw, warmup_cmd_yaw]

            if hasattr(self.simulator, "_max_terrain_level"):
                self.simulator._max_terrain_level = max(
                    1, min(self._stage3_warmup_max_terrain_level, self._stage3_hard_max_terrain_level)
                )
            if hasattr(self.simulator, "_terrain_levels"):
                self.simulator._terrain_levels.clamp_(max=self.simulator._max_terrain_level - 1)
            print(f"[Go2Stage3] Enter warmup phase (<= iter {self._stage3_warmup_iters}).")
        else:
            for k, v in self._stage3_hard_domain_rand.items():
                setattr(self.cfg.domain_rand, k, list(v) if isinstance(v, list) else v)
            self.cfg.domain_rand.push_robots = self._stage3_hard_push_robots
            self.cfg.terrain.curriculum = self._stage3_hard_terrain_curriculum
            self.cfg.commands.curriculum = self._stage3_hard_command_curriculum
            self.command_ranges["lin_vel_x"][0] = self._stage3_hard_command_lin_vel_x[0]
            self.command_ranges["lin_vel_x"][1] = self._stage3_hard_command_lin_vel_x[1]
            self.command_ranges["lin_vel_y"][0] = self._stage3_hard_command_lin_vel_y[0]
            self.command_ranges["lin_vel_y"][1] = self._stage3_hard_command_lin_vel_y[1]
            self.command_ranges["ang_vel_yaw"][0] = self._stage3_hard_command_ang_vel_yaw[0]
            self.command_ranges["ang_vel_yaw"][1] = self._stage3_hard_command_ang_vel_yaw[1]
            self.cfg.commands.ranges.ang_vel_yaw = list(self._stage3_hard_cfg_ang_vel_yaw)
            self.cfg.domain_rand.push_interval = int(
                torch.ceil(torch.tensor(self.cfg.domain_rand.push_interval_s / self.dt)).item()
            )
            if hasattr(self.simulator, "_max_terrain_level"):
                self.simulator._max_terrain_level = self._stage3_hard_max_terrain_level
            print("[Go2Stage3] Enter hard phase (cfg-defined target distribution).")

    # -------- NSR 在线推理 --------
    def _get_base_pose7(self) -> torch.Tensor:
        return torch.cat((self.simulator.base_pos, self.simulator.base_quat), dim=-1)

    def _nsr_quat_to_rotmat_xyzw_torch(self, q: torch.Tensor) -> torch.Tensor:
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

    def _nsr_quat_to_yaw_rotmat_xyzw_torch(self, q: torch.Tensor) -> torch.Tensor:
        n = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)
        qn = q / n
        x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        zeros = torch.zeros_like(cy)
        ones = torch.ones_like(cy)
        row0 = torch.stack([cy, -sy, zeros], dim=1)
        row1 = torch.stack([sy, cy, zeros], dim=1)
        row2 = torch.stack([zeros, zeros, ones], dim=1)
        return torch.stack([row0, row1, row2], dim=1)

    def _nsr_warp_prev_to_current(
        self,
        prev_map: torch.Tensor,
        prev_pose7: torch.Tensor,
        cur_pose7: torch.Tensor,
        map_size: float,
        prev_valid: torch.Tensor | None = None,
        gravity_aligned: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            prev_map, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        warped_valid = F.grid_sample(
            prev_valid, grid, mode="bilinear", padding_mode="zeros", align_corners=True
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
            "bij,bhwj->bhwi", rot_cur.transpose(1, 2), p_world_obj - pos_cur[:, None, None, :]
        )
        warped_h_cur = p_cur_obj[..., 2].unsqueeze(1)
        warped_h_cur = torch.where(warped_valid > valid_eps, warped_h_cur, torch.zeros_like(warped_h_cur))
        return warped_h_cur, warped_valid

    def _nsr_build_model_input(
        self, meas_h: torch.Tensor, meas_m: torch.Tensor, prev_h: torch.Tensor, prev_valid: torch.Tensor, in_channels: int
    ) -> torch.Tensor:
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

    def _nsr_binarize_valid(self, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        if threshold <= 0.0:
            return mask.clamp(0.0, 1.0)
        if threshold >= 1.0:
            return (mask >= 1.0).float()
        return (mask > float(threshold)).float()

    def _init_nsr_runtime(self):
        self.nsr_enabled = bool(getattr(getattr(self.cfg, "nsr", None), "enable", False))
        if not self.nsr_enabled:
            self.nsr_model = None
            return

        if not self.cfg.terrain.measure_heights:
            raise ValueError("NSR runtime requires terrain.measure_heights=True.")
        if not self.cfg.sensor.use_warp:
            raise ValueError("NSR runtime requires sensor.use_warp=True.")
        if not self.cfg.sensor.add_depth:
            raise ValueError("NSR runtime requires sensor.add_depth=True.")
        if not self.cfg.sensor.depth_camera_config.return_pointcloud:
            raise ValueError("NSR runtime requires return_pointcloud=True.")
        if not self.cfg.sensor.depth_camera_config.pointcloud_in_world_frame:
            raise ValueError("NSR runtime requires pointcloud_in_world_frame=True.")

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
        self.nsr_group_norm_groups = int(
            train_args.get("group_norm_groups", getattr(nsr_cfg, "group_norm_groups", 8))
        )
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
            raise ValueError(
                f"Invalid NSR map shape: map_size={self.nsr_map_size}, resolution={self.nsr_resolution}"
            )

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
        for p in self.nsr_model.parameters():
            p.requires_grad_(False)

        gs = self.nsr_grid_size
        self.nsr_prev_pred = torch.full(
            (self.num_envs, 1, gs, gs),
            float(self.nsr_fill_value),
            device=self.device,
            dtype=torch.float32,
        )
        self.nsr_prev_valid = torch.zeros(
            (self.num_envs, 1, gs, gs), device=self.device, dtype=torch.float32
        )
        self.nsr_prev_pose7 = self._get_base_pose7().clone().detach()

        if not hasattr(self.simulator, "_height_points"):
            raise AttributeError("Simulator missing _height_points required by NSR stage3.")
        gx = self.simulator._height_points[0, :, 0]
        gy = self.simulator._height_points[0, :, 1]
        half = 0.5 * self.nsr_map_size
        i = (gx + half) / self.nsr_resolution - 0.5
        j = (gy + half) / self.nsr_resolution - 0.5
        grid_x = 2.0 * (j / max(gs - 1, 1)) - 1.0
        grid_y = 2.0 * (i / max(gs - 1, 1)) - 1.0
        self.nsr_sample_grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, -1, 2).to(self.device)

        print(f"[Go2Stage3] NSR enabled. ckpt={ckpt_path}, grid={gs}x{gs}, in_channels={self.nsr_in_channels}")

    def _reset_nsr_state(self, env_ids: torch.Tensor):
        if not self.nsr_enabled or len(env_ids) == 0:
            return
        self.nsr_prev_pred[env_ids] = float(self.nsr_fill_value)
        self.nsr_prev_valid[env_ids] = 0.0
        self.nsr_prev_pose7[env_ids] = self._get_base_pose7()[env_ids]

    def _build_local_measurement_heightmaps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """将多相机点云栅格化为 NSR 输入高度图和可见 mask。"""
        points_world = self.simulator.depth_images
        if points_world.dim() == 4:
            points_world = points_world.unsqueeze(1)
        if points_world.dim() != 5 or points_world.shape[-1] != 3:
            raise ValueError(
                f"NSR expects pointcloud shape [N,S,H,W,3], got {tuple(points_world.shape)}"
            )
        num_envs, num_sensors, h, w, _ = points_world.shape
        points = points_world.view(num_envs, num_sensors, h * w, 3)

        cam_pos = getattr(self.simulator, "_sensor_pos_tensor", None)
        if cam_pos is None:
            raise ValueError("NSR requires simulator._sensor_pos_tensor from warp camera runtime.")
        if cam_pos.dim() == 2:
            cam_pos = cam_pos.unsqueeze(1)
        if cam_pos.shape[1] != num_sensors:
            raise ValueError(
                f"sensor pose count mismatch for NSR: points sensors={num_sensors}, poses={cam_pos.shape[1]}"
            )

        cam_pos = cam_pos[:, :, None, :]
        dist = torch.norm(points - cam_pos, dim=-1)
        valid = torch.isfinite(points).all(dim=-1) & (dist < (self.nsr_far_plane - 1e-3))

        points = points.view(num_envs, num_sensors * h * w, 3)
        valid = valid.view(num_envs, num_sensors * h * w)
        rel = points - self.simulator.base_pos[:, None, :]

        if self.nsr_gravity_aligned:
            x = self.simulator.base_quat[:, 0]
            y = self.simulator.base_quat[:, 1]
            z = self.simulator.base_quat[:, 2]
            wq = self.simulator.base_quat[:, 3]
            yaw = torch.atan2(2.0 * (wq * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            cy = torch.cos(yaw)[:, None]
            sy = torch.sin(yaw)[:, None]
            lx = cy * rel[..., 0] + sy * rel[..., 1]
            ly = -sy * rel[..., 0] + cy * rel[..., 1]
            lz = rel[..., 2]
        else:
            rel_flat = rel.reshape(-1, 3)
            quat_rep = self.simulator.base_quat[:, None, :].expand(-1, rel.shape[1], -1).reshape(-1, 4)
            local_flat = quat_rotate_inverse(quat_rep, rel_flat)
            local = local_flat.view(num_envs, -1, 3)
            lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]

        gs = self.nsr_grid_size
        half = 0.5 * self.nsr_map_size
        ix = torch.floor((lx + half) / self.nsr_resolution).to(torch.long)
        iy = torch.floor((ly + half) / self.nsr_resolution).to(torch.long)
        inside = valid & (ix >= 0) & (ix < gs) & (iy >= 0) & (iy < gs)

        meas_h = torch.full(
            (num_envs, 1, gs, gs), float(self.nsr_fill_value), device=self.device, dtype=torch.float32
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

    def _sample_local_map_at_height_points(self, local_map: torch.Tensor) -> torch.Tensor:
        sample_grid = self.nsr_sample_grid.expand(local_map.shape[0], -1, -1, -1)
        sampled = F.grid_sample(
            local_map, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled[:, 0, 0, :]

    def _local_height_to_world_height(self, local_h: torch.Tensor) -> torch.Tensor:
        if self.nsr_gravity_aligned:
            return local_h + self.simulator.base_pos[:, 2:3]

        pts_local = self.simulator._height_points.clone()
        pts_local[:, :, 2] = local_h
        pcount = pts_local.shape[1]
        pts_flat = pts_local.reshape(-1, 3)
        quat_rep = self.simulator.base_quat[:, None, :].expand(-1, pcount, -1).reshape(-1, 4)
        world_flat = quat_apply(quat_rep, pts_flat)
        world = world_flat.view(self.num_envs, pcount, 3) + self.simulator.base_pos[:, None, :]
        return world[:, :, 2]

    def _get_heights_nsr(self) -> torch.Tensor:
        """运行 NSR 递推，并采样回策略使用的 9x9 高度点。"""
        meas_h, meas_m = self._build_local_measurement_heightmaps()
        self._nsr_last_valid_ratio = meas_m.mean(dim=(1, 2, 3))
        cur_pose7 = self._get_base_pose7()

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

    # -------- 环境接口 --------
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        if self.nsr_enabled and self._track_runtime_metrics:
            steps = self._nsr_metric_steps[env_ids].clamp_min(1.0)
            valid = torch.mean(self._nsr_valid_ratio_sum[env_ids] / steps)
            runtime_ms = torch.mean(self._nsr_runtime_ms_sum[env_ids] / steps)
            self.extras["episode"]["nsr_valid_ratio"] = valid
            self.extras["episode"]["nsr_runtime_ms"] = runtime_ms

        if self.nsr_enabled:
            self._reset_nsr_state(env_ids)

        if self._track_runtime_metrics:
            self._nsr_valid_ratio_sum[env_ids] = 0.0
            self._nsr_runtime_ms_sum[env_ids] = 0.0
            self._nsr_metric_steps[env_ids] = 0.0

    def compute_observations(self):
        self.obs_buf = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,  # 3
                self.simulator.projected_gravity,  # 3
                self.simulator.base_ang_vel * self.obs_scales.ang_vel,  # 3
                (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,  # 12
                self.simulator.dof_vel * self.obs_scales.dof_vel,  # 12
                self.actions,  # 12
            ),
            dim=-1,
        )

        domain_randomization_info = torch.cat(
            (
                self.simulator._friction_values,  # 1
                self.simulator._added_base_mass,  # 1
                self.simulator._base_com_bias,  # 3
                self.simulator._rand_push_vels[:, :2],  # 2
                self.simulator._kp_scale,  # 12
                self.simulator._kd_scale,  # 12
            ),
            dim=-1,
        )

        # critic 使用真值地形。
        terrain_obs_gt = torch.clip(
            self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 - self.simulator.measured_heights,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements

        if self.nsr_enabled:
            t0 = time.perf_counter()
            nsr_heights_world = self._get_heights_nsr()
            self._nsr_last_runtime_ms = (time.perf_counter() - t0) * 1000.0
            terrain_obs_actor = torch.clip(
                self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 - nsr_heights_world,
                -1.0,
                1.0,
            ) * self.obs_scales.height_measurements
        else:
            self._nsr_last_runtime_ms = 0.0
            self._nsr_last_valid_ratio = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
            terrain_obs_actor = terrain_obs_gt

        # actor 侧地形输入不再按 privileged 语义命名。
        self.terrain_obs_buf = terrain_obs_actor
        self.terrain_obs_gt_buf = terrain_obs_gt

        # 非对称 critic 单帧: 45 + 31 + 3 + 17 + 81
        critic_obs = torch.cat(
            (
                self.obs_buf,
                domain_randomization_info,
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            ),
            dim=-1,
        )
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat((critic_obs, self.simulator.link_contact_states), dim=-1)
        if self.cfg.terrain.measure_heights:
            # critic 保留真值地形以稳定训练。
            critic_obs = torch.cat((critic_obs, terrain_obs_gt), dim=-1)

        self.critic_obs_deque.append(critic_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i] for i in range(self.critic_obs_deque.maxlen)], dim=-1
        )

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i] for i in range(self.obs_history_deque.maxlen)], dim=-1
        )

        # 兼容当前 TS runner。
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = self.terrain_obs_buf

        if self.nsr_enabled and self._track_runtime_metrics:
            self._nsr_valid_ratio_sum += self._nsr_last_valid_ratio
            self._nsr_runtime_ms_sum += self._nsr_last_runtime_ms
            self._nsr_metric_steps += 1.0
