from legged_gym import *

import torch
import torch.nn.functional as F

from legged_gym.envs.go2.go2_ts.go2_ts import Go2TS


class Go2Stage2(Go2TS):
    """代码 stage2：对应论文越障阶段一，2c 加入 NSR 类地形退化。"""

    def __init__(self, cfg, sim_params, sim_device, headless):
        super().__init__(cfg, sim_params, sim_device, headless)
        self._init_terrain_degrade()

    def _init_terrain_degrade(self):
        cfg = getattr(self.cfg, "terrain_degrade", None)
        self._terrain_degrade_enabled = bool(cfg is not None and getattr(cfg, "enable", False))
        self._stage2_iter = 0
        self._terrain_last_valid_ratio = torch.ones(self.num_envs, device=self.device, dtype=torch.float)

        if not self._terrain_degrade_enabled:
            self._terrain_prev_actor_heights_m = None
            self._terrain_prev_actor_valid = None
            return

        self._terrain_degrade_cfg = cfg
        self._terrain_degrade_total_iters = max(int(getattr(cfg, "curriculum_total_iters", 2000)), 1)
        ratios = list(getattr(cfg, "phase_ratios", [0.2, 0.6, 0.2]))
        if len(ratios) != 3 or sum(ratios) <= 0:
            ratios = [0.2, 0.6, 0.2]
        s = float(sum(ratios))
        self._terrain_phase_ratios = [float(r) / s for r in ratios]

        self._terrain_dim = int(self.num_privileged_obs)
        self._terrain_grid_size = int(round(self._terrain_dim ** 0.5))
        if self._terrain_grid_size * self._terrain_grid_size != self._terrain_dim:
            raise ValueError(
                f"Stage2 terrain degrade expects square terrain dim, got {self._terrain_dim}."
            )
        self._terrain_prev_actor_heights_m = torch.zeros(
            self.num_envs, self._terrain_dim, device=self.device, dtype=torch.float
        )
        # 每个环境单独标记，避免 reset 后首帧延迟污染。
        self._terrain_prev_actor_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        print("[Go2Stage2] Terrain degradation enabled for stage2c.")

    def set_training_iteration(self, it: int):
        self._stage2_iter = max(int(it), 0)

    def _interp_three_phase(self, anchors, progress: float) -> float:
        if len(anchors) != 3:
            raise ValueError(f"Expected 3 anchors, got {anchors}")
        a0, a1, a2 = float(anchors[0]), float(anchors[1]), float(anchors[2])
        p0, p1, p2 = self._terrain_phase_ratios
        if progress <= p0:
            t = progress / max(p0, 1e-6)
            return a0 + t * (a1 - a0)
        if progress <= p0 + p1:
            return a1
        t = (progress - p0 - p1) / max(p2, 1e-6)
        return a1 + t * (a2 - a1)

    def _current_degrade_profile(self):
        p = min(max(float(self._stage2_iter) / float(self._terrain_degrade_total_iters), 0.0), 1.0)
        cfg = self._terrain_degrade_cfg
        return {
            "noise_std_m": self._interp_three_phase(cfg.noise_std_m, p),
            "point_dropout_prob": self._interp_three_phase(cfg.point_dropout_prob, p),
            "patch_dropout_prob": self._interp_three_phase(cfg.patch_dropout_prob, p),
            "blur_prob": self._interp_three_phase(cfg.blur_prob, p),
            "bias_prob": self._interp_three_phase(cfg.bias_prob, p),
            "bias_std_m": self._interp_three_phase(cfg.bias_std_m, p),
            "delay_prob": self._interp_three_phase(cfg.delay_prob, p),
            "delay_mix": self._interp_three_phase(cfg.delay_mix, p),
            "patch_half_span_min": int(getattr(cfg, "patch_half_span_min", 1)),
            "patch_half_span_max": int(getattr(cfg, "patch_half_span_max", 2)),
        }

    def _apply_terrain_degrade(self, heights_m: torch.Tensor) -> torch.Tensor:
        """对 actor 侧局部高度加入 stage2c 退化。"""
        if not self._terrain_degrade_enabled:
            self._terrain_last_valid_ratio = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
            return heights_m

        prof = self._current_degrade_profile()
        n, dim = heights_m.shape
        g = self._terrain_grid_size
        if dim != self._terrain_dim:
            return heights_m

        cur = heights_m.view(n, g, g)
        prev = self._terrain_prev_actor_heights_m.view(n, g, g)
        if self._terrain_prev_actor_valid is not None:
            prev_valid = self._terrain_prev_actor_valid.view(n, 1, 1)
            # reset 后首帧用当前高度作为延迟参考。
            prev = torch.where(prev_valid, prev, cur)

        noise_std_m = float(max(prof["noise_std_m"], 0.0))
        if noise_std_m > 0.0:
            cur = cur + noise_std_m * torch.randn_like(cur)

        bias_prob = float(max(prof["bias_prob"], 0.0))
        bias_std_m = float(max(prof["bias_std_m"], 0.0))
        if bias_prob > 0.0 and bias_std_m > 0.0:
            apply = (torch.rand(n, 1, 1, device=self.device) < bias_prob).to(cur.dtype)
            bias_low = bias_std_m * torch.randn(n, 1, 3, 3, device=self.device, dtype=cur.dtype)
            bias_map = F.interpolate(bias_low, size=(g, g), mode="bilinear", align_corners=False).squeeze(1)
            cur = cur + apply * bias_map

        blur_prob = float(max(prof["blur_prob"], 0.0))
        if blur_prob > 0.0:
            padded = F.pad(cur.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
            blurred = F.avg_pool2d(padded, kernel_size=3, stride=1).squeeze(1)
            apply = torch.rand(n, 1, 1, device=self.device) < blur_prob
            cur = torch.where(apply, blurred, cur)

        delay_prob = float(max(prof["delay_prob"], 0.0))
        delay_mix = float(min(max(prof["delay_mix"], 0.0), 0.95))
        delayed = (1.0 - delay_mix) * cur + delay_mix * prev
        if delay_prob > 0.0 and delay_mix > 0.0:
            apply = torch.rand(n, 1, 1, device=self.device) < delay_prob
            cur = torch.where(apply, delayed, cur)

        point_dropout_prob = float(min(max(prof["point_dropout_prob"], 0.0), 0.95))
        valid = torch.rand_like(cur) > point_dropout_prob

        patch_dropout_prob = float(min(max(prof["patch_dropout_prob"], 0.0), 0.95))
        if patch_dropout_prob > 0.0:
            env_apply = torch.rand(n, device=self.device) < patch_dropout_prob
            idx = torch.nonzero(env_apply, as_tuple=False).flatten()
            if idx.numel() > 0:
                m = idx.numel()
                hmin = max(int(prof["patch_half_span_min"]), 0)
                hmax = max(int(prof["patch_half_span_max"]), hmin)
                cx = torch.randint(0, g, (m, 1, 1), device=self.device)
                cy = torch.randint(0, g, (m, 1, 1), device=self.device)
                hs = torch.randint(hmin, hmax + 1, (m, 1, 1), device=self.device)
                rr = torch.arange(g, device=self.device).view(1, g, 1)
                cc = torch.arange(g, device=self.device).view(1, 1, g)
                patch = (rr - cx).abs() <= hs
                patch = patch & ((cc - cy).abs() <= hs)
                valid[idx] = valid[idx] & (~patch)

        cur = torch.where(valid, cur, delayed)
        cur = torch.clamp(cur, -1.0, 1.0)
        self._terrain_prev_actor_heights_m = cur.view(n, dim).detach()
        if self._terrain_prev_actor_valid is not None:
            self._terrain_prev_actor_valid[:] = True
        self._terrain_last_valid_ratio = valid.float().mean(dim=(1, 2)).detach()
        return cur.view(n, dim)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if not self._terrain_degrade_enabled or len(env_ids) == 0:
            return
        self._terrain_prev_actor_heights_m[env_ids] = 0.0
        if self._terrain_prev_actor_valid is not None:
            self._terrain_prev_actor_valid[env_ids] = False
        self._terrain_last_valid_ratio[env_ids] = 1.0

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

        heights_gt_m = torch.clip(
            self.simulator.base_pos[:, 2].unsqueeze(1) - 0.5 - self.simulator.measured_heights,
            -1.0,
            1.0,
        )
        heights_gt = heights_gt_m * self.obs_scales.height_measurements
        heights_actor_m = self._apply_terrain_degrade(heights_gt_m)
        heights_actor = heights_actor_m * self.obs_scales.height_measurements

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
            critic_obs = torch.cat((critic_obs, heights_gt), dim=-1)

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

        self.terrain_obs_buf = heights_actor
        self.terrain_obs_gt_buf = heights_gt
        # 兼容 TS：这里实际承载 actor 侧地形输入。
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = heights_actor
