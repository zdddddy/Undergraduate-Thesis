from legged_gym import *

import torch

from legged_gym.envs.go2.go2 import GO2
from legged_gym.utils.math_utils import torch_rand_float


class GO2Stage1(GO2):
    """Two-phase stage1 locomotion base environment for Go2.

    Keeps deployment-style 45-dim proprioceptive observations (same as GO2),
    while optionally adding WTW-inspired gait/orientation/clearance priors via
    reward terms and internal behavior parameter sampling.
    """

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)

        # Periodic gait framework (WTW-inspired).
        self.a_swing = 0.0
        self.b_swing = self.cfg.rewards.periodic_reward_framework.b_swing * 2.0 * torch.pi
        self.b_stance = 2.0 * torch.pi

        # Behavior parameter ranges (sampled periodically).
        self.gait_period_min = self.cfg.rewards.behavior_params_range.gait_period_range[0]
        self.gait_period_max = self.cfg.rewards.behavior_params_range.gait_period_range[1]
        self.base_height_target_min = self.cfg.rewards.behavior_params_range.base_height_target_range[0]
        self.base_height_target_max = self.cfg.rewards.behavior_params_range.base_height_target_range[1]
        self.foot_clearance_target_min = self.cfg.rewards.behavior_params_range.foot_clearance_target_range[0]
        self.foot_clearance_target_max = self.cfg.rewards.behavior_params_range.foot_clearance_target_range[1]
        self.pitch_target_min = self.cfg.rewards.behavior_params_range.pitch_target_range[0]
        self.pitch_target_max = self.cfg.rewards.behavior_params_range.pitch_target_range[1]

        self.gait_period_range = [self.gait_period_min, self.gait_period_max]
        self.base_height_target_range = [self.base_height_target_min, self.base_height_target_max]
        self.foot_clearance_target_range = [self.foot_clearance_target_min, self.foot_clearance_target_max]
        self.pitch_target_range = [self.pitch_target_min, self.pitch_target_max]

        theta_fl = self.cfg.rewards.periodic_reward_framework.theta_fl_list
        theta_fr = self.cfg.rewards.periodic_reward_framework.theta_fr_list
        theta_rl = self.cfg.rewards.periodic_reward_framework.theta_rl_list
        theta_rr = self.cfg.rewards.periodic_reward_framework.theta_rr_list
        assert len(theta_fl) > 0, "periodic_reward_framework.theta_fl_list cannot be empty."
        assert len(theta_fl) == len(theta_fr) == len(theta_rl) == len(theta_rr), (
            "theta_*_list lengths must be identical for periodic gait reward."
        )
        self.num_gaits = len(theta_fl)

    def _init_buffers(self):
        super()._init_buffers()

        # Gait phase buffers.
        self.gait_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.phi = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        # Behavior params used in tracking-style priors.
        self.gait_period = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.base_height_target = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.foot_clearance_target = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.pitch_target = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)

        # Per-leg gait offsets (FL, FR, RL, RR).
        self.theta = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.theta_fl_candidates = torch.tensor(
            self.cfg.rewards.periodic_reward_framework.theta_fl_list,
            dtype=torch.float,
            device=self.device,
        )
        self.theta_fr_candidates = torch.tensor(
            self.cfg.rewards.periodic_reward_framework.theta_fr_list,
            dtype=torch.float,
            device=self.device,
        )
        self.theta_rl_candidates = torch.tensor(
            self.cfg.rewards.periodic_reward_framework.theta_rl_list,
            dtype=torch.float,
            device=self.device,
        )
        self.theta_rr_candidates = torch.tensor(
            self.cfg.rewards.periodic_reward_framework.theta_rr_list,
            dtype=torch.float,
            device=self.device,
        )

        # Initialize all envs with sampled behavior parameters.
        self._resample_behavior_params(torch.arange(self.num_envs, device=self.device))

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()

        # Update cyclic gait phase every step.
        self.gait_time += self.dt
        is_over_limit = self.gait_time >= (self.gait_period - self.dt / 2.0)
        self.gait_time[is_over_limit] = 0.0
        self.phi = self.gait_time / torch.clamp(self.gait_period, min=1e-3)

        # Resample behavior params periodically.
        resample_interval = int(self.cfg.rewards.behavior_params_range.resampling_time / self.dt)
        if resample_interval > 0:
            env_ids = (self.episode_length_buf % resample_interval == 0).nonzero(as_tuple=False).flatten()
            self._resample_behavior_params(env_ids)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_idx(env_ids)
        self.gait_time[env_ids] = 0.0
        self.phi[env_ids] = 0.0
        self._resample_behavior_params(env_ids)

        self.extras["episode"]["gait_period_max"] = self.gait_period_range[1]
        self.extras["episode"]["base_height_target_max"] = self.base_height_target_range[1]
        self.extras["episode"]["foot_clearance_target_max"] = self.foot_clearance_target_range[1]
        self.extras["episode"]["pitch_target_max"] = self.pitch_target_range[1]

    def _resample_behavior_params(self, env_ids):
        if len(env_ids) == 0:
            return

        self.gait_period[env_ids, :] = torch_rand_float(
            self.gait_period_range[0],
            self.gait_period_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        self.base_height_target[env_ids, :] = torch_rand_float(
            self.base_height_target_range[0],
            self.base_height_target_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        self.foot_clearance_target[env_ids, :] = torch_rand_float(
            self.foot_clearance_target_range[0],
            self.foot_clearance_target_range[1],
            (len(env_ids), 1),
            device=self.device,
        )
        self.pitch_target[env_ids, :] = torch_rand_float(
            self.pitch_target_range[0],
            self.pitch_target_range[1],
            (len(env_ids), 1),
            device=self.device,
        )

        gait_ids = torch.randint(0, self.num_gaits, (len(env_ids),), device=self.device)
        self.theta[env_ids, 0] = self.theta_fl_candidates[gait_ids]
        self.theta[env_ids, 1] = self.theta_fr_candidates[gait_ids]
        self.theta[env_ids, 2] = self.theta_rl_candidates[gait_ids]
        self.theta[env_ids, 3] = self.theta_rr_candidates[gait_ids]

    def _uniped_periodic_gait(self, foot_idx):
        q_frc = self.feet_force_norm[:, foot_idx].view(-1, 1)
        q_spd = torch.norm(self.simulator.feet_vel[:, foot_idx, :], dim=-1).view(-1, 1)

        phi = (self.phi + self.theta[:, foot_idx].unsqueeze(1)) % 1.0
        phi = phi * 2.0 * torch.pi

        exp_c_frc = torch.zeros_like(phi)
        exp_c_spd = torch.zeros_like(phi)
        is_swing = (phi >= self.a_swing) & (phi < self.b_swing)
        is_stance = (phi >= self.b_swing) & (phi < self.b_stance)
        exp_c_frc[is_swing] = -1.0
        exp_c_spd[is_swing] = 0.0
        exp_c_frc[is_stance] = 0.0
        exp_c_spd[is_stance] = -1.0

        return exp_c_spd * q_spd + exp_c_frc * q_frc

    def _reward_quad_periodic_gait(self):
        quad_reward = (
            self._uniped_periodic_gait(0)
            + self._uniped_periodic_gait(1)
            + self._uniped_periodic_gait(2)
            + self._uniped_periodic_gait(3)
        )
        return torch.exp(quad_reward.flatten())

    def _reward_tracking_base_height(self):
        base_height = torch.mean(
            self.simulator.base_pos[:, 2].unsqueeze(1) - self.simulator.measured_heights,
            dim=1,
        )
        error = torch.square(base_height - self.base_height_target.squeeze(1))
        return torch.exp(-error / self.cfg.rewards.base_height_tracking_sigma)

    def _reward_tracking_orientation(self):
        roll_error = torch.square(self.simulator.base_euler[:, 0])
        pitch_error = torch.square(self.simulator.base_euler[:, 1] - self.pitch_target.squeeze(1))
        return torch.exp(-(roll_error + pitch_error) / self.cfg.rewards.euler_tracking_sigma)

    def _reward_tracking_foot_clearance(self):
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm
            * torch.square(
                self.simulator.feet_pos[:, :, 2]
                - self.foot_clearance_target
                - self.cfg.rewards.foot_height_offset
            ),
            dim=-1,
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)

    def _reward_hip_pos(self):
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(
            torch.square(
                self.simulator.dof_pos[:, hip_joint_indices]
                - self.simulator.default_dof_pos[:, hip_joint_indices]
            ),
            dim=-1,
        )
        return dof_pos_error

    def _reward_thigh_pos(self):
        thigh_joint_indices = [1, 4, 7, 10]
        dof_pos_error = torch.sum(
            torch.square(
                self.simulator.dof_pos[:, thigh_joint_indices]
                - self.simulator.default_dof_pos[:, thigh_joint_indices]
            ),
            dim=-1,
        )
        return dof_pos_error
