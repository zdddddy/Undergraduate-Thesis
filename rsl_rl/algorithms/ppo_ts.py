from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO, DeviceType
from rsl_rl.modules import ActorCriticTS
from rsl_rl.storage import RolloutStorageTS


class PPO_TS(PPO):
    """PPO with teacher-student architecture.
    
    Refer to: https://github.com/Improbable-AI/rapid-locomotion-rl
    """
    
    actor_critic: ActorCriticTS
    storage: Optional[RolloutStorageTS]
    transition: RolloutStorageTS.Transition
    
    # TS-specific components
    rl_parameters: List[torch.nn.Parameter]
    history_encoder_optimizer: optim.Optimizer
    encoder_lr: float
    num_encoder_epochs: int
    teacher_actor_critic: Optional[ActorCriticTS]

    def __init__(
        self,
        actor_critic: ActorCriticTS,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: Optional[float] = 0.01,
        use_spo: bool = False,
        device: DeviceType = 'cpu',
        encoder_lr: float = 1e-3,
        num_encoder_epochs: int = 1,
        distill_action_coef: float = 0.0,
        distill_action_coef_final: float = 0.0,
        distill_latent_coef: float = 0.0,
        distill_latent_coef_final: float = 0.0,
        distill_height_coef: float = 0.0,
        distill_total_iters: int = 0,
        distill_terrain_dim: int = 0,
        privilege_encoder_freeze_iters: int = 0,
        privilege_encoder_grad_scale: float = 1.0,
        lr_decay_total_iters: int = 0,
    ) -> None:

        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            use_spo,
            device,
        )
        self.encoder_lr = encoder_lr
        self.num_encoder_epochs = num_encoder_epochs
        self.base_learning_rate = float(learning_rate)
        self.lr_decay_total_iters = int(lr_decay_total_iters)

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        # Only include parameter of actor, critic, privilege encoder and std
        self.rl_parameters = list(self.actor_critic.actor.parameters()) + \
                             list(self.actor_critic.critic.parameters()) + \
                             list(self.actor_critic.privilege_encoder.parameters()) + \
                             [self.actor_critic.std]
        self.optimizer = optim.Adam(self.rl_parameters, lr=learning_rate)
        self.history_encoder_optimizer = optim.Adam(
            self.actor_critic.history_encoder.parameters(), lr=encoder_lr
        )
        self.transition = RolloutStorageTS.Transition()

        # Optional distillation from a frozen GT teacher.
        self.teacher_actor_critic = None
        self.distill_action_coef = float(distill_action_coef)
        self.distill_action_coef_final = float(distill_action_coef_final)
        self.distill_latent_coef = float(distill_latent_coef)
        self.distill_latent_coef_final = float(distill_latent_coef_final)
        self.distill_height_coef = float(distill_height_coef)
        self.distill_total_iters = int(distill_total_iters)
        self.distill_terrain_dim = int(distill_terrain_dim)
        self._update_step = 0
        self.distill_action_coef_curr = self.distill_action_coef
        self.distill_latent_coef_curr = self.distill_latent_coef
        self._distill_warned = False
        self.privilege_encoder_freeze_iters = int(privilege_encoder_freeze_iters)
        self.privilege_encoder_grad_scale = float(privilege_encoder_grad_scale)
        self._privilege_encoder_trainable = True
        if self.privilege_encoder_freeze_iters > 0:
            self._set_privilege_encoder_trainable(False)

    def _set_privilege_encoder_trainable(self, trainable: bool) -> None:
        if self._privilege_encoder_trainable == trainable:
            return
        self._privilege_encoder_trainable = trainable
        for p in self.actor_critic.privilege_encoder.parameters():
            p.requires_grad_(trainable)

    def _apply_linear_lr_decay_if_needed(self) -> None:
        if self.schedule != "linear" or self.lr_decay_total_iters <= 0:
            return
        progress = min(max(float(self._update_step) / float(self.lr_decay_total_iters), 0.0), 1.0)
        self.learning_rate = max(1e-5, self.base_learning_rate * (1.0 - progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _apply_privilege_encoder_grad_scale(self) -> None:
        if self.privilege_encoder_grad_scale >= 0.999:
            return
        scale = max(self.privilege_encoder_grad_scale, 0.0)
        for p in self.actor_critic.privilege_encoder.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    def set_distillation_teacher(self, teacher_actor_critic: ActorCriticTS) -> None:
        """Attach a frozen GT teacher for stage3 distillation."""
        self.teacher_actor_critic = teacher_actor_critic.to(self.device)
        self.teacher_actor_critic.eval()
        for p in self.teacher_actor_critic.parameters():
            p.requires_grad_(False)

    def _scheduled_coef(self, start: float, end: float) -> float:
        if self.distill_total_iters <= 0:
            return float(end)
        progress = min(max(float(self._update_step) / float(self.distill_total_iters), 0.0), 1.0)
        return float(start + progress * (end - start))

    def init_storage(  # type: ignore[override]
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Tuple[int, ...],
        privileged_obs_shape: Tuple[int, ...],
        obs_history_shape: Tuple[int, ...],
        critic_obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
    ) -> None:
        self.storage = RolloutStorageTS(
            num_envs, num_transitions_per_env, actor_obs_shape,
            privileged_obs_shape, obs_history_shape, critic_obs_shape, 
            action_shape, self.device
        )

    def act(  # type: ignore[override]
        self, 
        obs: torch.Tensor, 
        privileged_obs: torch.Tensor, 
        obs_history: torch.Tensor, 
        critic_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute actions using teacher-student architecture.
        
        Args:
            obs: Actor observations. Shape: [num_envs, obs_dim]
            privileged_obs: Privileged observations. Shape: [num_envs, priv_dim]
            obs_history: Observation history for encoder. Shape: [num_envs, history_dim]
            critic_obs: Critic observations. Shape: [num_envs, critic_obs_dim]
            
        Returns:
            actions: Sampled actions. Shape: [num_envs, action_dim]
        """
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, privileged_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def update(self) -> Tuple[float, float, float, float, float, float, float]:  # type: ignore[override]
        """Update policy and history encoder.
        
        Returns:
            mean_value_loss: Average value function loss
            mean_surrogate_loss: Average surrogate loss
            mean_encoder_loss: Average encoder loss
        """
        assert self.storage is not None  # storage is initialized in init_storage()
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_encoder_loss = 0.0
        mean_distill_loss = 0.0
        mean_action_distill_loss = 0.0
        mean_latent_distill_loss = 0.0
        mean_height_consistency_loss = 0.0

        self._apply_linear_lr_decay_if_needed()
        self._set_privilege_encoder_trainable(self._update_step >= self.privilege_encoder_freeze_iters)

        self.distill_action_coef_curr = self._scheduled_coef(
            self.distill_action_coef, self.distill_action_coef_final
        )
        self.distill_latent_coef_curr = self._scheduled_coef(
            self.distill_latent_coef, self.distill_latent_coef_final
        )

        if (
            self.teacher_actor_critic is None
            and not self._distill_warned
            and (self.distill_action_coef_curr > 0.0 or self.distill_latent_coef_curr > 0.0)
        ):
            print("[PPO_TS] Distillation weights are non-zero but no teacher is attached. Skipping distillation.")
            self._distill_warned = True

        generator = self._get_data_generator()
        for obs_batch, privileged_obs_batch, obs_histories_batch, critic_obs_batch, terminated_batch, \
            actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            loss, surrogate_loss, value_loss, distill_loss, action_distill_loss, latent_distill_loss, height_consistency_loss = self._compute_rl_loss(
                obs_batch, privileged_obs_batch, obs_histories_batch, critic_obs_batch,
                actions_batch, target_values_batch, advantages_batch, returns_batch, 
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, 
                hid_states_batch, masks_batch
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self._apply_privilege_encoder_grad_scale()
            nn.utils.clip_grad_norm_(self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_distill_loss += distill_loss.item()
            mean_action_distill_loss += action_distill_loss.item()
            mean_latent_distill_loss += latent_distill_loss.item()
            mean_height_consistency_loss += height_consistency_loss.item()
        
        # encoder update
        generator = self._get_data_generator()
        for obs_batch, privileged_obs_batch, obs_histories_batch, critic_obs_batch, terminated_batch, \
            actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            
            # history encoder gradient step
            for _ in range(self.num_encoder_epochs):
                encoder_loss = self._compute_encoder_loss(
                    obs_histories_batch, privileged_obs_batch, terminated_batch
                )
                self.history_encoder_optimizer.zero_grad()
                encoder_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.history_encoder.parameters(), self.max_grad_norm
                )
                self.history_encoder_optimizer.step()
                mean_encoder_loss += encoder_loss.item()
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_encoder_loss /= (num_updates * self.num_encoder_epochs)
        mean_distill_loss /= num_updates
        mean_action_distill_loss /= num_updates
        mean_latent_distill_loss /= num_updates
        mean_height_consistency_loss /= num_updates
        self.storage.clear()
        self._update_step += 1

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_encoder_loss,
            mean_distill_loss,
            mean_action_distill_loss,
            mean_latent_distill_loss,
            mean_height_consistency_loss,
        )
    
    def _compute_rl_loss(  # type: ignore[override]
        self,
        obs_batch: torch.Tensor,
        privileged_obs_batch: torch.Tensor,
        obs_histories_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        hid_states_batch: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        masks_batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.actor_critic.act(
            obs_batch, privileged_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
        )
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy

        self._adjust_learning_rate(sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate_loss = self._compute_surrogate_loss(ratio, advantages_batch)

        # Value function loss
        value_loss = self._compute_value_function_loss(value_batch, returns_batch, target_values_batch)

        # Optional stage3 distillation losses:
        # 1) action mean distill
        # 2) terrain latent distill
        # 3) small terrain consistency regularization
        distill_action_loss = torch.zeros((), device=self.device)
        distill_latent_loss = torch.zeros((), device=self.device)
        height_consistency_loss = torch.zeros((), device=self.device)
        distill_loss = torch.zeros((), device=self.device)
        if (
            self.teacher_actor_critic is not None
            and self.distill_terrain_dim > 0
            and critic_obs_batch.shape[1] >= self.distill_terrain_dim
            and privileged_obs_batch.shape[1] == self.distill_terrain_dim
        ):
            teacher_terrain_obs = critic_obs_batch[:, -self.distill_terrain_dim :]
            with torch.no_grad():
                teacher_action_mean = self.teacher_actor_critic.act_teacher(obs_batch, teacher_terrain_obs)
                teacher_latent = self.teacher_actor_critic.privilege_encoder(teacher_terrain_obs)

            if self.distill_action_coef_curr > 0.0:
                distill_action_loss = nn.functional.mse_loss(mu_batch, teacher_action_mean)
            if self.distill_latent_coef_curr > 0.0:
                # Distill the deploy encoder target into the auxiliary history encoder.
                student_latent = self.actor_critic._encode_aux_latent(obs_histories_batch)
                distill_latent_loss = nn.functional.mse_loss(student_latent, teacher_latent)
            if self.distill_height_coef > 0.0:
                height_consistency_loss = nn.functional.smooth_l1_loss(
                    privileged_obs_batch, teacher_terrain_obs
                )

            distill_loss = (
                self.distill_action_coef_curr * distill_action_loss
                + self.distill_latent_coef_curr * distill_latent_loss
                + self.distill_height_coef * height_consistency_loss
            )

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
        loss = loss + distill_loss
                
        return (
            loss,
            surrogate_loss,
            value_loss,
            distill_loss,
            distill_action_loss,
            distill_latent_loss,
            height_consistency_loss,
        )
    
    def _compute_encoder_loss(
        self,
        obs_histories_batch: torch.Tensor,
        privileged_obs_batch: torch.Tensor,
        terminated_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute encoder loss for distilling privileged info."""
        if self.actor_critic.history_encoder_type == "TCN":
            if obs_histories_batch.dim() == 2:
                # input shape (batch_size, obs_history_len) -> (batch_size, 1, obs_history_len)
                obs_histories_batch = obs_histories_batch.unsqueeze(1)
        encoder_predictions = self.actor_critic.history_encoder(obs_histories_batch)
        
        with torch.no_grad():
            encoder_targets = self.actor_critic.privilege_encoder(privileged_obs_batch)

        encoder_loss = nn.functional.mse_loss(
            encoder_predictions * terminated_batch, encoder_targets * terminated_batch
        )
        
        return encoder_loss
