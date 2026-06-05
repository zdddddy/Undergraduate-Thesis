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

from __future__ import annotations

import time
import os
from collections import deque
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import wandb
from datetime import datetime
import torch

from rsl_rl.algorithms import PPO_TS
from rsl_rl.modules import ActorCriticTS
from rsl_rl.env import VecEnv
from .on_policy_runner import OnPolicyRunner, TrainConfig


class TSRunner(OnPolicyRunner):
    """Teacher-Student runner for training with distillation."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: TrainConfig,
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
    
    def _init_agent_and_algo(self) -> None:
        """Initialize the TS actor-critic and PPO_TS algorithm."""
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCriticTS = actor_critic_class(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_privileged_obs,
            self.env.num_history_obs,  # type: ignore[attr-defined]
            self.env.num_latent_dims,  # type: ignore[attr-defined]
            self.env.num_critic_obs,  # type: ignore[attr-defined]
            **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_TS = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self._maybe_attach_distillation_teacher(actor_critic_class)

    def _maybe_attach_distillation_teacher(self, actor_critic_class) -> None:
        action_coef = max(
            float(self.alg_cfg.get("distill_action_coef", 0.0)),
            float(self.alg_cfg.get("distill_action_coef_final", 0.0)),
        )
        latent_coef = max(
            float(self.alg_cfg.get("distill_latent_coef", 0.0)),
            float(self.alg_cfg.get("distill_latent_coef_final", 0.0)),
        )
        requires_teacher = (action_coef > 0.0) or (latent_coef > 0.0)

        teacher_model_path = str(self.cfg.get("teacher_model_path", "")).strip()
        if not teacher_model_path:
            if requires_teacher:
                raise ValueError(
                    "Distillation is enabled (non-zero action/latent distill coeffs), "
                    "but runner.teacher_model_path is empty. "
                    "Please provide --teacher_model_path."
                )
            return

        candidates = [
            teacher_model_path,
            os.path.expanduser(teacher_model_path),
            os.path.join(os.getcwd(), teacher_model_path),
            os.path.join(os.getcwd(), "logs", teacher_model_path),
        ]
        teacher_ckpt_path = next((os.path.abspath(p) for p in candidates if os.path.isfile(p)), None)
        if teacher_ckpt_path is None:
            raise FileNotFoundError(
                f"teacher_model_path not found: {teacher_model_path}. "
                f"Tried: {candidates}"
            )

        loaded_dict = torch.load(teacher_ckpt_path, map_location=self.device, weights_only=False)
        teacher_state = loaded_dict["model_state_dict"] if isinstance(loaded_dict, dict) and "model_state_dict" in loaded_dict else loaded_dict

        teacher_ac: ActorCriticTS = actor_critic_class(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_privileged_obs,
            self.env.num_history_obs,  # type: ignore[attr-defined]
            self.env.num_latent_dims,  # type: ignore[attr-defined]
            self.env.num_critic_obs,  # type: ignore[attr-defined]
            **self.policy_cfg
        ).to(self.device)
        teacher_ac.load_state_dict(teacher_state, strict=False)
        self.alg.set_distillation_teacher(teacher_ac)
        print(f"[TSRunner] Attached distillation teacher: {teacher_ckpt_path}")
        
    def _init_storage(self) -> None:
        """Initialize the TS rollout storage."""
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env, 
            (self.env.num_obs,),
            (self.env.num_privileged_obs,), 
            (self.env.num_history_obs,),  # type: ignore[attr-defined]
            (self.env.num_critic_obs,),  # type: ignore[attr-defined]
            (self.env.num_actions,),
        )
    
    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run TS training loop for a specified number of iterations.

        Args:
            num_learning_iterations: Number of learning iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        self._pre_learn(init_at_random_ep_len)
        obs, privileged_obs, obs_history, critic_obs = self.env.get_observations()  # type: ignore[misc]
        obs, privileged_obs, obs_history, critic_obs = (
            obs.to(self.device),
            privileged_obs.to(self.device),
            obs_history.to(self.device),
            critic_obs.to(self.device),
        )
        self.alg.actor_critic.train()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if hasattr(self.env, "set_training_iteration"):
                self.env.set_training_iteration(it)

            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, obs_history, critic_obs)
                    obs, privileged_obs, obs_history, critic_obs, rewards, dones, infos = self.env.step(actions)  # type: ignore[misc]
                    obs, privileged_obs, obs_history, rewards, dones, critic_obs = (
                        obs.to(self.device),
                        privileged_obs.to(self.device),
                        obs_history.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        critic_obs.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            update_stats = self.alg.update()
            if len(update_stats) == 3:
                mean_value_loss, mean_surrogate_loss, mean_encoder_loss = update_stats
                mean_distill_loss = 0.0
                mean_action_distill_loss = 0.0
                mean_latent_distill_loss = 0.0
                mean_height_consistency_loss = 0.0
            else:
                mean_value_loss, mean_surrogate_loss, mean_encoder_loss, mean_distill_loss, mean_action_distill_loss, mean_latent_distill_loss, mean_height_consistency_loss = update_stats
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                assert self.log_dir is not None
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        assert self.log_dir is not None
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(
        self,
        locs: Dict[str, Any],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Log TS training metrics to tensorboard and console.

        Args:
            locs: Dictionary containing iteration metrics and buffers.
            width: Width of the log output.
            pad: Padding for log formatting.
        """
        assert self.writer is not None
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/encoder', locs['mean_encoder_loss'], locs['it'])
        self.writer.add_scalar('Loss/distill_total', locs['mean_distill_loss'], locs['it'])
        self.writer.add_scalar('Loss/distill_action', locs['mean_action_distill_loss'], locs['it'])
        self.writer.add_scalar('Loss/distill_latent', locs['mean_latent_distill_loss'], locs['it'])
        self.writer.add_scalar('Loss/height_consistency', locs['mean_height_consistency_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar("Loss/history_encoder_learning_rate", self.alg.encoder_lr, locs['it'])
        self.writer.add_scalar("Loss/distill_action_coef", self.alg.distill_action_coef_curr, locs['it'])
        self.writer.add_scalar("Loss/distill_latent_coef", self.alg.distill_latent_coef_curr, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str_iter = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Encoder loss:':>{pad}} {locs['mean_encoder_loss']:.4f}\n"""
                          f"""{'Distill total loss:':>{pad}} {locs['mean_distill_loss']:.4f}\n"""
                          f"""{'Distill action loss:':>{pad}} {locs['mean_action_distill_loss']:.4f}\n"""
                          f"""{'Distill latent loss:':>{pad}} {locs['mean_latent_distill_loss']:.4f}\n"""
                          f"""{'Height consistency loss:':>{pad}} {locs['mean_height_consistency_loss']:.4f}\n"""
                          f"""{'Distill action coef:':>{pad}} {self.alg.distill_action_coef_curr:.4f}\n"""
                          f"""{'Distill latent coef:':>{pad}} {self.alg.distill_latent_coef_curr:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Distill total loss:':>{pad}} {locs['mean_distill_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def get_deploy_inference_policy(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get the deploy policy (runtime terrain input branch) for inference."""
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_deploy

    def get_aux_inference_policy(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get the auxiliary history branch policy for inference/debug."""
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_aux

    def get_inference_policy(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Compatibility API: returns the auxiliary history branch policy."""
        return self.get_aux_inference_policy(device=device)

    def _warm_start_actor_from_checkpoint(self, model_state_dict: Dict[str, torch.Tensor]) -> bool:
        """Warm-start TS actor from a possibly non-TS checkpoint.

        Copies actor trunk weights when shapes are compatible and maps the
        first-layer proprioceptive channels (first `num_obs` dims) explicitly.
        Latent-related columns in the TS actor input are zero-initialized.
        """
        target_state = self.alg.actor_critic.state_dict()
        copied_keys: List[str] = []
        num_obs = int(self.env.num_obs)

        # Copy action std if available and shape matches.
        if "std" in model_state_dict and "std" in target_state and model_state_dict["std"].shape == target_state["std"].shape:
            target_state["std"] = model_state_dict["std"]
            copied_keys.append("std")

        # Special handling for actor first layer: copy proprio columns only.
        if "actor.0.weight" in model_state_dict and "actor.0.weight" in target_state:
            src_w0 = model_state_dict["actor.0.weight"]
            dst_w0 = target_state["actor.0.weight"].clone()
            if src_w0.ndim == 2 and dst_w0.ndim == 2 and src_w0.shape[0] == dst_w0.shape[0]:
                if src_w0.shape[1] >= num_obs and dst_w0.shape[1] >= num_obs:
                    dst_w0.zero_()
                    dst_w0[:, :num_obs] = src_w0[:, :num_obs]
                    target_state["actor.0.weight"] = dst_w0
                    copied_keys.append("actor.0.weight[:,:num_obs]")

        # Copy remaining actor layers when shapes match.
        for key in [
            "actor.0.bias",
            "actor.2.weight",
            "actor.2.bias",
            "actor.4.weight",
            "actor.4.bias",
            "actor.6.weight",
            "actor.6.bias",
        ]:
            if key in model_state_dict and key in target_state and model_state_dict[key].shape == target_state[key].shape:
                target_state[key] = model_state_dict[key]
                copied_keys.append(key)

        if not copied_keys:
            return False

        self.alg.actor_critic.load_state_dict(target_state, strict=False)
        print(f"[TSRunner] Warm-started actor from checkpoint. Copied keys: {copied_keys}")
        return True

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint with TS strict load first, then warm-start fallback."""
        loaded_dict = torch.load(path, weights_only=False)
        model_state_dict = loaded_dict["model_state_dict"]
        reset_optimizer_on_resume = bool(self.cfg.get("reset_optimizer_on_resume", False))
        reset_iteration_on_resume = bool(self.cfg.get("reset_iteration_on_resume", False))

        try:
            self.alg.actor_critic.load_state_dict(model_state_dict)
            if load_optimizer and (not reset_optimizer_on_resume) and "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            elif load_optimizer and reset_optimizer_on_resume:
                print("[TSRunner] reset_optimizer_on_resume=True, skip loading optimizer state.")

            if reset_iteration_on_resume:
                self.current_learning_iteration = 0
                print("[TSRunner] reset_iteration_on_resume=True, reset iteration to 0 for this stage.")
            else:
                self.current_learning_iteration = loaded_dict.get("iter", 0)
            return loaded_dict.get("infos")
        except RuntimeError as err:
            print(f"[TSRunner] Strict checkpoint load failed: {err}")
            if not self._warm_start_actor_from_checkpoint(model_state_dict):
                raise
            if load_optimizer:
                print("[TSRunner] Skipping optimizer state load due architecture mismatch.")
            # Warm-start initialization starts a new stage from iteration 0.
            self.current_learning_iteration = 0
            return loaded_dict.get("infos")
