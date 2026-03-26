# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import copy
import json
import os
import time
from datetime import datetime

import isaacgym  # noqa: F401
from isaacgym import gymutil
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry

import numpy as np
import torch


def get_eval_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "rough_go2_stage1_blind_hardened", "help": "Task name."},
        {"name": "--resume", "action": "store_true", "default": True, "help": "Load policy checkpoint."},
        {"name": "--experiment_name", "type": str, "help": "Experiment name override."},
        {"name": "--run_name", "type": str, "help": "Run name override."},
        {"name": "--load_run", "type": str, "help": "Run to load when resume=True. -1 loads last run."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint number. -1 loads last checkpoint."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Disable viewer."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device."},
        {"name": "--num_envs", "type": int, "default": 512, "help": "Number of environments."},
        {"name": "--seed", "type": int, "help": "Random seed override."},
        {
            "name": "--max_iterations",
            "type": int,
            "help": "Compatibility arg for shared cfg updater (unused in eval).",
        },
        {
            "name": "--keep_height_measurements",
            "action": "store_true",
            "default": False,
            "help": "Keep terrain height observations enabled in all scenarios (useful for stage-2 235-dim policies).",
        },

        # Nominal tracking scenario
        {"name": "--cruise_speed_mps", "type": float, "default": 0.7, "help": "Nominal commanded forward speed."},
        {"name": "--nominal_target_episodes", "type": int, "default": 800},
        {"name": "--nominal_max_steps", "type": int, "default": 160000},

        # Push recovery scenario
        {"name": "--push_target_episodes", "type": int, "default": 800},
        {"name": "--push_max_steps", "type": int, "default": 180000},
        {"name": "--push_interval_s", "type": float, "default": 8.0, "help": "Mild push interval for push-recovery eval."},
        {"name": "--push_max_vel_xy", "type": float, "default": 0.7, "help": "Mild push magnitude."},
        {"name": "--push_detect_dv_mps", "type": float, "default": 0.45, "help": "Velocity-jump threshold to detect push events."},
        {"name": "--push_settle_err_mps", "type": float, "default": 0.25, "help": "Tracking-error threshold for push recovery."},
        {"name": "--push_settle_hold_steps", "type": int, "default": 5, "help": "Consecutive settled steps to count recovery."},
        {"name": "--push_cooldown_steps", "type": int, "default": 30, "help": "Cooldown between detected push events per env."},
        {"name": "--push_max_recovery_steps", "type": int, "default": 120, "help": "Cap for recovery horizon in policy steps."},

        # Robustness scenario (mild randomization + pushes)
        {"name": "--robust_target_episodes", "type": int, "default": 1200},
        {"name": "--robust_max_steps", "type": int, "default": 240000},
        {"name": "--robust_push_interval_s", "type": float, "default": 10.0},
        {"name": "--robust_push_max_vel_xy", "type": float, "default": 0.8},
        {"name": "--robust_friction_min", "type": float, "default": 0.4},
        {"name": "--robust_friction_max", "type": float, "default": 1.5},
        {"name": "--robust_added_mass_min", "type": float, "default": -1.0},
        {"name": "--robust_added_mass_max", "type": float, "default": 1.5},

        # Play/profile consistency proxy scenario
        {"name": "--play_target_episodes", "type": int, "default": 400},
        {"name": "--play_max_steps", "type": int, "default": 100000},
        {"name": "--sim_target_episodes", "type": int, "default": 400},
        {"name": "--sim_max_steps", "type": int, "default": 100000},

        # Gates (deployment-friendly defaults; tune as needed)
        {"name": "--pass_tracking_cruise_err_mps", "type": float, "default": 0.22},
        {"name": "--pass_base_height_mean_m", "type": float, "default": 0.22},
        {"name": "--pass_base_height_p50_m", "type": float, "default": 0.22},
        {"name": "--pass_push_mean_recovery_steps", "type": float, "default": 45.0},
        {"name": "--pass_push_p90_recovery_steps", "type": float, "default": 80.0},
        {"name": "--pass_push_recovered_rate", "type": float, "default": 0.85},
        {"name": "--pass_robust_success_rate", "type": float, "default": 0.92},
        {"name": "--pass_robust_collision_frame_rate", "type": float, "default": 0.12},
        {"name": "--pass_robust_stumble_frame_rate", "type": float, "default": 0.08},
        {"name": "--pass_robust_nonfinite_frames", "type": int, "default": 0},
        {"name": "--pass_consistency_success_gap", "type": float, "default": 0.08},
        {"name": "--pass_consistency_tracking_gap_mps", "type": float, "default": 0.10},
        {"name": "--base_height_sample_envs", "type": int, "default": 32},
        {"name": "--base_height_sample_stride", "type": int, "default": 4},

        # Output
        {"name": "--out_json", "type": str, "default": "", "help": "Path to write JSON report."},
    ]
    args = gymutil.parse_arguments(description="Evaluate stage-1 blind locomotion backbone", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def _sync_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device=device)


def _safe_rate(num, den):
    if den <= 0:
        return None
    return float(num) / float(den)


def _mean_or_none(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _p_or_none(values, p):
    if len(values) == 0:
        return None
    return float(np.percentile(values, p))


def _set_constant_forward_command(cfg, speed_mps):
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 2.0
    cfg.commands.ranges.lin_vel_x = [float(speed_mps), float(speed_mps)]
    cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.ranges.heading = [0.0, 0.0]


def _apply_common_runtime(cfg, num_envs):
    cfg.env.num_envs = int(num_envs)
    cfg.env.send_timeouts = True


def _apply_nominal_cfg(cfg, args):
    _apply_common_runtime(cfg, args.num_envs)
    cfg.terrain.mesh_type = "plane"
    if not bool(args.keep_height_measurements):
        cfg.terrain.measure_heights = False
    cfg.terrain.curriculum = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    _set_constant_forward_command(cfg, args.cruise_speed_mps)


def _apply_push_cfg(cfg, args):
    _apply_common_runtime(cfg, args.num_envs)
    cfg.terrain.mesh_type = "plane"
    if not bool(args.keep_height_measurements):
        cfg.terrain.measure_heights = False
    cfg.terrain.curriculum = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = True
    cfg.domain_rand.push_interval_s = float(args.push_interval_s)
    cfg.domain_rand.max_push_vel_xy = float(args.push_max_vel_xy)
    _set_constant_forward_command(cfg, args.cruise_speed_mps)


def _apply_robust_cfg(cfg, args):
    _apply_common_runtime(cfg, args.num_envs)
    cfg.terrain.curriculum = False
    if not bool(args.keep_height_measurements):
        cfg.terrain.measure_heights = False
    cfg.domain_rand.randomize_friction = True
    cfg.domain_rand.friction_range = [float(args.robust_friction_min), float(args.robust_friction_max)]
    cfg.domain_rand.randomize_base_mass = True
    cfg.domain_rand.added_mass_range = [float(args.robust_added_mass_min), float(args.robust_added_mass_max)]
    cfg.domain_rand.push_robots = True
    cfg.domain_rand.push_interval_s = float(args.robust_push_interval_s)
    cfg.domain_rand.max_push_vel_xy = float(args.robust_push_max_vel_xy)


def _apply_play_profile_cfg(cfg, args):
    _apply_common_runtime(cfg, min(int(args.num_envs), 100))
    cfg.terrain.curriculum = False
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    if not bool(args.keep_height_measurements):
        cfg.terrain.measure_heights = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False


def _apply_first_sim_profile_cfg(cfg, args):
    _apply_common_runtime(cfg, args.num_envs)
    cfg.terrain.curriculum = False
    if not bool(args.keep_height_measurements):
        cfg.terrain.measure_heights = False
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False


def _make_env_and_policy(task_name, args, env_cfg, train_cfg, log_root):
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=task_name,
        args=args,
        train_cfg=train_cfg,
        log_root=log_root,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    return env, policy, train_cfg


def _run_rollout(
    env,
    policy,
    target_episodes,
    max_steps,
    cruise_speed_mps,
    push_eval=False,
    push_detect_dv_mps=0.45,
    push_settle_err_mps=0.25,
    push_settle_hold_steps=5,
    push_cooldown_steps=30,
    push_max_recovery_steps=120,
    base_height_sample_envs=32,
    base_height_sample_stride=4,
):
    dt = float(env.dt)
    num_envs = int(env.num_envs)

    ep_len = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    ep_return = torch.zeros(num_envs, dtype=torch.float, device=env.device)

    episodes = 0
    success_episodes = 0
    ep_lens = []
    ep_returns = []

    step_count = 0
    collision_frame_acc = 0.0
    stumble_frame_acc = 0.0

    lin_err_sum = 0.0
    lin_err_count = 0
    lin_err_cruise_sum = 0.0
    lin_err_cruise_count = 0

    nonfinite_action_frames = 0
    nonfinite_obs_frames = 0
    nonfinite_rew_frames = 0

    base_height_sum = 0.0
    base_height_count = 0
    base_height_samples = []

    # Push recovery bookkeeping
    prev_base_lin_xy = env.base_lin_vel[:, :2].clone()
    recovering = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    recover_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    settle_hold = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    push_cooldown = torch.zeros(num_envs, dtype=torch.long, device=env.device)

    push_events = 0
    push_recovered = 0
    push_recovery_steps = []

    policy_time_s = 0.0
    env_step_time_s = 0.0

    obs = env.get_observations()

    while step_count < int(max_steps) and episodes < int(target_episodes):
        _sync_if_cuda(env.device)
        t0 = time.perf_counter()
        actions = policy(obs.detach())
        _sync_if_cuda(env.device)
        t1 = time.perf_counter()
        obs, _, rews, dones, infos = env.step(actions.detach())
        _sync_if_cuda(env.device)
        t2 = time.perf_counter()

        policy_time_s += (t1 - t0)
        env_step_time_s += (t2 - t1)

        step_count += 1
        ep_len += 1
        ep_return += rews

        # Non-finite checks
        if not torch.isfinite(actions).all():
            nonfinite_action_frames += 1
        if not torch.isfinite(obs).all():
            nonfinite_obs_frames += 1
        if not torch.isfinite(rews).all():
            nonfinite_rew_frames += 1

        cmd_xy = env.commands[:, :2]
        vel_xy = env.base_lin_vel[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        tracking_err = torch.norm(cmd_xy - vel_xy, dim=1)
        base_height = env.root_states[:, 2]

        base_height_sum += float(base_height.sum().item())
        base_height_count += int(base_height.numel())
        stride = max(int(base_height_sample_stride), 1)
        if (step_count % stride) == 0:
            n_sample = min(max(int(base_height_sample_envs), 1), int(base_height.shape[0]))
            base_height_samples.append(base_height[:n_sample].detach().cpu().numpy())

        lin_err_sum += float(tracking_err.sum().item())
        lin_err_count += int(tracking_err.numel())

        cruise_mask = (cmd_norm >= 0.9 * cruise_speed_mps) & (cmd_norm <= 1.1 * cruise_speed_mps)
        if torch.any(cruise_mask):
            lin_err_cruise_sum += float(tracking_err[cruise_mask].sum().item())
            lin_err_cruise_count += int(cruise_mask.sum().item())

        if hasattr(env, "penalised_contact_indices") and env.penalised_contact_indices.numel() > 0:
            c = torch.any(torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)
            collision_frame_acc += float(c.float().mean().item())

        if hasattr(env, "feet_indices") and env.feet_indices.numel() > 0:
            f_xy = torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)
            f_z = torch.abs(env.contact_forces[:, env.feet_indices, 2])
            st = torch.any(f_xy > (5.0 * f_z), dim=1)
            stumble_frame_acc += float(st.float().mean().item())

        if push_eval:
            curr_base_lin_xy = env.base_lin_vel[:, :2]
            dv = torch.norm(curr_base_lin_xy - prev_base_lin_xy, dim=1)

            if torch.any(push_cooldown > 0):
                push_cooldown = torch.clamp(push_cooldown - 1, min=0)

            eligible = (~recovering) & (push_cooldown == 0) & (~dones.bool())
            detected = eligible & (dv >= float(push_detect_dv_mps))
            if torch.any(detected):
                ids = torch.nonzero(detected, as_tuple=False).flatten()
                recovering[ids] = True
                recover_steps[ids] = 0
                settle_hold[ids] = 0
                push_cooldown[ids] = int(push_cooldown_steps)
                push_events += int(ids.numel())

            active_ids = torch.nonzero(recovering, as_tuple=False).flatten()
            if active_ids.numel() > 0:
                recover_steps[active_ids] += 1
                settled = tracking_err[active_ids] <= float(push_settle_err_mps)
                settle_hold[active_ids] = torch.where(
                    settled,
                    settle_hold[active_ids] + 1,
                    torch.zeros_like(settle_hold[active_ids]),
                )

                recovered_mask = settle_hold[active_ids] >= int(push_settle_hold_steps)
                if torch.any(recovered_mask):
                    rec_ids = active_ids[recovered_mask]
                    push_recovered += int(rec_ids.numel())
                    push_recovery_steps.extend(recover_steps[rec_ids].detach().cpu().tolist())
                    recovering[rec_ids] = False
                    recover_steps[rec_ids] = 0
                    settle_hold[rec_ids] = 0

                timeout_mask = recover_steps[active_ids] >= int(push_max_recovery_steps)
                if torch.any(timeout_mask):
                    fail_ids = active_ids[timeout_mask]
                    push_recovery_steps.extend(
                        [int(push_max_recovery_steps) for _ in range(int(fail_ids.numel()))]
                    )
                    recovering[fail_ids] = False
                    recover_steps[fail_ids] = 0
                    settle_hold[fail_ids] = 0

            # Clone to avoid aliasing env.base_lin_vel storage across steps.
            # Without clone(), dv can collapse to zero and miss push events.
            prev_base_lin_xy = curr_base_lin_xy.clone()

        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_ids.numel() > 0:
            if "time_outs" in infos:
                timeout_flags = infos["time_outs"][done_ids].bool()
            else:
                timeout_flags = torch.zeros(done_ids.numel(), dtype=torch.bool, device=env.device)

            done_lens = ep_len[done_ids].detach().cpu().numpy()
            done_rets = ep_return[done_ids].detach().cpu().numpy()
            done_success = timeout_flags.detach().cpu().numpy().astype(np.bool_)

            for i in range(done_ids.numel()):
                episodes += 1
                success_episodes += int(bool(done_success[i]))
                ep_lens.append(int(done_lens[i]))
                ep_returns.append(float(done_rets[i]))

            # If env resets mid-recovery, mark those unresolved events as max-recovery failures.
            if push_eval:
                done_rec = done_ids[recovering[done_ids]]
                if done_rec.numel() > 0:
                    push_recovery_steps.extend(
                        [int(push_max_recovery_steps) for _ in range(int(done_rec.numel()))]
                    )
                    recovering[done_rec] = False
                    recover_steps[done_rec] = 0
                    settle_hold[done_rec] = 0
                    push_cooldown[done_rec] = 0

            ep_len[done_ids] = 0
            ep_return[done_ids] = 0.0

    if episodes == 0:
        raise RuntimeError("No finished episodes were collected. Increase max_steps or reduce difficulty.")

    total_steps = max(1, step_count)
    total_loop_time = policy_time_s + env_step_time_s
    if len(base_height_samples) > 0:
        base_height_arr = np.concatenate(base_height_samples)
        base_height_p10 = float(np.percentile(base_height_arr, 10))
        base_height_p50 = float(np.percentile(base_height_arr, 50))
        base_height_p90 = float(np.percentile(base_height_arr, 90))
        base_height_sample_count = int(base_height_arr.shape[0])
    else:
        base_height_p10 = None
        base_height_p50 = None
        base_height_p90 = None
        base_height_sample_count = 0

    out = {
        "overall": {
            "episodes": int(episodes),
            "success_rate_timeout": float(success_episodes) / float(episodes),
            "failure_rate_termination": 1.0 - float(success_episodes) / float(episodes),
            "mean_episode_length_steps": float(np.mean(ep_lens)),
            "p50_episode_length_steps": float(np.percentile(ep_lens, 50)),
            "p90_episode_length_steps": float(np.percentile(ep_lens, 90)),
            "mean_episode_length_seconds": float(np.mean(ep_lens) * dt),
            "mean_episode_return": float(np.mean(ep_returns)),
            "collision_frame_rate": float(collision_frame_acc / total_steps),
            "stumble_frame_rate": float(stumble_frame_acc / total_steps),
        },
        "tracking": {
            "mean_lin_vel_tracking_error_mps": float(lin_err_sum / max(1, lin_err_count)),
            "mean_lin_vel_tracking_error_cruise_mps": (
                None if lin_err_cruise_count == 0 else float(lin_err_cruise_sum / lin_err_cruise_count)
            ),
            "cruise_samples": int(lin_err_cruise_count),
            "cruise_speed_mps": float(cruise_speed_mps),
        },
        "numerics": {
            "nonfinite_action_frames": int(nonfinite_action_frames),
            "nonfinite_obs_frames": int(nonfinite_obs_frames),
            "nonfinite_rew_frames": int(nonfinite_rew_frames),
            "nonfinite_total_frames": int(nonfinite_action_frames + nonfinite_obs_frames + nonfinite_rew_frames),
        },
        "posture": {
            "base_height_mean_m": float(base_height_sum / max(1, base_height_count)),
            "base_height_p10_m": base_height_p10,
            "base_height_p50_m": base_height_p50,
            "base_height_p90_m": base_height_p90,
            "base_height_sample_count": int(base_height_sample_count),
            "base_height_sample_envs": int(base_height_sample_envs),
            "base_height_sample_stride": int(base_height_sample_stride),
        },
        "timing": {
            "steps_collected": int(step_count),
            "policy_inference_ms_per_step": float(policy_time_s * 1000.0 / total_steps),
            "env_step_ms_per_step": float(env_step_time_s * 1000.0 / total_steps),
            "total_loop_ms_per_step": float(total_loop_time * 1000.0 / total_steps),
            "sim_fps": float(step_count * num_envs / max(total_loop_time, 1e-9)),
        },
    }

    if push_eval:
        out["push_recovery"] = {
            "push_events": int(push_events),
            "recovered_events": int(push_recovered),
            "recovered_rate": _safe_rate(push_recovered, push_events),
            "mean_recovery_steps": _mean_or_none(push_recovery_steps),
            "p90_recovery_steps": _p_or_none(push_recovery_steps, 90),
            "mean_recovery_seconds": (None if len(push_recovery_steps) == 0 else float(np.mean(push_recovery_steps) * dt)),
            "p90_recovery_seconds": (None if len(push_recovery_steps) == 0 else float(np.percentile(push_recovery_steps, 90) * dt)),
            "recovery_samples": int(len(push_recovery_steps)),
            "detect_dv_mps": float(push_detect_dv_mps),
            "settle_err_mps": float(push_settle_err_mps),
            "settle_hold_steps": int(push_settle_hold_steps),
            "max_recovery_steps": int(push_max_recovery_steps),
        }

    return out


def _scenario_eval(name, task_name, args, base_env_cfg, base_train_cfg, log_root, cfg_fn, target_episodes, max_steps, push_eval=False):
    env_cfg = copy.deepcopy(base_env_cfg)
    train_cfg = copy.deepcopy(base_train_cfg)
    cfg_fn(env_cfg, args)

    env = None
    policy = None
    result = None
    try:
        env, policy, _ = _make_env_and_policy(task_name, args, env_cfg, train_cfg, log_root)

        result = _run_rollout(
            env=env,
            policy=policy,
            target_episodes=int(target_episodes),
            max_steps=int(max_steps),
            cruise_speed_mps=float(args.cruise_speed_mps),
            push_eval=bool(push_eval),
            push_detect_dv_mps=float(args.push_detect_dv_mps),
            push_settle_err_mps=float(args.push_settle_err_mps),
            push_settle_hold_steps=int(args.push_settle_hold_steps),
            push_cooldown_steps=int(args.push_cooldown_steps),
            push_max_recovery_steps=int(args.push_max_recovery_steps),
            base_height_sample_envs=int(args.base_height_sample_envs),
            base_height_sample_stride=int(args.base_height_sample_stride),
        )
    finally:
        if env is not None:
            try:
                if getattr(env, "viewer", None) is not None:
                    env.gym.destroy_viewer(env.viewer)
            except Exception:
                pass
            try:
                if getattr(env, "sim", None) is not None:
                    env.gym.destroy_sim(env.sim)
            except Exception:
                pass
        del policy
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result["scenario"] = name
    result["task"] = task_name
    result["target_episodes"] = int(target_episodes)
    result["max_steps"] = int(max_steps)
    return result


def main():
    args = get_eval_args()

    base_env_cfg, base_train_cfg = task_registry.get_cfgs(name=args.task)
    exp_name_for_load = args.experiment_name if args.experiment_name is not None else base_train_cfg.runner.experiment_name
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", str(exp_name_for_load))

    print("=== Stage-1 Backbone Eval: nominal tracking ===")
    nominal = _scenario_eval(
        "nominal_tracking",
        args.task,
        args,
        base_env_cfg,
        base_train_cfg,
        log_root,
        _apply_nominal_cfg,
        args.nominal_target_episodes,
        args.nominal_max_steps,
        push_eval=False,
    )

    print("=== Stage-1 Backbone Eval: push recovery ===")
    push = _scenario_eval(
        "push_recovery",
        args.task,
        args,
        base_env_cfg,
        base_train_cfg,
        log_root,
        _apply_push_cfg,
        args.push_target_episodes,
        args.push_max_steps,
        push_eval=True,
    )

    print("=== Stage-1 Backbone Eval: randomization robustness ===")
    robust = _scenario_eval(
        "robust_randomization",
        args.task,
        args,
        base_env_cfg,
        base_train_cfg,
        log_root,
        _apply_robust_cfg,
        args.robust_target_episodes,
        args.robust_max_steps,
        push_eval=False,
    )

    print("=== Stage-1 Backbone Eval: play profile ===")
    play_profile = _scenario_eval(
        "play_profile",
        args.task,
        args,
        base_env_cfg,
        base_train_cfg,
        log_root,
        _apply_play_profile_cfg,
        args.play_target_episodes,
        args.play_max_steps,
        push_eval=False,
    )

    print("=== Stage-1 Backbone Eval: first-sim profile (sim2sim proxy) ===")
    first_sim_profile = _scenario_eval(
        "first_sim_profile",
        args.task,
        args,
        base_env_cfg,
        base_train_cfg,
        log_root,
        _apply_first_sim_profile_cfg,
        args.sim_target_episodes,
        args.sim_max_steps,
        push_eval=False,
    )

    # Gates
    nominal_cruise_err = nominal["tracking"]["mean_lin_vel_tracking_error_cruise_mps"]
    gate_tracking = (nominal_cruise_err is not None) and (nominal_cruise_err <= float(args.pass_tracking_cruise_err_mps))
    nominal_base_height_mean = nominal["posture"]["base_height_mean_m"]
    nominal_base_height_p50 = nominal["posture"]["base_height_p50_m"]
    gate_posture_height = (
        (nominal_base_height_mean is not None)
        and (nominal_base_height_p50 is not None)
        and (nominal_base_height_mean >= float(args.pass_base_height_mean_m))
        and (nominal_base_height_p50 >= float(args.pass_base_height_p50_m))
    )

    push_metrics = push.get("push_recovery", {})
    push_rec_rate = push_metrics.get("recovered_rate")
    push_mean_steps = push_metrics.get("mean_recovery_steps")
    push_p90_steps = push_metrics.get("p90_recovery_steps")
    gate_push = (
        (push_rec_rate is not None)
        and (push_mean_steps is not None)
        and (push_p90_steps is not None)
        and (push_rec_rate >= float(args.pass_push_recovered_rate))
        and (push_mean_steps <= float(args.pass_push_mean_recovery_steps))
        and (push_p90_steps <= float(args.pass_push_p90_recovery_steps))
    )

    robust_success = robust["overall"]["success_rate_timeout"]
    robust_collision = robust["overall"]["collision_frame_rate"]
    robust_stumble = robust["overall"]["stumble_frame_rate"]
    robust_nonfinite = robust["numerics"]["nonfinite_total_frames"]
    gate_robust = (
        (robust_success >= float(args.pass_robust_success_rate))
        and (robust_collision <= float(args.pass_robust_collision_frame_rate))
        and (robust_stumble <= float(args.pass_robust_stumble_frame_rate))
        and (robust_nonfinite <= int(args.pass_robust_nonfinite_frames))
    )

    play_success = play_profile["overall"]["success_rate_timeout"]
    sim_success = first_sim_profile["overall"]["success_rate_timeout"]
    play_track = play_profile["tracking"]["mean_lin_vel_tracking_error_mps"]
    sim_track = first_sim_profile["tracking"]["mean_lin_vel_tracking_error_mps"]

    success_gap = abs(float(play_success) - float(sim_success))
    tracking_gap = abs(float(play_track) - float(sim_track))
    gate_consistency_proxy = (
        (success_gap <= float(args.pass_consistency_success_gap))
        and (tracking_gap <= float(args.pass_consistency_tracking_gap_mps))
    )

    gates = {
        "tracking_cruise": {
            "pass": bool(gate_tracking),
            "value": nominal_cruise_err,
            "threshold_max": float(args.pass_tracking_cruise_err_mps),
        },
        "posture_height": {
            "pass": bool(gate_posture_height),
            "base_height_mean_m": nominal_base_height_mean,
            "threshold_min_mean_m": float(args.pass_base_height_mean_m),
            "base_height_p50_m": nominal_base_height_p50,
            "threshold_min_p50_m": float(args.pass_base_height_p50_m),
        },
        "push_recovery": {
            "pass": bool(gate_push),
            "recovered_rate": push_rec_rate,
            "threshold_min_recovered_rate": float(args.pass_push_recovered_rate),
            "mean_recovery_steps": push_mean_steps,
            "threshold_max_mean_steps": float(args.pass_push_mean_recovery_steps),
            "p90_recovery_steps": push_p90_steps,
            "threshold_max_p90_steps": float(args.pass_push_p90_recovery_steps),
        },
        "robust_randomization": {
            "pass": bool(gate_robust),
            "success_rate_timeout": float(robust_success),
            "threshold_min_success_rate": float(args.pass_robust_success_rate),
            "collision_frame_rate": float(robust_collision),
            "threshold_max_collision_frame_rate": float(args.pass_robust_collision_frame_rate),
            "stumble_frame_rate": float(robust_stumble),
            "threshold_max_stumble_frame_rate": float(args.pass_robust_stumble_frame_rate),
            "nonfinite_total_frames": int(robust_nonfinite),
            "threshold_max_nonfinite_frames": int(args.pass_robust_nonfinite_frames),
        },
        "play_vs_first_sim_consistency_proxy": {
            "pass": bool(gate_consistency_proxy),
            "note": "Proxy only: this repo has no separate sim2sim runtime; replace with your external sim2sim metric when available.",
            "success_rate_gap": float(success_gap),
            "threshold_max_success_rate_gap": float(args.pass_consistency_success_gap),
            "tracking_error_gap_mps": float(tracking_gap),
            "threshold_max_tracking_error_gap_mps": float(args.pass_consistency_tracking_gap_mps),
        },
    }

    overall_pass = all(v["pass"] for v in gates.values())

    report = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now().isoformat(),
        "overall_pass": bool(overall_pass),
        "gates": gates,
        "scenarios": {
            "nominal_tracking": nominal,
            "push_recovery": push,
            "robust_randomization": robust,
            "play_profile": play_profile,
            "first_sim_profile": first_sim_profile,
        },
    }

    if args.out_json:
        out_json = args.out_json
    else:
        out_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            str(exp_name_for_load),
            "eval_reports",
        )
        os.makedirs(out_dir, exist_ok=True)
        ckpt_tag = f"{int(args.checkpoint)}" if args.checkpoint is not None else "latest"
        out_json = os.path.join(
            out_dir,
            f"stage1_backbone_eval_{args.task}_{ckpt_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Stage-1 Backbone Gate Summary ===")
    print(f"task={args.task} load_run={args.load_run} checkpoint={args.checkpoint}")
    print(f"overall_pass={overall_pass}")
    print(
        "tracking: "
        f"cruise_err={nominal_cruise_err if nominal_cruise_err is not None else 'n/a'} "
        f"<= {args.pass_tracking_cruise_err_mps} => {gate_tracking}"
    )
    print(
        "posture: "
        f"base_h_mean={nominal_base_height_mean if nominal_base_height_mean is not None else 'n/a'} "
        f"base_h_p50={nominal_base_height_p50 if nominal_base_height_p50 is not None else 'n/a'} "
        f">= ({args.pass_base_height_mean_m}, {args.pass_base_height_p50_m}) => {gate_posture_height}"
    )
    print(
        "push: "
        f"recovered_rate={push_rec_rate if push_rec_rate is not None else 'n/a'} "
        f"mean_steps={push_mean_steps if push_mean_steps is not None else 'n/a'} "
        f"p90_steps={push_p90_steps if push_p90_steps is not None else 'n/a'} => {gate_push}"
    )
    print(
        "robust: "
        f"success={robust_success:.3f} collision={robust_collision:.3f} "
        f"stumble={robust_stumble:.3f} nonfinite={robust_nonfinite} => {gate_robust}"
    )
    print(
        "consistency_proxy: "
        f"success_gap={success_gap:.3f} tracking_gap={tracking_gap:.3f} => {gate_consistency_proxy}"
    )
    print(f"report_json={out_json}")


if __name__ == "__main__":
    main()
