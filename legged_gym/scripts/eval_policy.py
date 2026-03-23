# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import isaacgym  # noqa: F401
from isaacgym import gymutil
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.utils.terrain_labels import terrain_label_from_col

import numpy as np
import torch


def get_eval_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "rough_go2_stage1_gt_clean", "help": "Task name."},
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
        {"name": "--target_episodes", "type": int, "default": 2000, "help": "Stop after this many finished episodes."},
        {"name": "--max_steps", "type": int, "default": 200000, "help": "Safety cap on rollout steps."},
        {
            "name": "--disable_push",
            "action": "store_true",
            "default": True,
            "help": "Disable random pushes for base-line evaluation (default: true).",
        },
        {
            "name": "--enable_push",
            "dest": "disable_push",
            "action": "store_false",
            "help": "Enable random pushes during evaluation.",
        },
        {
            "name": "--disable_curriculum",
            "action": "store_true",
            "default": True,
            "help": "Disable terrain curriculum for stationary difficulty distribution (default: true).",
        },
        {
            "name": "--enable_curriculum",
            "dest": "disable_curriculum",
            "action": "store_false",
            "help": "Enable terrain curriculum during evaluation.",
        },
        {
            "name": "--encounter_gate_m",
            "type": float,
            "default": 0.8,
            "help": "Encounter gate (m): episode counts as 'encountered terrain' if max XY travel from episode start >= this value.",
        },
        {
            "name": "--progress_gate_m",
            "type": float,
            "default": 0.5,
            "help": "Progress gate (m): minimum max XY travel from episode start for gated success stats.",
        },
        {"name": "--out_json", "type": str, "default": "", "help": "Path to write JSON report."},
    ]
    args = gymutil.parse_arguments(description="Evaluate trained policy", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args
def add_episode(stats_dict, label, is_success, ep_len, ep_return):
    s = stats_dict[label]
    s["episodes"] += 1
    s["success"] += int(is_success)
    s["len_sum"] += float(ep_len)
    s["ret_sum"] += float(ep_return)


def default_bucket():
    return {"episodes": 0, "success": 0, "len_sum": 0.0, "ret_sum": 0.0}


def add_episode_robust(stats_dict, label, is_success, encountered, progressed, max_xy_travel_m):
    s = stats_dict[label]
    s["episodes"] += 1
    s["encounter"] += int(encountered)
    s["progress"] += int(progressed)
    s["success_encounter"] += int(is_success and encountered)
    s["success_progress"] += int(is_success and progressed)
    s["max_xy_travel_sum_m"] += float(max_xy_travel_m)


def default_robust_bucket():
    return {
        "episodes": 0,
        "encounter": 0,
        "progress": 0,
        "success_encounter": 0,
        "success_progress": 0,
        "max_xy_travel_sum_m": 0.0,
    }


def main():
    args = get_eval_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = int(args.num_envs)
    if args.disable_curriculum:
        env_cfg.terrain.curriculum = False
    if args.disable_push:
        env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    exp_name_for_load = args.experiment_name if args.experiment_name is not None else train_cfg.runner.experiment_name
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", str(exp_name_for_load))
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=log_root
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    dt = float(env.dt)
    num_envs = int(env.num_envs)
    max_steps = int(args.max_steps)
    target_episodes = int(args.target_episodes)

    terrain_props = getattr(env.cfg.terrain, "terrain_proportions", [1.0])
    terrain_props_cumsum = np.cumsum(np.asarray(terrain_props, dtype=np.float64)).tolist()
    terrain_num_cols = int(getattr(env.cfg.terrain, "num_cols", 1))

    ep_len = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    ep_return = torch.zeros(num_envs, dtype=torch.float, device=env.device)
    ep_start_xy = env.root_states[:, :2].clone()
    ep_max_xy_travel = torch.zeros(num_envs, dtype=torch.float, device=env.device)

    episodes = 0
    success_episodes = 0
    encounter_episodes = 0
    progress_episodes = 0
    success_encounter_episodes = 0
    success_progress_episodes = 0
    ep_lens = []
    ep_returns = []
    ep_max_xy_travels = []
    by_terrain = defaultdict(default_bucket)
    by_group = defaultdict(default_bucket)
    by_terrain_robust = defaultdict(default_robust_bucket)
    by_group_robust = defaultdict(default_robust_bucket)

    step_count = 0
    collision_frame_acc = 0.0
    stumble_frame_acc = 0.0
    lin_vel_err_acc = 0.0

    policy_time_s = 0.0
    env_step_time_s = 0.0

    def _sync_if_cuda(device):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device=device)

    encounter_gate_m = float(max(args.encounter_gate_m, 0.0))
    progress_gate_m = float(max(args.progress_gate_m, 0.0))

    while step_count < max_steps and episodes < target_episodes:
        # Track progress relative to each episode's own start state (not env origin).
        curr_xy_travel = torch.norm(env.root_states[:, :2] - ep_start_xy, dim=1)
        ep_max_xy_travel = torch.maximum(ep_max_xy_travel, curr_xy_travel)
        terrain_cols_before = env.terrain_types.clone()

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

        ep_len += 1
        ep_return += rews
        step_count += 1

        lin_err = torch.norm(env.commands[:, :2] - env.base_lin_vel[:, :2], dim=1).mean().item()
        lin_vel_err_acc += float(lin_err)

        if hasattr(env, "penalised_contact_indices") and env.penalised_contact_indices.numel() > 0:
            c = torch.any(torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)
            collision_frame_acc += float(c.float().mean().item())
        if hasattr(env, "feet_indices") and env.feet_indices.numel() > 0:
            f_xy = torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)
            f_z = torch.abs(env.contact_forces[:, env.feet_indices, 2])
            st = torch.any(f_xy > (5.0 * f_z), dim=1)
            stumble_frame_acc += float(st.float().mean().item())

        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        if "time_outs" in infos:
            timeout_flags = infos["time_outs"][done_ids].bool()
        else:
            timeout_flags = torch.zeros(done_ids.numel(), dtype=torch.bool, device=env.device)

        done_cols = terrain_cols_before[done_ids].detach().cpu().numpy()
        done_lens = ep_len[done_ids].detach().cpu().numpy()
        done_rets = ep_return[done_ids].detach().cpu().numpy()
        done_max_xy_travel = ep_max_xy_travel[done_ids].detach().cpu().numpy()
        done_success = timeout_flags.detach().cpu().numpy().astype(np.bool_)

        for i in range(done_ids.numel()):
            label = terrain_label_from_col(int(done_cols[i]), terrain_num_cols, terrain_props_cumsum)
            succ = bool(done_success[i])
            this_len = int(done_lens[i])
            this_ret = float(done_rets[i])
            this_max_xy_travel = float(done_max_xy_travel[i])
            encountered = this_max_xy_travel >= encounter_gate_m
            progressed = this_max_xy_travel >= progress_gate_m

            add_episode(by_terrain, label, succ, this_len, this_ret)
            add_episode_robust(
                by_terrain_robust,
                label,
                succ,
                encountered,
                progressed,
                this_max_xy_travel,
            )

            if label in ("smooth_slope", "rough_slope"):
                add_episode(by_group, "slope", succ, this_len, this_ret)
                add_episode_robust(
                    by_group_robust,
                    "slope",
                    succ,
                    encountered,
                    progressed,
                    this_max_xy_travel,
                )
            elif label in ("stairs_up", "stairs_down"):
                add_episode(by_group, "stairs", succ, this_len, this_ret)
                add_episode_robust(
                    by_group_robust,
                    "stairs",
                    succ,
                    encountered,
                    progressed,
                    this_max_xy_travel,
                )
            elif label == "discrete_obstacles":
                add_episode(by_group, "discrete_obstacles", succ, this_len, this_ret)
                add_episode_robust(
                    by_group_robust,
                    "discrete_obstacles",
                    succ,
                    encountered,
                    progressed,
                    this_max_xy_travel,
                )
            else:
                add_episode(by_group, "other", succ, this_len, this_ret)
                add_episode_robust(
                    by_group_robust,
                    "other",
                    succ,
                    encountered,
                    progressed,
                    this_max_xy_travel,
                )

            episodes += 1
            success_episodes += int(succ)
            encounter_episodes += int(encountered)
            progress_episodes += int(progressed)
            success_encounter_episodes += int(succ and encountered)
            success_progress_episodes += int(succ and progressed)
            ep_lens.append(this_len)
            ep_returns.append(this_ret)
            ep_max_xy_travels.append(this_max_xy_travel)

        ep_len[done_ids] = 0
        ep_return[done_ids] = 0.0
        ep_max_xy_travel[done_ids] = 0.0
        ep_start_xy[done_ids] = env.root_states[done_ids, :2]

    if episodes == 0:
        raise RuntimeError("No finished episodes were collected. Increase --max_steps or reduce difficulty.")

    def finalize_bucket(d):
        out = {}
        for k, v in d.items():
            n = int(v["episodes"])
            if n == 0:
                continue
            out[k] = {
                "episodes": n,
                "success_rate": float(v["success"]) / float(n),
                "mean_episode_length_steps": float(v["len_sum"]) / float(n),
                "mean_episode_length_seconds": float(v["len_sum"]) * dt / float(n),
                "mean_episode_return": float(v["ret_sum"]) / float(n),
            }
        return out

    def _safe_rate(num, den):
        if den <= 0:
            return None
        return float(num) / float(den)

    def finalize_robust_bucket(d):
        out = {}
        for k, v in d.items():
            n = int(v["episodes"])
            if n == 0:
                continue
            n_enc = int(v["encounter"])
            n_prog = int(v["progress"])
            out[k] = {
                "episodes": n,
                "encounter_rate": _safe_rate(n_enc, n),
                "success_rate_given_encounter": _safe_rate(int(v["success_encounter"]), n_enc),
                "progress_gate_rate": _safe_rate(n_prog, n),
                "success_rate_given_progress": _safe_rate(int(v["success_progress"]), n_prog),
                "timeout_success_rate_with_progress_gate": _safe_rate(int(v["success_progress"]), n),
                "mean_max_xy_travel_m": float(v["max_xy_travel_sum_m"]) / float(n),
            }
        return out

    total_steps = max(1, step_count)
    total_loop_time = policy_time_s + env_step_time_s
    report = {
        "task": args.task,
        "load_run": args.load_run,
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "num_envs": num_envs,
            "target_episodes": target_episodes,
            "max_steps": max_steps,
            "disable_push": bool(args.disable_push),
            "disable_curriculum": bool(args.disable_curriculum),
            "encounter_gate_m": encounter_gate_m,
            "progress_gate_m": progress_gate_m,
        },
        "overall": {
            "episodes": int(episodes),
            "success_rate_timeout": float(success_episodes) / float(episodes),
            "failure_rate_termination": 1.0 - float(success_episodes) / float(episodes),
            "mean_episode_length_steps": float(np.mean(ep_lens)),
            "p50_episode_length_steps": float(np.percentile(ep_lens, 50)),
            "p90_episode_length_steps": float(np.percentile(ep_lens, 90)),
            "mean_episode_length_seconds": float(np.mean(ep_lens) * dt),
            "mean_episode_return": float(np.mean(ep_returns)),
            "mean_lin_vel_tracking_error_mps": float(lin_vel_err_acc / total_steps),
            "collision_frame_rate": float(collision_frame_acc / total_steps),
            "stumble_frame_rate": float(stumble_frame_acc / total_steps),
        },
        "overall_robust": {
            "encounter_gate_m": encounter_gate_m,
            "progress_gate_m": progress_gate_m,
            "encounter_rate": _safe_rate(encounter_episodes, episodes),
            "success_rate_given_encounter": _safe_rate(success_encounter_episodes, encounter_episodes),
            "progress_gate_rate": _safe_rate(progress_episodes, episodes),
            "success_rate_given_progress": _safe_rate(success_progress_episodes, progress_episodes),
            "timeout_success_rate_with_progress_gate": _safe_rate(success_progress_episodes, episodes),
            "mean_max_xy_travel_m": float(np.mean(ep_max_xy_travels)),
        },
        "timing": {
            "steps_collected": int(step_count),
            "policy_inference_ms_per_step": float(policy_time_s * 1000.0 / total_steps),
            "env_step_ms_per_step": float(env_step_time_s * 1000.0 / total_steps),
            "total_loop_ms_per_step": float(total_loop_time * 1000.0 / total_steps),
            "sim_fps": float(step_count * num_envs / max(total_loop_time, 1e-9)),
        },
        "by_terrain": finalize_bucket(by_terrain),
        "by_group": finalize_bucket(by_group),
        "by_terrain_robust": finalize_robust_bucket(by_terrain_robust),
        "by_group_robust": finalize_robust_bucket(by_group_robust),
    }

    if args.out_json:
        out_json = args.out_json
    else:
        out_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, "logs", str(train_cfg.runner.experiment_name), "eval_reports"
        )
        os.makedirs(out_dir, exist_ok=True)
        ckpt_tag = f"{int(args.checkpoint)}" if args.checkpoint is not None else "latest"
        out_json = os.path.join(out_dir, f"eval_{args.task}_{ckpt_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Policy Evaluation Summary ===")
    print(f"task={report['task']} load_run={report['load_run']} checkpoint={report['checkpoint']}")
    print(
        "overall: "
        f"episodes={report['overall']['episodes']} "
        f"success_rate={report['overall']['success_rate_timeout']:.3f} "
        f"mean_len_s={report['overall']['mean_episode_length_seconds']:.2f} "
        f"mean_return={report['overall']['mean_episode_return']:.3f}"
    )
    print(
        "safety: "
        f"failure_rate={report['overall']['failure_rate_termination']:.3f} "
        f"collision_frame_rate={report['overall']['collision_frame_rate']:.3f} "
        f"stumble_frame_rate={report['overall']['stumble_frame_rate']:.3f}"
    )
    robust = report["overall_robust"]

    def _fmt_rate(x):
        return "n/a" if x is None else f"{x:.3f}"

    print(
        "robust: "
        f"encounter_rate={_fmt_rate(robust['encounter_rate'])} "
        f"succ|encounter={_fmt_rate(robust['success_rate_given_encounter'])} "
        f"progress_rate={_fmt_rate(robust['progress_gate_rate'])} "
        f"succ|progress={_fmt_rate(robust['success_rate_given_progress'])} "
        f"mean_max_xy_travel_m={robust['mean_max_xy_travel_m']:.2f}"
    )
    print(
        "timing: "
        f"policy_ms={report['timing']['policy_inference_ms_per_step']:.3f} "
        f"env_ms={report['timing']['env_step_ms_per_step']:.3f} "
        f"loop_ms={report['timing']['total_loop_ms_per_step']:.3f} "
        f"sim_fps={report['timing']['sim_fps']:.1f}"
    )
    print("by_group:")
    for k, v in sorted(report["by_group"].items()):
        print(
            f"  {k:18s} episodes={v['episodes']:5d} success={v['success_rate']:.3f} "
            f"mean_len_s={v['mean_episode_length_seconds']:.2f}"
        )
    print(f"report_json={out_json}")


if __name__ == "__main__":
    main()
