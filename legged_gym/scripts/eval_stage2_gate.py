# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Stage-2 gate evaluator:
# 1) traversal stability on stairs / discrete obstacles / slope proxy
# 2) GT-map vs degraded-map behavior shift (catastrophic switch check)

import gc
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import isaacgym  # noqa: F401
from isaacgym import gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.utils.terrain_labels import terrain_label_from_col

import numpy as np
import torch


def get_eval_args():
    custom_parameters = [
        {"name": "--task_gt", "type": str, "default": "go2_stage2_gt_teacher", "help": "Task used for GT-map evaluation."},
        {"name": "--task_deg", "type": str, "default": "go2_stage2_gt_teacher_hardened", "help": "Task used for mild-degradation evaluation."},
        {"name": "--resume", "action": "store_true", "default": True, "help": "Load policy checkpoint."},
        {"name": "--experiment_name", "type": str, "help": "Experiment name override used by both tasks."},
        {"name": "--run_name", "type": str, "help": "Run name override (optional)."},
        {"name": "--load_run", "type": str, "help": "Run folder to load when resume=True. -1 loads latest."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint id. -1 loads latest."},
        {"name": "--resume_path", "type": str, "help": "Optional absolute/relative checkpoint path."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Disable viewer."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device."},
        {"name": "--num_envs", "type": int, "default": 512, "help": "Number of environments."},
        {"name": "--seed", "type": int, "help": "Random seed override."},
        {"name": "--max_iterations", "type": int, "help": "Compatibility arg for shared cfg updater (unused in eval)."},
        {"name": "--target_episodes", "type": int, "default": 3000, "help": "Stop after this many finished episodes."},
        {"name": "--max_steps", "type": int, "default": 300000, "help": "Safety cap on rollout steps."},
        {
            "name": "--disable_push",
            "action": "store_true",
            "default": True,
            "help": "Disable random pushes during evaluation (default: true).",
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
        {"name": "--encounter_gate_m", "type": float, "default": 0.8, "help": "Encounter gate in meters."},
        {"name": "--progress_gate_m", "type": float, "default": 0.5, "help": "Progress gate in meters."},
        {"name": "--min_progress_episodes", "type": int, "default": 80, "help": "Minimum progressed episodes required per terrain group."},
        {"name": "--pass_stairs_success_progress", "type": float, "default": 0.85, "help": "Gate1 threshold for stairs succ|progress."},
        {"name": "--pass_discrete_success_progress", "type": float, "default": 0.80, "help": "Gate1 threshold for discrete_obstacles succ|progress."},
        {"name": "--pass_slope_success_progress", "type": float, "default": 0.90, "help": "Gate1 threshold for slope succ|progress."},
        {"name": "--max_overall_drop", "type": float, "default": 0.10, "help": "Gate2 threshold: max drop in overall timeout success from GT to degraded."},
        {"name": "--max_group_drop", "type": float, "default": 0.15, "help": "Gate2 threshold: max drop in succ|progress for each group."},
        {"name": "--catastrophic_floor", "type": float, "default": 0.40, "help": "Gate2 threshold: degraded succ|progress must stay above this floor."},
        {"name": "--out_json", "type": str, "default": "", "help": "Path to write JSON report."},
    ]
    args = gymutil.parse_arguments(description="Evaluate stage-2 teacher gates", custom_parameters=custom_parameters)
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


def _default_bucket():
    return {"episodes": 0, "success": 0, "len_sum": 0.0, "ret_sum": 0.0}


def _default_robust_bucket():
    return {
        "episodes": 0,
        "encounter": 0,
        "progress": 0,
        "success_encounter": 0,
        "success_progress": 0,
        "max_xy_travel_sum_m": 0.0,
    }


def _add_episode(stats_dict, label, is_success, ep_len, ep_return):
    s = stats_dict[label]
    s["episodes"] += 1
    s["success"] += int(is_success)
    s["len_sum"] += float(ep_len)
    s["ret_sum"] += float(ep_return)


def _add_episode_robust(stats_dict, label, is_success, encountered, progressed, max_xy_travel_m):
    s = stats_dict[label]
    s["episodes"] += 1
    s["encounter"] += int(encountered)
    s["progress"] += int(progressed)
    s["success_encounter"] += int(is_success and encountered)
    s["success_progress"] += int(is_success and progressed)
    s["max_xy_travel_sum_m"] += float(max_xy_travel_m)


def _finalize_bucket(stats):
    out = {}
    for k, v in stats.items():
        n = int(v["episodes"])
        if n == 0:
            continue
        out[k] = {
            "episodes": n,
            "success_rate": float(v["success"]) / float(n),
            "mean_episode_length_steps": float(v["len_sum"]) / float(n),
            "mean_episode_return": float(v["ret_sum"]) / float(n),
        }
    return out


def _finalize_robust_bucket(stats):
    out = {}
    for k, v in stats.items():
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


def _destroy_env(env):
    if env is None:
        return
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
    del env
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _evaluate_one_task(args, task_name):
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = int(args.num_envs)
    if args.disable_curriculum:
        env_cfg.terrain.curriculum = False
    if args.disable_push:
        env_cfg.domain_rand.push_robots = False

    exp_name_for_load = args.experiment_name if args.experiment_name is not None else train_cfg.runner.experiment_name
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", str(exp_name_for_load))

    env = None
    ppo_runner = None
    try:
        env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
        obs = env.get_observations()

        train_cfg.runner.resume = True
        ppo_runner, _ = task_registry.make_alg_runner(
            env=env,
            name=task_name,
            args=args,
            train_cfg=train_cfg,
            log_root=log_root,
        )
        policy = ppo_runner.get_inference_policy(device=env.device)

        dt = float(env.dt)
        num_envs = int(env.num_envs)
        max_steps = int(args.max_steps)
        target_episodes = int(args.target_episodes)

        terrain_props = getattr(env.cfg.terrain, "terrain_proportions", [1.0])
        terrain_props_cumsum = np.cumsum(np.asarray(terrain_props, dtype=np.float64)).tolist()
        terrain_num_cols = int(getattr(env.cfg.terrain, "num_cols", 1))
        has_terrain_labels = hasattr(env, "terrain_types")

        ep_len = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        ep_return = torch.zeros(num_envs, dtype=torch.float, device=env.device)
        ep_start_xy = env.root_states[:, :2].clone()
        ep_max_xy_travel = torch.zeros(num_envs, dtype=torch.float, device=env.device)

        episodes = 0
        success_episodes = 0
        ep_lens = []
        ep_returns = []
        ep_max_xy_travels = []
        by_terrain = defaultdict(_default_bucket)
        by_group = defaultdict(_default_bucket)
        by_terrain_robust = defaultdict(_default_robust_bucket)
        by_group_robust = defaultdict(_default_robust_bucket)

        step_count = 0
        collision_frame_acc = 0.0
        stumble_frame_acc = 0.0
        lin_vel_err_acc = 0.0

        policy_time_s = 0.0
        env_step_time_s = 0.0

        encounter_gate_m = float(max(args.encounter_gate_m, 0.0))
        progress_gate_m = float(max(args.progress_gate_m, 0.0))

        while step_count < max_steps and episodes < target_episodes:
            curr_xy_travel = torch.norm(env.root_states[:, :2] - ep_start_xy, dim=1)
            ep_max_xy_travel = torch.maximum(ep_max_xy_travel, curr_xy_travel)
            terrain_cols_before = env.terrain_types.clone() if has_terrain_labels else None

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

            if terrain_cols_before is not None:
                done_cols = terrain_cols_before[done_ids].detach().cpu().numpy()
            else:
                done_cols = np.zeros(done_ids.numel(), dtype=np.int64)
            done_lens = ep_len[done_ids].detach().cpu().numpy()
            done_rets = ep_return[done_ids].detach().cpu().numpy()
            done_max_xy_travel = ep_max_xy_travel[done_ids].detach().cpu().numpy()
            done_success = timeout_flags.detach().cpu().numpy().astype(np.bool_)

            for i in range(done_ids.numel()):
                if has_terrain_labels:
                    label = terrain_label_from_col(int(done_cols[i]), terrain_num_cols, terrain_props_cumsum)
                else:
                    label = "flat"
                succ = bool(done_success[i])
                this_len = int(done_lens[i])
                this_ret = float(done_rets[i])
                this_max_xy_travel = float(done_max_xy_travel[i])
                encountered = this_max_xy_travel >= encounter_gate_m
                progressed = this_max_xy_travel >= progress_gate_m

                _add_episode(by_terrain, label, succ, this_len, this_ret)
                _add_episode_robust(
                    by_terrain_robust,
                    label,
                    succ,
                    encountered,
                    progressed,
                    this_max_xy_travel,
                )

                if label == "flat":
                    group = "flat"
                elif label in ("smooth_slope", "rough_slope"):
                    group = "slope"
                elif label in ("stairs_up", "stairs_down"):
                    group = "stairs"
                elif label == "discrete_obstacles":
                    group = "discrete_obstacles"
                else:
                    group = "other"

                _add_episode(by_group, group, succ, this_len, this_ret)
                _add_episode_robust(
                    by_group_robust,
                    group,
                    succ,
                    encountered,
                    progressed,
                    this_max_xy_travel,
                )

                episodes += 1
                success_episodes += int(succ)
                ep_lens.append(this_len)
                ep_returns.append(this_ret)
                ep_max_xy_travels.append(this_max_xy_travel)

            ep_len[done_ids] = 0
            ep_return[done_ids] = 0.0
            ep_max_xy_travel[done_ids] = 0.0
            ep_start_xy[done_ids] = env.root_states[done_ids, :2]

        if episodes == 0:
            raise RuntimeError(f"No finished episodes for task={task_name}. Increase --max_steps or reduce difficulty.")

        total_steps = max(1, step_count)
        total_loop_time = policy_time_s + env_step_time_s
        report = {
            "task": task_name,
            "effective_experiment_name": exp_name_for_load,
            "load_run": args.load_run,
            "checkpoint": args.checkpoint,
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
                "mean_episode_return": float(np.mean(ep_returns)),
                "mean_lin_vel_tracking_error_mps": float(lin_vel_err_acc / total_steps),
                "collision_frame_rate": float(collision_frame_acc / total_steps),
                "stumble_frame_rate": float(stumble_frame_acc / total_steps),
            },
            "overall_robust": {
                "encounter_rate": _safe_rate(sum(int(v["encounter"]) for v in by_group_robust.values()), episodes),
                "progress_gate_rate": _safe_rate(sum(int(v["progress"]) for v in by_group_robust.values()), episodes),
                "mean_max_xy_travel_m": float(np.mean(ep_max_xy_travels)),
            },
            "timing": {
                "steps_collected": int(step_count),
                "policy_inference_ms_per_step": float(policy_time_s * 1000.0 / total_steps),
                "env_step_ms_per_step": float(env_step_time_s * 1000.0 / total_steps),
                "total_loop_ms_per_step": float(total_loop_time * 1000.0 / total_steps),
                "sim_fps": float(step_count * num_envs / max(total_loop_time, 1e-9)),
            },
            "by_terrain": _finalize_bucket(by_terrain),
            "by_group": _finalize_bucket(by_group),
            "by_terrain_robust": _finalize_robust_bucket(by_terrain_robust),
            "by_group_robust": _finalize_robust_bucket(by_group_robust),
            "runtime_reference": {
                "obs_dim": int(getattr(env, "num_obs", env_cfg.env.num_observations)),
                "action_dim": int(getattr(env, "num_actions", 0)),
                "sim_dt_s": float(env_cfg.sim.dt),
                "control_decimation": int(env_cfg.control.decimation),
                "policy_dt_s": float(env_cfg.sim.dt) * float(env_cfg.control.decimation),
                "policy_rate_hz": 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.control.decimation)),
                "action_scale": float(env_cfg.control.action_scale),
                "clip_actions": float(env_cfg.normalization.clip_actions),
            },
        }
        return report
    finally:
        _destroy_env(env)
        del ppo_runner


def _extract_group_metrics(report, group):
    grp = report.get("by_group_robust", {}).get(group, {})
    episodes = int(grp.get("episodes", 0))
    progress_rate = grp.get("progress_gate_rate", None)
    if progress_rate is None:
        progress_eps = 0
    else:
        progress_eps = int(round(float(progress_rate) * episodes))
    return {
        "episodes": episodes,
        "progress_episodes": int(progress_eps),
        "success_rate_given_progress": grp.get("success_rate_given_progress", None),
        "progress_gate_rate": progress_rate,
    }


def _gate_pass(flag):
    return bool(flag)


def main():
    args = get_eval_args()

    gt_report = _evaluate_one_task(args, args.task_gt)
    deg_report = _evaluate_one_task(args, args.task_deg)

    groups = {
        "stairs": float(args.pass_stairs_success_progress),
        "discrete_obstacles": float(args.pass_discrete_success_progress),
        "slope": float(args.pass_slope_success_progress),
    }

    gate1_groups = {}
    for group_name, threshold in groups.items():
        m = _extract_group_metrics(deg_report, group_name)
        succ = m["success_rate_given_progress"]
        passed = (
            (m["progress_episodes"] >= int(args.min_progress_episodes))
            and (succ is not None)
            and (float(succ) >= threshold)
        )
        gate1_groups[group_name] = {
            "pass": _gate_pass(passed),
            "threshold_min_success_rate_given_progress": float(threshold),
            "threshold_min_progress_episodes": int(args.min_progress_episodes),
            **m,
        }
    gate1_pass = all(v["pass"] for v in gate1_groups.values())

    gt_overall = float(gt_report["overall"]["success_rate_timeout"])
    deg_overall = float(deg_report["overall"]["success_rate_timeout"])
    overall_drop = gt_overall - deg_overall
    overall_drop_pass = overall_drop <= float(args.max_overall_drop)

    gate2_groups = {}
    for group_name in groups.keys():
        gt_m = _extract_group_metrics(gt_report, group_name)
        deg_m = _extract_group_metrics(deg_report, group_name)
        gt_succ = gt_m["success_rate_given_progress"]
        deg_succ = deg_m["success_rate_given_progress"]
        if gt_succ is None or deg_succ is None:
            drop = None
            passed = False
        else:
            drop = float(gt_succ) - float(deg_succ)
            passed = (
                (gt_m["progress_episodes"] >= int(args.min_progress_episodes))
                and (deg_m["progress_episodes"] >= int(args.min_progress_episodes))
                and (drop <= float(args.max_group_drop))
                and (float(deg_succ) >= float(args.catastrophic_floor))
            )
        gate2_groups[group_name] = {
            "pass": _gate_pass(passed),
            "threshold_max_drop": float(args.max_group_drop),
            "threshold_min_progress_episodes": int(args.min_progress_episodes),
            "threshold_min_degraded_success_rate_given_progress": float(args.catastrophic_floor),
            "gt": gt_m,
            "degraded": deg_m,
            "drop_gt_minus_degraded": drop,
        }

    gate2_pass = overall_drop_pass and all(v["pass"] for v in gate2_groups.values())
    overall_pass = gate1_pass and gate2_pass

    report = {
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "task_gt": args.task_gt,
            "task_deg": args.task_deg,
            "experiment_name": args.experiment_name,
            "load_run": args.load_run,
            "checkpoint": args.checkpoint,
            "target_episodes": int(args.target_episodes),
            "max_steps": int(args.max_steps),
            "num_envs": int(args.num_envs),
            "disable_curriculum": bool(args.disable_curriculum),
            "disable_push": bool(args.disable_push),
            "encounter_gate_m": float(args.encounter_gate_m),
            "progress_gate_m": float(args.progress_gate_m),
        },
        "gates": {
            "gate1_traversal_stability_degraded": {
                "pass": bool(gate1_pass),
                "groups": gate1_groups,
            },
            "gate2_no_catastrophic_switch_gt_vs_degraded": {
                "pass": bool(gate2_pass),
                "overall_drop": {
                    "pass": bool(overall_drop_pass),
                    "gt_success_rate_timeout": gt_overall,
                    "degraded_success_rate_timeout": deg_overall,
                    "drop_gt_minus_degraded": overall_drop,
                    "threshold_max_drop": float(args.max_overall_drop),
                },
                "groups": gate2_groups,
            },
            "overall_pass": bool(overall_pass),
        },
        "sim2sim_note": {
            "status": "manual_required",
            "note": "This script gates items (1) and (2). For strict sim2sim, run external runtime and verify obs/action/control alignment against runtime_reference below.",
            "runtime_reference_gt": gt_report.get("runtime_reference", {}),
            "runtime_reference_deg": deg_report.get("runtime_reference", {}),
        },
        "reports": {
            "gt": gt_report,
            "degraded": deg_report,
        },
    }

    if args.out_json:
        out_json = args.out_json
    else:
        exp_name = args.experiment_name if args.experiment_name else "stage2_gate"
        out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", str(exp_name), "eval_reports")
        os.makedirs(out_dir, exist_ok=True)
        ckpt_tag = f"{int(args.checkpoint)}" if args.checkpoint is not None else "latest"
        out_json = os.path.join(out_dir, f"stage2_gate_{ckpt_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Stage-2 Gate Summary ===")
    print(f"gt_task={args.task_gt} deg_task={args.task_deg}")
    print(f"experiment_name={args.experiment_name} load_run={args.load_run} checkpoint={args.checkpoint}")
    print(f"gate1_traversal_stability={gate1_pass}")
    for group_name in ("stairs", "discrete_obstacles", "slope"):
        g = gate1_groups[group_name]
        succ = g["success_rate_given_progress"]
        succ_str = "n/a" if succ is None else f"{float(succ):.3f}"
        print(
            f"  {group_name:18s} pass={g['pass']} "
            f"succ|progress={succ_str} "
            f"progress_eps={g['progress_episodes']}"
        )
    print(
        "gate2_no_catastrophic_switch="
        f"{gate2_pass} overall_drop={overall_drop:.3f} "
        f"(gt={gt_overall:.3f} deg={deg_overall:.3f})"
    )
    for group_name in ("stairs", "discrete_obstacles", "slope"):
        g = gate2_groups[group_name]
        drop = g["drop_gt_minus_degraded"]
        drop_str = "n/a" if drop is None else f"{float(drop):.3f}"
        deg_succ = g["degraded"]["success_rate_given_progress"]
        deg_succ_str = "n/a" if deg_succ is None else f"{float(deg_succ):.3f}"
        print(
            f"  {group_name:18s} pass={g['pass']} "
            f"drop(gt-deg)={drop_str} deg_succ|progress={deg_succ_str}"
        )
    print(f"overall_pass={overall_pass}")
    print(f"report_json={out_json}")


if __name__ == "__main__":
    main()
