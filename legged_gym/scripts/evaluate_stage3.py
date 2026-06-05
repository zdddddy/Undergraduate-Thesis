import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if LEGGED_GYM_ROOT not in sys.path:
    sys.path.insert(0, LEGGED_GYM_ROOT)

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry

import numpy as np
import torch


TS_LIKE_TYPES = {
    "stage2",
    "stage2a",
    "stage2b",
    "stage2c",
    "stage3",
    "stage3a",
    "stage3b",
}

STAGE3_TYPES = {"stage3", "stage3a", "stage3b"}

TRAIN_TERRAIN_HEIGHT_RANGE = [0.06, 0.28]

TERRAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "stairs": {
        "type": "terrain_utils.random_pyramid_stairs_terrain",
        "step_width": 0.31,
        "min_step_height": TRAIN_TERRAIN_HEIGHT_RANGE[0],
        "max_step_height": TRAIN_TERRAIN_HEIGHT_RANGE[1],
        "direction": -1.0,
        "platform_size": 3.0,
    },
    "slope": {
        "type": "terrain_utils.pyramid_sloped_terrain",
        "slope": -0.35,
        "platform_size": 3.0,
    },
    "discrete": {
        "type": "terrain_utils.random_discrete_obstacles_terrain",
        "min_height": TRAIN_TERRAIN_HEIGHT_RANGE[0],
        "max_height": TRAIN_TERRAIN_HEIGHT_RANGE[1],
        "min_size": 1.0,
        "max_size": 2.0,
        "num_rects": 20,
        "platform_size": 3.0,
    },
    "rough": {
        "type": "terrain_utils.random_uniform_terrain",
        "min_height": -0.08,
        "max_height": 0.08,
        "step": 0.005,
        "downsampled_scale": 0.2,
    },
}

# Unified degraded-GT profile for fair stage2b vs stage2c comparison.
STAGE2C_DEGRADE_PROFILE: Dict[str, Any] = {
    "noise_std_m": [0.0020, 0.0060, 0.0120],
    "point_dropout_prob": [0.03, 0.12, 0.22],
    "patch_dropout_prob": [0.00, 0.08, 0.14],
    "blur_prob": [0.08, 0.25, 0.40],
    "bias_prob": [0.06, 0.20, 0.32],
    "bias_std_m": [0.0020, 0.0060, 0.0120],
    "delay_prob": [0.03, 0.14, 0.26],
    "delay_mix": [0.08, 0.24, 0.40],
    "patch_half_span_min": 1,
    "patch_half_span_max": 2,
    "curriculum_total_iters": 2000,
    "phase_ratios": [0.2, 0.6, 0.2],
}


@dataclass
class Candidate:
    name: str
    task: str
    load_run: str
    ckpt: int
    policy_mode: str  # deploy | aux
    teacher_model_path: Optional[str] = None


def build_runner_args(
    task: str,
    load_run: str,
    ckpt: int,
    num_envs: int,
    teacher_model_path: Optional[str] = None,
    headless: bool = True,
    cpu: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        headless=headless,
        cpu=cpu,
        num_envs=num_envs,
        max_iterations=None,
        resume=True,
        sync_wandb=False,
        export_onnx=False,
        debug=False,
        load_run=load_run,
        ckpt=ckpt,
        use_joystick=False,
        joystick_type="xbox",
        use_teacher=False,
        use_aux_policy=False,
        follow_robot=False,
        motion_file=None,
        motion_out_dir=None,
        distill=False,
        teacher_model_path=teacher_model_path,
    )


def task_type_of(task: str) -> str:
    return "_".join(task.split("_")[1:])


def configure_eval_env_cfg(
    env_cfg,
    terrain_name: str,
    num_envs: int,
    cmd_vx: float,
    cmd_vy: float,
    cmd_yaw: float,
    terrain_degrade_mode: str,
    terrain_degrade_profile: str,
) -> None:
    env_cfg.env.num_envs = int(num_envs)
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = min(int(env_cfg.env.num_camera_envs), int(num_envs))

    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 2
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.terrain.terrain_kwargs = copy.deepcopy(TERRAIN_PRESETS[terrain_name])

    env_cfg.env.debug = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_x = [cmd_vx, cmd_vx]
    env_cfg.commands.ranges.lin_vel_y = [cmd_vy, cmd_vy]
    env_cfg.commands.ranges.ang_vel_yaw = [cmd_yaw, cmd_yaw]
    if hasattr(env_cfg.commands.ranges, "heading"):
        env_cfg.commands.ranges.heading = [0.0, 0.0]

    if hasattr(env_cfg, "terrain_degrade"):
        if terrain_degrade_mode == "off":
            env_cfg.terrain_degrade.enable = False
        elif terrain_degrade_mode == "on":
            env_cfg.terrain_degrade.enable = True
            if terrain_degrade_profile == "stage2c":
                for k, v in STAGE2C_DEGRADE_PROFILE.items():
                    setattr(env_cfg.terrain_degrade, k, copy.deepcopy(v))
        elif terrain_degrade_mode != "default":
            raise ValueError(
                f"Unknown terrain_degrade_mode={terrain_degrade_mode}. Use default/off/on."
            )


def disable_stage3_distill_for_eval(train_cfg, task_type: str) -> None:
    if task_type not in STAGE3_TYPES:
        return
    for key in [
        "distill_action_coef",
        "distill_action_coef_final",
        "distill_latent_coef",
        "distill_latent_coef_final",
        "distill_height_coef",
        "distill_total_iters",
    ]:
        if hasattr(train_cfg.algorithm, key):
            if key == "distill_total_iters":
                setattr(train_cfg.algorithm, key, 0)
            else:
                setattr(train_cfg.algorithm, key, 0.0)


def init_policy_for_candidate(
    candidate: Candidate,
    terrain_name: str,
    num_envs: int,
    cmd_vx: float,
    cmd_vy: float,
    cmd_yaw: float,
    terrain_degrade_mode: str,
    terrain_degrade_profile: str,
    headless: bool,
    cpu: bool,
):
    args = build_runner_args(
        task=candidate.task,
        load_run=candidate.load_run,
        ckpt=candidate.ckpt,
        num_envs=num_envs,
        teacher_model_path=candidate.teacher_model_path,
        headless=headless,
        cpu=cpu,
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=candidate.task)
    configure_eval_env_cfg(
        env_cfg,
        terrain_name,
        num_envs,
        cmd_vx,
        cmd_vy,
        cmd_yaw,
        terrain_degrade_mode,
        terrain_degrade_profile,
    )

    t_type = task_type_of(candidate.task)
    disable_stage3_distill_for_eval(train_cfg, t_type)

    env, _ = task_registry.make_env(name=candidate.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=candidate.task,
        args=args,
        train_cfg=train_cfg,
    )

    if t_type in TS_LIKE_TYPES:
        if candidate.policy_mode == "deploy":
            policy = runner.get_deploy_inference_policy(device=env.device)
        elif candidate.policy_mode == "aux":
            policy = runner.get_aux_inference_policy(device=env.device)
        else:
            raise ValueError(f"Unsupported policy_mode={candidate.policy_mode}. Use deploy/aux.")
    else:
        policy = runner.get_inference_policy(device=env.device)

    return env, policy, t_type


def zeros_like_env(num_envs: int, device: torch.device):
    return {
        "steps": torch.zeros(num_envs, device=device, dtype=torch.float),
        "lin_err_sq_sum": torch.zeros(num_envs, device=device, dtype=torch.float),
        "yaw_err_sq_sum": torch.zeros(num_envs, device=device, dtype=torch.float),
        "collision_step_sum": torch.zeros(num_envs, device=device, dtype=torch.float),
        "energy_sum": torch.zeros(num_envs, device=device, dtype=torch.float),
        "distance_sum": torch.zeros(num_envs, device=device, dtype=torch.float),
    }


def prepare_initial_obs(env, t_type: str):
    if t_type in TS_LIKE_TYPES:
        obs, privileged_obs, obs_history, critic_obs = env.get_observations()
        return obs, privileged_obs, obs_history, critic_obs
    obs = env.get_observations()
    return obs, None, None, None


def step_env(
    env,
    policy,
    t_type: str,
    policy_mode: str,
    obs,
    privileged_obs,
    obs_history,
    terrain_input_ablation: str,
):
    with torch.no_grad():
        if t_type in TS_LIKE_TYPES:
            if policy_mode == "deploy":
                if terrain_input_ablation == "zero":
                    terrain_in = torch.zeros_like(privileged_obs)
                elif terrain_input_ablation == "shuffle":
                    perm = torch.randperm(privileged_obs.shape[0], device=privileged_obs.device)
                    terrain_in = privileged_obs[perm]
                elif terrain_input_ablation == "none":
                    terrain_in = privileged_obs
                else:
                    raise ValueError(
                        f"Unknown terrain_input_ablation={terrain_input_ablation}. Use none/zero/shuffle."
                    )
                actions = policy(obs, terrain_in)
            else:
                actions = policy(obs, obs_history)
            obs, privileged_obs, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
            return actions, obs, privileged_obs, obs_history, critic_obs, rews, dones, infos
        actions = policy(obs)
        obs, _, rews, dones, infos = env.step(actions.detach())
        return actions, obs, None, None, None, rews, dones, infos


def finalize_episode_for_env(
    env_id: int,
    env,
    acc: Dict[str, torch.Tensor],
    completed_episode_metrics: Dict[str, List[float]],
) -> None:
    steps = float(acc["steps"][env_id].item())
    if steps <= 0:
        return

    timeout = bool(env.time_out_buf[env_id].item())
    pass_flag = 1.0 if timeout else 0.0
    fall_flag = 0.0 if timeout else 1.0

    lin_rmse = math.sqrt(float(acc["lin_err_sq_sum"][env_id].item()) / steps)
    yaw_rmse = math.sqrt(float(acc["yaw_err_sq_sum"][env_id].item()) / steps)
    collision_rate = float(acc["collision_step_sum"][env_id].item()) / steps
    distance = float(acc["distance_sum"][env_id].item())
    energy = float(acc["energy_sum"][env_id].item())
    energy_per_meter = energy / max(distance, 1e-6)

    completed_episode_metrics["pass_rate"].append(pass_flag)
    completed_episode_metrics["fall_rate"].append(fall_flag)
    completed_episode_metrics["episode_length"].append(steps)
    completed_episode_metrics["collision_rate"].append(collision_rate)
    completed_episode_metrics["lin_vel_rmse"].append(lin_rmse)
    completed_episode_metrics["yaw_rmse"].append(yaw_rmse)
    completed_episode_metrics["energy_per_meter"].append(energy_per_meter)
    completed_episode_metrics["distance"].append(distance)

    for key in acc:
        acc[key][env_id] = 0.0


def evaluate_candidate_on_terrain(
    candidate: Candidate,
    terrain_name: str,
    num_envs: int,
    episodes_per_terrain: int,
    max_steps: int,
    cmd_vx: float,
    cmd_vy: float,
    cmd_yaw: float,
    edge_grad_threshold: float,
    terrain_degrade_mode: str,
    terrain_degrade_profile: str,
    terrain_degrade_iter: int,
    terrain_input_ablation: str,
    headless: bool,
    cpu: bool,
) -> Dict[str, Any]:
    env, policy, t_type = init_policy_for_candidate(
        candidate,
        terrain_name,
        num_envs,
        cmd_vx,
        cmd_vy,
        cmd_yaw,
        terrain_degrade_mode,
        terrain_degrade_profile,
        headless,
        cpu,
    )
    if terrain_degrade_iter >= 0 and hasattr(env, "set_training_iteration"):
        env.set_training_iteration(int(terrain_degrade_iter))

    obs, privileged_obs, obs_history, _ = prepare_initial_obs(env, t_type)
    acc = zeros_like_env(env.num_envs, env.device)
    dt = float(env.dt)

    metrics: Dict[str, List[float]] = {
        "pass_rate": [],
        "fall_rate": [],
        "episode_length": [],
        "collision_rate": [],
        "lin_vel_rmse": [],
        "yaw_rmse": [],
        "energy_per_meter": [],
        "distance": [],
    }

    # Representation metrics for stage3 (actor NSR heights vs GT local heights)
    repr_err_chunks: List[torch.Tensor] = []
    edge_abs_sum = 0.0
    edge_count = 0
    non_edge_abs_sum = 0.0
    non_edge_count = 0
    valid_sum = 0.0
    valid_count = 0
    runtime_samples_ms: List[float] = []

    completed = 0
    total_steps = 0

    while completed < episodes_per_terrain and total_steps < max_steps:
        (
            _actions,
            obs,
            privileged_obs,
            obs_history,
            _critic_obs,
            _rews,
            dones,
            _infos,
        ) = step_env(
            env,
            policy,
            t_type,
            candidate.policy_mode,
            obs,
            privileged_obs,
            obs_history,
            terrain_input_ablation,
        )

        total_steps += 1

        cmd_xy = env.commands[:, :2]
        vel_xy = env.simulator.base_lin_vel[:, :2]
        yaw_cmd = env.commands[:, 2]
        yaw_vel = env.simulator.base_ang_vel[:, 2]

        lin_err_sq = torch.sum(torch.square(cmd_xy - vel_xy), dim=1)
        yaw_err_sq = torch.square(yaw_cmd - yaw_vel)
        collision_step = torch.any(env.penalized_bodies_force_norm > 10.0, dim=1).float()
        # Mechanical power proxy (absolute joint power integral).
        inst_power = torch.sum(torch.abs(env.simulator.torques * env.simulator.dof_vel), dim=1)
        distance_step = torch.linalg.norm(vel_xy, dim=1)

        acc["steps"] += 1.0
        acc["lin_err_sq_sum"] += lin_err_sq
        acc["yaw_err_sq_sum"] += yaw_err_sq
        acc["collision_step_sum"] += collision_step
        acc["energy_sum"] += inst_power * dt
        acc["distance_sum"] += distance_step * dt

        if t_type in STAGE3_TYPES and hasattr(env, "terrain_obs_buf") and hasattr(env, "terrain_obs_gt_buf"):
            pred = env.terrain_obs_buf
            gt = env.terrain_obs_gt_buf
            err = torch.abs(pred - gt)
            repr_err_chunks.append(err.detach().cpu())

            # Edge-sensitive error on 9x9 local map.
            if gt.shape[1] == 81:
                gt_grid = gt.view(-1, 9, 9)
                err_grid = err.view(-1, 9, 9)
                dx = torch.abs(gt_grid[:, 1:, :] - gt_grid[:, :-1, :])
                dy = torch.abs(gt_grid[:, :, 1:] - gt_grid[:, :, :-1])
                edge_mask = torch.zeros_like(gt_grid, dtype=torch.bool)
                edge_dx = dx > edge_grad_threshold
                edge_dy = dy > edge_grad_threshold
                edge_mask[:, 1:, :] |= edge_dx
                edge_mask[:, :-1, :] |= edge_dx
                edge_mask[:, :, 1:] |= edge_dy
                edge_mask[:, :, :-1] |= edge_dy

                edge_vals = err_grid[edge_mask]
                non_edge_vals = err_grid[~edge_mask]
                if edge_vals.numel() > 0:
                    edge_abs_sum += float(edge_vals.sum().item())
                    edge_count += int(edge_vals.numel())
                if non_edge_vals.numel() > 0:
                    non_edge_abs_sum += float(non_edge_vals.sum().item())
                    non_edge_count += int(non_edge_vals.numel())

            if hasattr(env, "_nsr_last_valid_ratio"):
                valid = env._nsr_last_valid_ratio
                valid_sum += float(valid.sum().item())
                valid_count += int(valid.numel())
            if hasattr(env, "_nsr_last_runtime_ms"):
                runtime_samples_ms.append(float(env._nsr_last_runtime_ms))

        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        for env_id_t in done_ids:
            if completed >= episodes_per_terrain:
                break
            env_id = int(env_id_t.item())
            finalize_episode_for_env(env_id, env, acc, metrics)
            completed += 1

    out: Dict[str, Any] = {
        "candidate": candidate.name,
        "task": candidate.task,
        "policy_mode": candidate.policy_mode,
        "terrain": terrain_name,
        "episodes_target": int(episodes_per_terrain),
        "episodes_collected": int(completed),
        "total_steps": int(total_steps),
        "truncated": bool(completed < episodes_per_terrain),
        "terrain_degrade_mode": terrain_degrade_mode,
        "terrain_degrade_profile": terrain_degrade_profile,
        "terrain_degrade_iter": int(terrain_degrade_iter),
        "terrain_input_ablation": terrain_input_ablation,
        "control": {},
        "representation": None,
    }

    def summarize(xs: List[float]) -> Dict[str, float]:
        if len(xs) == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    out["control"] = {
        "pass_rate": summarize(metrics["pass_rate"]),
        "fall_rate": summarize(metrics["fall_rate"]),
        "episode_length": summarize(metrics["episode_length"]),
        "collision_rate": summarize(metrics["collision_rate"]),
        "lin_vel_rmse": summarize(metrics["lin_vel_rmse"]),
        "yaw_rmse": summarize(metrics["yaw_rmse"]),
        "energy_per_meter": summarize(metrics["energy_per_meter"]),
        "distance": summarize(metrics["distance"]),
    }

    if len(repr_err_chunks) > 0:
        repr_err = torch.cat(repr_err_chunks, dim=0).flatten().numpy()
        repr_err_mae = float(np.mean(repr_err))
        repr_err_p90 = float(np.percentile(repr_err, 90))
        repr_err_p95 = float(np.percentile(repr_err, 95))
        scale_h = float(getattr(env.obs_scales, "height_measurements", 1.0))

        edge_mae = edge_abs_sum / max(edge_count, 1)
        non_edge_mae = non_edge_abs_sum / max(non_edge_count, 1)

        out["representation"] = {
            "height_scale": scale_h,
            "mae_scaled": repr_err_mae,
            "p90_scaled": repr_err_p90,
            "p95_scaled": repr_err_p95,
            "mae_m": repr_err_mae / max(scale_h, 1e-6),
            "p90_m": repr_err_p90 / max(scale_h, 1e-6),
            "p95_m": repr_err_p95 / max(scale_h, 1e-6),
            "edge_mae_scaled": edge_mae,
            "non_edge_mae_scaled": non_edge_mae,
            "edge_over_non_edge": edge_mae / max(non_edge_mae, 1e-6),
            "valid_ratio_mean": valid_sum / max(valid_count, 1),
            "runtime_ms_mean": float(np.mean(runtime_samples_ms)) if runtime_samples_ms else float("nan"),
            "runtime_ms_p95": float(np.percentile(runtime_samples_ms, 95)) if runtime_samples_ms else float("nan"),
        }

    return out


def parse_candidate(spec: str) -> Candidate:
    # Format: name,task,load_run,ckpt,mode[,teacher_model_path]
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) not in (5, 6):
        raise ValueError(
            f"Invalid --candidate='{spec}'. Expected 5 or 6 comma-separated fields: "
            "name,task,load_run,ckpt,mode[,teacher_model_path]"
        )
    name, task, load_run, ckpt_s, mode = parts[:5]
    teacher_model_path = parts[5] if len(parts) == 6 else None
    return Candidate(
        name=name,
        task=task,
        load_run=load_run,
        ckpt=int(ckpt_s),
        policy_mode=mode,
        teacher_model_path=teacher_model_path,
    )


def default_output_dir() -> Path:
    out_dir = Path(LEGGED_GYM_RESULTS_DIR) / "evaluations" / "go2_stage3_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def mean_over_terrains(results: List[Dict[str, Any]], metric_path: Tuple[str, str]) -> float:
    vals: List[float] = []
    top, key = metric_path
    for row in results:
        if top not in row or row[top] is None:
            continue
        if key not in row[top]:
            continue
        val = row[top][key]["mean"] if isinstance(row[top][key], dict) and "mean" in row[top][key] else row[top][key]
        if np.isfinite(val):
            vals.append(float(val))
    return float(np.mean(vals)) if vals else float("nan")


def print_summary_table(flat_rows: List[Dict[str, Any]]) -> None:
    print("\n=== Stage3 Evaluation (Per Terrain) ===")
    print(
        "candidate\tterrain\tpass_rate\tfall_rate\tcoll_rate\tlin_rmse\tyaw_rmse\tep_len\tenergy_per_m\tvalid_ratio\tmae\tp95"
    )
    for r in flat_rows:
        c = r["control"]
        rep = r["representation"] or {}
        print(
            f"{r['candidate']}\t{r['terrain']}\t"
            f"{c['pass_rate']['mean']:.3f}\t{c['fall_rate']['mean']:.3f}\t"
            f"{c['collision_rate']['mean']:.3f}\t{c['lin_vel_rmse']['mean']:.3f}\t"
            f"{c['yaw_rmse']['mean']:.3f}\t{c['episode_length']['mean']:.1f}\t"
            f"{c['energy_per_meter']['mean']:.2f}\t"
            f"{rep.get('valid_ratio_mean', float('nan')):.3f}\t"
            f"{rep.get('mae_scaled', float('nan')):.4f}\t"
            f"{rep.get('p95_scaled', float('nan')):.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help=(
            "Candidate spec: name,task,load_run,ckpt,mode[,teacher_model_path]. "
            "Example: stage3a,go2_stage3a,/abs/run,900,deploy"
        ),
    )
    parser.add_argument("--terrains", type=str, default="stairs,slope,discrete,rough")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--episodes_per_terrain", type=int, default=64)
    parser.add_argument("--max_steps_per_terrain", type=int, default=3500)
    parser.add_argument("--cmd_vx", type=float, default=0.6)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--edge_grad_threshold", type=float, default=0.35)
    parser.add_argument(
        "--terrain_degrade_mode",
        type=str,
        default="default",
        choices=["default", "off", "on"],
        help="Force terrain-degrade path: default/off/on",
    )
    parser.add_argument(
        "--terrain_degrade_profile",
        type=str,
        default="stage2c",
        choices=["stage2c"],
        help="Degraded terrain profile used when --terrain_degrade_mode=on",
    )
    parser.add_argument(
        "--terrain_degrade_iter",
        type=int,
        default=-1,
        help="Set eval iteration for degradation curriculum (>=0 to enable)",
    )
    parser.add_argument(
        "--terrain_input_ablation",
        type=str,
        default="none",
        choices=["none", "zero", "shuffle"],
        help="Ablate deploy terrain input in actor path",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if len(args.candidate) == 0:
        raise ValueError("Please provide at least one --candidate spec.")

    candidates = [parse_candidate(spec) for spec in args.candidate]

    terrains = [t.strip() for t in args.terrains.split(",") if t.strip()]
    for t in terrains:
        if t not in TERRAIN_PRESETS:
            raise ValueError(f"Unknown terrain '{t}'. Available: {list(TERRAIN_PRESETS.keys())}")

    all_rows: List[Dict[str, Any]] = []

    for cand in candidates:
        for terrain in terrains:
            print(f"\n[Eval] candidate={cand.name}, task={cand.task}, mode={cand.policy_mode}, terrain={terrain}")
            row = evaluate_candidate_on_terrain(
                candidate=cand,
                terrain_name=terrain,
                num_envs=args.num_envs,
                episodes_per_terrain=args.episodes_per_terrain,
                max_steps=args.max_steps_per_terrain,
                cmd_vx=args.cmd_vx,
                cmd_vy=args.cmd_vy,
                cmd_yaw=args.cmd_yaw,
                edge_grad_threshold=args.edge_grad_threshold,
                terrain_degrade_mode=args.terrain_degrade_mode,
                terrain_degrade_profile=args.terrain_degrade_profile,
                terrain_degrade_iter=args.terrain_degrade_iter,
                terrain_input_ablation=args.terrain_input_ablation,
                headless=args.headless,
                cpu=args.cpu,
            )
            all_rows.append(row)

    print_summary_table(all_rows)

    out_dir = Path(args.out) if args.out else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"stage3_eval_{stamp}.json"

    payload = {
        "timestamp": stamp,
        "settings": {
            "terrains": terrains,
            "num_envs": args.num_envs,
            "episodes_per_terrain": args.episodes_per_terrain,
            "max_steps_per_terrain": args.max_steps_per_terrain,
            "cmd_vx": args.cmd_vx,
            "cmd_vy": args.cmd_vy,
            "cmd_yaw": args.cmd_yaw,
            "edge_grad_threshold": args.edge_grad_threshold,
            "terrain_degrade_mode": args.terrain_degrade_mode,
            "terrain_degrade_profile": args.terrain_degrade_profile,
            "terrain_degrade_iter": args.terrain_degrade_iter,
            "terrain_input_ablation": args.terrain_input_ablation,
        },
        "candidates": [cand.__dict__ for cand in candidates],
        "results": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation json: {json_path}")


if __name__ == "__main__":
    main()
