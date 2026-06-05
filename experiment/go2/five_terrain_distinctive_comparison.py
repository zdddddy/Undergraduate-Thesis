import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from legged_gym.scripts import evaluate_stage3
from legged_gym.utils import task_registry
from experiment.go2.paper_common import TERRAIN_LABELS, disable_eval_randomness, performance_note

import matplotlib
import numpy as np
import torch


DEFAULT_OUT_DIR = ROOT_DIR / "results" / "experiments" / "go2" / "five_terrain_distinctive_comparison"
DEFAULT_BLIND_RUN = ROOT_DIR / "results/training_logs/go2_blind/May25_01-15-41_blind_isaacgym"
DEFAULT_NSR_RUN = ROOT_DIR / "results/training_logs/go2_stage3/May04_23-49-59_stage3_isaacgym"

DISTINCTIVE_TERRAIN_ORDER = [
    "smooth_slope",
    "rough_slope",
    "stairs_up",
    "stairs_down",
    "discrete_obstacles",
]
STAIR_HEIGHT_RANGE = [0.11, 0.17]
DISCRETE_OBSTACLE_HEIGHT_RANGE = [0.12, 0.18]
DEFAULT_SEED = 1
UNEXPECTED_CONTACT_FORCE_THRESHOLD_N = 1.0

DISTINCTIVE_TERRAIN_PRESETS = {
    "smooth_slope": {
        "type": "terrain_utils.pyramid_sloped_terrain",
        "slope": -0.40,
        "platform_size": 3.0,
    },
    "rough_slope": {
        "type": "terrain_utils.rough_sloped_terrain",
        "slope": -0.40,
        "min_height": -0.08,
        "max_height": 0.08,
        "step": 0.005,
        "downsampled_scale": 0.16,
        "platform_size": 3.0,
    },
    "stairs_up": {
        "type": "terrain_utils.random_pyramid_stairs_terrain",
        "step_width": 0.31,
        "min_step_height": STAIR_HEIGHT_RANGE[0],
        "max_step_height": STAIR_HEIGHT_RANGE[1],
        "direction": 1.0,
        "platform_size": 3.0,
    },
    "stairs_down": {
        "type": "terrain_utils.random_pyramid_stairs_terrain",
        "step_width": 0.31,
        "min_step_height": STAIR_HEIGHT_RANGE[0],
        "max_step_height": STAIR_HEIGHT_RANGE[1],
        "direction": -1.0,
        "platform_size": 3.0,
    },
    "discrete_obstacles": {
        "type": "terrain_utils.random_discrete_obstacles_terrain",
        "min_height": DISCRETE_OBSTACLE_HEIGHT_RANGE[0],
        "max_height": DISCRETE_OBSTACLE_HEIGHT_RANGE[1],
        "min_size": 0.6,
        "max_size": 1.8,
        "num_rects": 36,
        "platform_size": 3.0,
    },
}

DREAMWAQ_LIKE_TYPES = {"dreamwaq", "blind"}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    name: str
    terrain_input: str
    task: str
    load_run: Path
    ckpt: int
    policy_mode: str
    terrain_degrade_mode: str = "default"
    note: str = ""


METHODS = {
    "blind": MethodSpec(
        key="blind",
        name="Blind-History",
        terrain_input="历史本体状态，无显式高度图",
        task="go2_blind",
        load_run=DEFAULT_BLIND_RUN,
        ckpt=8000,
        policy_mode="policy",
        terrain_degrade_mode="off",
        note="DreamWaQ-style history-proprioception baseline.",
    ),
    "nsr": MethodSpec(
        key="nsr",
        name="NSR",
        terrain_input="NSR 预测高度图",
        task="go2_stage3",
        load_run=DEFAULT_NSR_RUN,
        ckpt=3000,
        policy_mode="deploy",
        terrain_degrade_mode="default",
        note="Final Stage3 NSR terrain input policy.",
    ),
}


def _pct(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{100.0 * float(value):.1f}%"


def _mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _configure_distinctive_terrain() -> None:
    evaluate_stage3.TERRAIN_PRESETS.clear()
    evaluate_stage3.TERRAIN_PRESETS.update(DISTINCTIVE_TERRAIN_PRESETS)


def _init_env_policy(method: MethodSpec, terrain: str, args):
    _configure_distinctive_terrain()
    runner_args = evaluate_stage3.build_runner_args(
        task=method.task,
        load_run=str(method.load_run),
        ckpt=int(method.ckpt),
        num_envs=args.num_envs,
        headless=args.headless,
        cpu=args.cpu,
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=method.task)
    env_cfg.seed = int(args.seed)
    train_cfg.seed = int(args.seed)
    env_cfg.terrain.mesh_type = "trimesh"
    evaluate_stage3.configure_eval_env_cfg(
        env_cfg,
        terrain_name=terrain,
        num_envs=args.num_envs,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_yaw=args.cmd_yaw,
        terrain_degrade_mode=method.terrain_degrade_mode,
        terrain_degrade_profile="stage2c",
    )
    disable_eval_randomness(env_cfg)

    task_type = evaluate_stage3.task_type_of(method.task)
    evaluate_stage3.disable_stage3_distill_for_eval(train_cfg, task_type)
    env, _ = task_registry.make_env(name=method.task, args=runner_args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=method.task,
        args=runner_args,
        train_cfg=train_cfg,
    )

    if task_type in evaluate_stage3.TS_LIKE_TYPES:
        if method.policy_mode == "deploy":
            policy = runner.get_deploy_inference_policy(device=env.device)
        elif method.policy_mode == "aux":
            policy = runner.get_aux_inference_policy(device=env.device)
        else:
            raise ValueError(f"Unsupported TS policy_mode={method.policy_mode}")
    else:
        policy = runner.get_inference_policy(device=env.device)
    return env, policy, task_type


def _prepare_initial_obs(env, task_type: str):
    if task_type in DREAMWAQ_LIKE_TYPES:
        obs, privileged_obs, obs_history, _explicit_labels, _next_state = env.get_observations()
        return obs, privileged_obs, obs_history, None
    return evaluate_stage3.prepare_initial_obs(env, task_type)


def _step_env(env, policy, task_type, policy_mode, obs, privileged_obs, obs_history):
    if task_type in DREAMWAQ_LIKE_TYPES:
        with torch.no_grad():
            actions = policy(obs, obs_history)
            obs, privileged_obs, obs_history, _explicit_labels, _next_state, rews, dones, infos = env.step(
                actions.detach()
            )
        return actions, obs, privileged_obs, obs_history, None, rews, dones, infos
    return evaluate_stage3.step_env(
        env,
        policy,
        task_type,
        policy_mode,
        obs,
        privileged_obs,
        obs_history,
        terrain_input_ablation="none",
    )


def _empty_acc(num_envs: int, device: torch.device):
    return {
        "steps": torch.zeros(num_envs, device=device, dtype=torch.float),
        "vx_err_sq": torch.zeros(num_envs, device=device, dtype=torch.float),
        "collision_steps": torch.zeros(num_envs, device=device, dtype=torch.float),
    }


def _finalize_episode(env_id: int, env, acc, metrics) -> None:
    steps = float(acc["steps"][env_id].item())
    if steps <= 0.0:
        return
    timeout = bool(env.time_out_buf[env_id].item())
    metrics["success"].append(1.0 if timeout else 0.0)
    metrics["fall"].append(0.0 if timeout else 1.0)
    metrics["episode_length"].append(steps)
    metrics["vx_rmse"].append((float(acc["vx_err_sq"][env_id].item()) / steps) ** 0.5)
    metrics["collision_rate"].append(float(acc["collision_steps"][env_id].item()) / steps)
    for key in acc:
        acc[key][env_id] = 0.0


def evaluate_one(method: MethodSpec, terrain: str, args):
    env, policy, task_type = _init_env_policy(method, terrain, args)
    obs, privileged_obs, obs_history, _ = _prepare_initial_obs(env, task_type)
    acc = _empty_acc(env.num_envs, env.device)
    metrics = {
        "success": [],
        "fall": [],
        "episode_length": [],
        "vx_rmse": [],
        "collision_rate": [],
    }

    total_steps = 0
    completed = 0
    while completed < args.episodes_per_terrain and total_steps < args.max_steps_per_terrain:
        _, obs, privileged_obs, obs_history, _, _, dones, _ = _step_env(
            env,
            policy,
            task_type,
            method.policy_mode,
            obs,
            privileged_obs,
            obs_history,
        )
        total_steps += 1
        acc["steps"] += 1.0
        acc["vx_err_sq"] += torch.square(env.commands[:, 0] - env.simulator.base_lin_vel[:, 0])
        unexpected_contact = torch.any(
            env.penalized_bodies_force_norm > UNEXPECTED_CONTACT_FORCE_THRESHOLD_N,
            dim=1,
        )
        acc["collision_steps"] += unexpected_contact.float()

        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        for env_id_t in done_ids:
            if completed >= args.episodes_per_terrain:
                break
            env_id = int(env_id_t.item())
            _finalize_episode(env_id, env, acc, metrics)
            completed += 1

    success_rate = _mean(metrics["success"])
    fall_rate = _mean(metrics["fall"])
    episode_length = _mean(metrics["episode_length"])
    return {
        "method_key": method.key,
        "method": method.name,
        "terrain_input": method.terrain_input,
        "terrain_key": terrain,
        "terrain_type": TERRAIN_LABELS[terrain],
        "episodes": int(completed),
        "success_rate": success_rate,
        "episode_length": episode_length,
        "speed_tracking_error": _mean(metrics["vx_rmse"]),
        "fall_rate": fall_rate,
        "collision_rate": _mean(metrics["collision_rate"]),
        "total_steps": int(total_steps),
        "truncated": bool(completed < args.episodes_per_terrain),
        "task": method.task,
        "checkpoint": str(method.load_run / f"model_{method.ckpt}.pt"),
        "note": performance_note(success_rate, fall_rate, episode_length),
    }


def _write_detail_csv(path: Path, rows) -> None:
    fields = [
        "method_key",
        "method",
        "terrain_input",
        "terrain_key",
        "terrain_type",
        "episodes",
        "success_rate",
        "episode_length",
        "speed_tracking_error",
        "fall_rate",
        "collision_rate",
        "total_steps",
        "truncated",
        "task",
        "checkpoint",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, rows) -> None:
    fields = [
        "method_key",
        "method",
        "terrain_input",
        "episodes",
        "success_rate",
        "episode_length",
        "speed_tracking_error",
        "fall_rate",
        "collision_rate",
        "task",
        "checkpoint",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_by_method(rows):
    summary = []
    for method_key in METHODS:
        group = [row for row in rows if row["method_key"] == method_key]
        if not group:
            continue
        method = METHODS[method_key]
        summary.append(
            {
                "method_key": method_key,
                "method": method.name,
                "terrain_input": method.terrain_input,
                "episodes": int(sum(int(row["episodes"]) for row in group)),
                "success_rate": _mean([row["success_rate"] for row in group]),
                "episode_length": _mean([row["episode_length"] for row in group]),
                "speed_tracking_error": _mean([row["speed_tracking_error"] for row in group]),
                "fall_rate": _mean([row["fall_rate"] for row in group]),
                "collision_rate": _mean([row["collision_rate"] for row in group]),
                "task": method.task,
                "checkpoint": str(method.load_run / f"model_{method.ckpt}.pt"),
            }
        )
    return summary


def _read_partial_rows(partials_dir: Path):
    rows = []
    for csv_path in sorted(partials_dir.glob("*/five_terrain_single_metrics.csv")):
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for key in [
                    "success_rate",
                    "episode_length",
                    "speed_tracking_error",
                    "fall_rate",
                    "collision_rate",
                ]:
                    row[key] = float(row[key])
                for key in ["episodes", "total_steps"]:
                    row[key] = int(row[key])
                row["truncated"] = str(row["truncated"]).lower() in {"1", "true", "yes"}
                rows.append(row)
    order = {terrain: idx for idx, terrain in enumerate(DISTINCTIVE_TERRAIN_ORDER)}
    method_order = {method: idx for idx, method in enumerate(METHODS)}
    return sorted(rows, key=lambda r: (order[r["terrain_key"]], method_order[r["method_key"]]))


def _write_report(path: Path, summary_rows, detail_rows, settings) -> None:
    lines = [
        "# 五类地形高区分度越障对比实验",
        "",
        "## 表：五类地形条件下 Blind-History 与 NSR 对比",
        "",
        "| 方法 | 地形输入 | 测试 episode 数 | 成功率 | 平均 episode length | 速度跟踪误差 | 跌倒率 | 碰撞率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {terrain_input} | {episodes} | {success} | {length:.1f} | {speed:.3f} | {fall} | {collision} |".format(
                method=row["method"],
                terrain_input=row["terrain_input"],
                episodes=int(row["episodes"]),
                success=_pct(row["success_rate"]),
                length=float(row["episode_length"]),
                speed=float(row["speed_tracking_error"]),
                fall=_pct(row["fall_rate"]),
                collision=_pct(row["collision_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## 分地形详细结果",
            "",
            "| 地形类型 | 方法 | episode 数 | 成功率 | 平均 episode length | 速度跟踪误差 | 跌倒率 | 碰撞率 | 通过表现简述 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in detail_rows:
        lines.append(
            "| {terrain} | {method} | {episodes} | {success} | {length:.1f} | {speed:.3f} | {fall} | {collision} | {note} |".format(
                terrain=row["terrain_type"],
                method=row["method"],
                episodes=int(row["episodes"]),
                success=_pct(row["success_rate"]),
                length=float(row["episode_length"]),
                speed=float(row["speed_tracking_error"]),
                fall=_pct(row["fall_rate"]),
                collision=_pct(row["collision_rate"]),
                note=row["note"],
            )
        )

    lines.extend(
        [
            "",
            "## 评估设置",
            "",
            f"- 地形类型: `{', '.join(settings['terrains'])}`。",
            f"- 平滑坡面: `slope={DISTINCTIVE_TERRAIN_PRESETS['smooth_slope']['slope']}`。",
            f"- 粗糙坡面: `slope={DISTINCTIVE_TERRAIN_PRESETS['rough_slope']['slope']}`, 随机高度 `{DISTINCTIVE_TERRAIN_PRESETS['rough_slope']['min_height']:.2f}-{DISTINCTIVE_TERRAIN_PRESETS['rough_slope']['max_height']:.2f} m`。",
            f"- 台阶高度: `{STAIR_HEIGHT_RANGE[0]:.2f}-{STAIR_HEIGHT_RANGE[1]:.2f} m` 随机采样。",
            f"- 离散障碍物高度: `{DISCRETE_OBSTACLE_HEIGHT_RANGE[0]:.2f}-{DISCRETE_OBSTACLE_HEIGHT_RANGE[1]:.2f} m` 随机采样；障碍物数量 `{DISTINCTIVE_TERRAIN_PRESETS['discrete_obstacles']['num_rects']}`，尺寸 `{DISTINCTIVE_TERRAIN_PRESETS['discrete_obstacles']['min_size']:.1f}-{DISTINCTIVE_TERRAIN_PRESETS['discrete_obstacles']['max_size']:.1f} m`。",
            f"- 指令速度: `vx={settings['cmd_vx']}`, `vy={settings['cmd_vy']}`, `yaw={settings['cmd_yaw']}`。",
            f"- 每个方法-地形组合目标 episode 数: `{settings['episodes_per_terrain']}`；并行环境数: `{settings['num_envs']}`；随机种子: `{settings['seed']}`。",
            "- 成功率按 episode 是否正常达到 time-out 统计；非 time-out 终止计为跌倒。",
            f"- 速度跟踪误差采用 `vx RMSE (m/s)`；碰撞率为时间步级非期望碰撞率：足端正常接触不计，当小腿、髋部、机身等非足端 penalized body 接触力超过 `{UNEXPECTED_CONTACT_FORCE_THRESHOLD_N:.0f} N` 时，该控制步记为发生非期望碰撞；碰撞率为非期望碰撞步数占 episode 总步数的比例。",
            "- 本组评估关闭观测噪声、外部 push 和动力学随机化，用于隔离高难度地形本身对策略的影响。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(out_dir: Path, detail_rows) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )

    labels = [TERRAIN_LABELS[key] for key in DISTINCTIVE_TERRAIN_ORDER]
    method_labels = [METHODS[key].name for key in METHODS]
    colors = {"Blind-History": "#737373", "NSR": "#3f6f8f"}
    x = np.arange(len(labels))
    width = 0.34
    metrics = [
        ("success_rate", "Success rate", "%"),
        ("speed_tracking_error", "Speed tracking error", "m/s"),
        ("collision_rate", "Collision rate", "%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), constrained_layout=True)
    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        for offset_i, method_name in enumerate(method_labels):
            vals = []
            for terrain_key in DISTINCTIVE_TERRAIN_ORDER:
                row = next(
                    r
                    for r in detail_rows
                    if r["terrain_key"] == terrain_key and r["method"] == method_name
                )
                value = float(row[metric_key])
                vals.append(value * 100.0 if ylabel == "%" else value)
            offset = (offset_i - 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors.get(method_name, "#777777"),
                alpha=0.88,
                label=method_name,
            )
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.28, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"five_terrain_distinctive_comparison.{ext}", **kwargs)
    plt.close(fig)


def _aggregate(out_dir: Path, settings: dict) -> None:
    partials_dir = out_dir / "partials"
    rows = _read_partial_rows(partials_dir)
    if not rows:
        raise RuntimeError(f"No partial metrics found under {partials_dir}")
    summary_rows = _summarize_by_method(rows)
    _write_detail_csv(out_dir / "five_terrain_detailed.csv", rows)
    _write_summary_csv(out_dir / "table_five_terrain_method_summary.csv", summary_rows)
    (out_dir / "five_terrain_results.json").write_text(
        json.dumps(
            {"settings": settings, "summary_rows": summary_rows, "detail_rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(out_dir / "report.md", summary_rows, rows, settings)
    _plot(out_dir, rows)


def _run_all(args) -> None:
    out_dir = Path(args.out_dir)
    partials_dir = out_dir / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    terrains = [item.strip() for item in args.terrains.split(",") if item.strip()]

    for method_key in methods:
        for terrain in terrains:
            single_out = partials_dir / f"{method_key}_{terrain}"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--method",
                method_key,
                "--terrain",
                terrain,
                "--episodes_per_terrain",
                str(args.episodes_per_terrain),
                "--num_envs",
                str(args.num_envs),
                "--max_steps_per_terrain",
                str(args.max_steps_per_terrain),
                "--cmd_vx",
                str(args.cmd_vx),
                "--cmd_vy",
                str(args.cmd_vy),
                "--cmd_yaw",
                str(args.cmd_yaw),
                "--seed",
                str(args.seed),
                "--out_dir",
                str(single_out),
                "--headless",
            ]
            if args.cpu:
                cmd.append("--cpu")
            print(f"[Run] method={method_key}, terrain={terrain}", flush=True)
            subprocess.run(cmd, check=True)

    settings = _settings(args)
    _aggregate(out_dir, settings)
    print("Saved:", out_dir / "table_five_terrain_method_summary.csv", flush=True)
    print("Saved:", out_dir / "five_terrain_detailed.csv", flush=True)
    print("Saved:", out_dir / "report.md", flush=True)


def _settings(args) -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "methods": [item.strip() for item in args.methods.split(",") if item.strip()],
        "terrains": [item.strip() for item in args.terrains.split(",") if item.strip()],
        "episodes_per_terrain": int(args.episodes_per_terrain),
        "num_envs": int(args.num_envs),
        "max_steps_per_terrain": int(args.max_steps_per_terrain),
        "cmd_vx": float(args.cmd_vx),
        "cmd_vy": float(args.cmd_vy),
        "cmd_yaw": float(args.cmd_yaw),
        "seed": int(args.seed),
        "stair_height_range": STAIR_HEIGHT_RANGE,
        "discrete_obstacle_height_range": DISCRETE_OBSTACLE_HEIGHT_RANGE,
        "terrain_presets": DISTINCTIVE_TERRAIN_PRESETS,
        "unexpected_contact_force_threshold_n": UNEXPECTED_CONTACT_FORCE_THRESHOLD_N,
        "collision_definition": "non-foot penalized body contact timesteps divided by episode timesteps",
        "blind_checkpoint": str(METHODS["blind"].load_run / f"model_{METHODS['blind'].ckpt}.pt"),
        "nsr_checkpoint": str(METHODS["nsr"].load_run / f"model_{METHODS['nsr'].ckpt}.pt"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Blind-History and NSR on five more distinctive terrain conditions."
    )
    parser.add_argument("--run_all", action="store_true", help="Run all method-terrain pairs in subprocesses.")
    parser.add_argument("--aggregate_only", action="store_true", help="Aggregate existing partial outputs.")
    parser.add_argument("--methods", type=str, default="blind,nsr")
    parser.add_argument("--terrains", type=str, default=",".join(DISTINCTIVE_TERRAIN_ORDER))
    parser.add_argument("--method", type=str, default="blind")
    parser.add_argument("--terrain", type=str, default="stairs_up")
    parser.add_argument("--episodes_per_terrain", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_steps_per_terrain", type=int, default=3500)
    parser.add_argument("--cmd_vx", type=float, default=0.8)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    unknown_methods = [item for item in [args.method] if item not in METHODS]
    if args.run_all or args.aggregate_only:
        unknown_methods = [item.strip() for item in args.methods.split(",") if item.strip() and item.strip() not in METHODS]
    unknown_terrains = [item for item in [args.terrain] if item not in DISTINCTIVE_TERRAIN_PRESETS]
    if args.run_all or args.aggregate_only:
        unknown_terrains = [
            item.strip()
            for item in args.terrains.split(",")
            if item.strip() and item.strip() not in DISTINCTIVE_TERRAIN_PRESETS
        ]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Available: {list(METHODS)}")
    if unknown_terrains:
        raise ValueError(f"Unknown terrains: {unknown_terrains}. Available: {DISTINCTIVE_TERRAIN_ORDER}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_all:
        _run_all(args)
        return
    if args.aggregate_only:
        _aggregate(out_dir, _settings(args))
        return

    method = METHODS[args.method]
    print(f"[Eval] method={method.name}, terrain={args.terrain}, ckpt={method.ckpt}", flush=True)
    row = evaluate_one(method, args.terrain, args)
    _write_detail_csv(out_dir / "five_terrain_single_metrics.csv", [row])
    (out_dir / "five_terrain_single_results.json").write_text(
        json.dumps({"settings": _settings(args), "row": row}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"  success={_pct(row['success_rate'])}, len={row['episode_length']:.1f}, "
        f"vx_rmse={row['speed_tracking_error']:.3f}, fall={_pct(row['fall_rate'])}, "
        f"collision={_pct(row['collision_rate'])}",
        flush=True,
    )
    print("Saved:", out_dir / "five_terrain_single_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
