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
from experiment.go2.five_terrain_distinctive_comparison import (
    DEFAULT_NSR_RUN,
    DISCRETE_OBSTACLE_HEIGHT_RANGE,
    DISTINCTIVE_TERRAIN_ORDER,
    DISTINCTIVE_TERRAIN_PRESETS,
    STAIR_HEIGHT_RANGE,
    UNEXPECTED_CONTACT_FORCE_THRESHOLD_N,
)
from experiment.go2.paper_common import TERRAIN_LABELS, disable_eval_randomness, performance_note

import matplotlib
import numpy as np
import torch


DEFAULT_OUT_DIR = ROOT_DIR / "results" / "experiments" / "go2" / "nsr_five_terrain_robustness"


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    name: str
    note: str


CONDITIONS = {
    "clean": ConditionSpec(
        key="clean",
        name="Clean",
        note="关闭外部 push、动力学随机化和观测噪声。",
    ),
    "push": ConditionSpec(
        key="push",
        name="External Push",
        note="仅启用外部速度扰动：push_interval_s=6, max_push_vel_xy=1.0。",
    ),
    "dr_push": ConditionSpec(
        key="dr_push",
        name="DR + Push",
        note="启用训练级动力学随机化与更强外部 push。",
    ),
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    task: str
    load_run: Path
    ckpt: int
    policy_mode: str
    terrain_degrade_mode: str = "default"


NSR_METHOD = MethodSpec(
    name="NSR",
    task="go2_stage3",
    load_run=DEFAULT_NSR_RUN,
    ckpt=3000,
    policy_mode="deploy",
    terrain_degrade_mode="default",
)


def _pct(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{100.0 * float(value):.1f}%"


def _mean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _configure_terrain() -> None:
    evaluate_stage3.TERRAIN_PRESETS.clear()
    evaluate_stage3.TERRAIN_PRESETS.update(DISTINCTIVE_TERRAIN_PRESETS)


def _apply_robustness_condition(env_cfg, condition_key: str) -> None:
    disable_eval_randomness(env_cfg)
    if not hasattr(env_cfg, "domain_rand"):
        return
    dr = env_cfg.domain_rand
    if condition_key == "clean":
        return
    if condition_key == "push":
        dr.push_robots = True
        dr.push_interval_s = 6
        dr.max_push_vel_xy = 1.0
        return
    if condition_key == "dr_push":
        dr.randomize_friction = True
        dr.friction_range = [0.1, 2.5]
        dr.randomize_base_mass = True
        dr.added_mass_range = [-2.0, 2.0]
        dr.randomize_com_displacement = True
        dr.com_pos_x_range = [-0.05, 0.05]
        dr.com_pos_y_range = [-0.05, 0.05]
        dr.com_pos_z_range = [-0.04, 0.04]
        dr.randomize_pd_gain = True
        dr.kp_range = [0.7, 1.3]
        dr.kd_range = [0.7, 1.3]
        dr.push_robots = True
        dr.push_interval_s = 6
        dr.max_push_vel_xy = 1.6
        return
    raise ValueError(f"Unknown condition: {condition_key}")


def _init_env_policy(terrain: str, condition_key: str, args):
    _configure_terrain()
    runner_args = evaluate_stage3.build_runner_args(
        task=NSR_METHOD.task,
        load_run=str(NSR_METHOD.load_run),
        ckpt=int(NSR_METHOD.ckpt),
        num_envs=args.num_envs,
        headless=args.headless,
        cpu=args.cpu,
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=NSR_METHOD.task)
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
        terrain_degrade_mode=NSR_METHOD.terrain_degrade_mode,
        terrain_degrade_profile="stage2c",
    )
    _apply_robustness_condition(env_cfg, condition_key)

    task_type = evaluate_stage3.task_type_of(NSR_METHOD.task)
    evaluate_stage3.disable_stage3_distill_for_eval(train_cfg, task_type)
    env, _ = task_registry.make_env(name=NSR_METHOD.task, args=runner_args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=NSR_METHOD.task,
        args=runner_args,
        train_cfg=train_cfg,
    )
    policy = runner.get_deploy_inference_policy(device=env.device)
    return env, policy, task_type


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


def evaluate_one(terrain: str, condition_key: str, args):
    env, policy, task_type = _init_env_policy(terrain, condition_key, args)
    obs, privileged_obs, obs_history, _ = evaluate_stage3.prepare_initial_obs(env, task_type)
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
        _, obs, privileged_obs, obs_history, _, _, dones, _ = evaluate_stage3.step_env(
            env,
            policy,
            task_type,
            NSR_METHOD.policy_mode,
            obs,
            privileged_obs,
            obs_history,
            terrain_input_ablation="none",
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
    condition = CONDITIONS[condition_key]
    return {
        "condition_key": condition.key,
        "condition": condition.name,
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
        "task": NSR_METHOD.task,
        "checkpoint": str(NSR_METHOD.load_run / f"model_{NSR_METHOD.ckpt}.pt"),
        "note": performance_note(success_rate, fall_rate, episode_length),
    }


def _write_detail_csv(path: Path, rows) -> None:
    fields = [
        "condition_key",
        "condition",
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
        "condition_key",
        "condition",
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


def _summarize_by_condition(rows):
    summary = []
    for key, condition in CONDITIONS.items():
        group = [row for row in rows if row["condition_key"] == key]
        if not group:
            continue
        summary.append(
            {
                "condition_key": key,
                "condition": condition.name,
                "episodes": int(sum(int(row["episodes"]) for row in group)),
                "success_rate": _mean([row["success_rate"] for row in group]),
                "episode_length": _mean([row["episode_length"] for row in group]),
                "speed_tracking_error": _mean([row["speed_tracking_error"] for row in group]),
                "fall_rate": _mean([row["fall_rate"] for row in group]),
                "collision_rate": _mean([row["collision_rate"] for row in group]),
                "task": NSR_METHOD.task,
                "checkpoint": str(NSR_METHOD.load_run / f"model_{NSR_METHOD.ckpt}.pt"),
            }
        )
    return summary


def _read_partial_rows(partials_dir: Path):
    rows = []
    for csv_path in sorted(partials_dir.glob("*/robustness_single_metrics.csv")):
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
    condition_order = {key: idx for idx, key in enumerate(CONDITIONS)}
    terrain_order = {key: idx for idx, key in enumerate(DISTINCTIVE_TERRAIN_ORDER)}
    return sorted(rows, key=lambda r: (condition_order[r["condition_key"]], terrain_order[r["terrain_key"]]))


def _write_report(path: Path, summary_rows, detail_rows, settings) -> None:
    lines = [
        "# NSR 策略五类地形鲁棒性实验",
        "",
        "## 表：NSR 策略在不同扰动条件下的鲁棒性统计",
        "",
        "| 条件 | 测试 episode 数 | 成功率 | 平均 episode length | 速度跟踪误差 | 跌倒率 | 碰撞率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {episodes} | {success} | {length:.1f} | {speed:.3f} | {fall} | {collision} |".format(
                condition=row["condition"],
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
            "| 条件 | 地形类型 | episode 数 | 成功率 | 平均 episode length | 速度跟踪误差 | 跌倒率 | 碰撞率 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in detail_rows:
        lines.append(
            "| {condition} | {terrain} | {episodes} | {success} | {length:.1f} | {speed:.3f} | {fall} | {collision} |".format(
                condition=row["condition"],
                terrain=row["terrain_type"],
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
            "## 评估设置",
            "",
            f"- 策略 checkpoint: `{settings['checkpoint']}`。",
            f"- 地形类型: `{', '.join(settings['terrains'])}`。",
            f"- 台阶高度: `{STAIR_HEIGHT_RANGE[0]:.2f}-{STAIR_HEIGHT_RANGE[1]:.2f} m` 随机采样。",
            f"- 离散障碍物高度: `{DISCRETE_OBSTACLE_HEIGHT_RANGE[0]:.2f}-{DISCRETE_OBSTACLE_HEIGHT_RANGE[1]:.2f} m` 随机采样；障碍物数量 `{DISTINCTIVE_TERRAIN_PRESETS['discrete_obstacles']['num_rects']}`，尺寸 `{DISTINCTIVE_TERRAIN_PRESETS['discrete_obstacles']['min_size']:.1f}-{DISTINCTIVE_TERRAIN_PRESETS['discrete_obstacles']['max_size']:.1f} m`。",
            f"- 指令速度: `vx={settings['cmd_vx']}`, `vy={settings['cmd_vy']}`, `yaw={settings['cmd_yaw']}`。",
            f"- 每个条件-地形组合目标 episode 数: `{settings['episodes_per_terrain']}`；并行环境数: `{settings['num_envs']}`；随机种子: `{settings['seed']}`。",
            f"- 碰撞率为时间步级非期望碰撞率：足端正常接触不计，非足端 penalized body 接触力超过 `{UNEXPECTED_CONTACT_FORCE_THRESHOLD_N:.0f} N` 的控制步数 / episode 总步数。",
            "- Clean: 关闭外部 push、动力学随机化和观测噪声。",
            "- External Push: 仅启用 `push_interval_s=6`, `max_push_vel_xy=1.0` 的外部速度扰动。",
            "- DR + Push: 启用 friction `[0.1, 2.5]`、base mass `[-2, 2] kg`、CoM 位移、PD gain `[0.7, 1.3]` 与 `max_push_vel_xy=1.6`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(out_dir: Path, summary_rows, detail_rows) -> None:
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

    labels = [row["condition"] for row in summary_rows]
    x = np.arange(len(labels))
    colors = ["#4f7f45", "#bc7c32", "#8f3f3f"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), constrained_layout=True)
    specs = [
        ("success_rate", "Success rate", "%"),
        ("speed_tracking_error", "Speed tracking error", "m/s"),
        ("collision_rate", "Collision rate", "%"),
    ]
    for ax, (metric_key, title, ylabel) in zip(axes, specs):
        vals = [
            float(row[metric_key]) * 100.0 if ylabel == "%" else float(row[metric_key])
            for row in summary_rows
        ]
        ax.bar(x, vals, color=colors[: len(vals)], width=0.62, alpha=0.88)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.28, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out_dir / f"nsr_five_terrain_robustness.{ext}", **kwargs)
    plt.close(fig)


def _aggregate(out_dir: Path, settings: dict) -> None:
    rows = _read_partial_rows(out_dir / "partials")
    if not rows:
        raise RuntimeError(f"No partial metrics found under {out_dir / 'partials'}")
    summary_rows = _summarize_by_condition(rows)
    _write_detail_csv(out_dir / "nsr_five_terrain_robustness_detailed.csv", rows)
    _write_summary_csv(out_dir / "table_nsr_robustness_summary.csv", summary_rows)
    (out_dir / "nsr_five_terrain_robustness_results.json").write_text(
        json.dumps(
            {"settings": settings, "summary_rows": summary_rows, "detail_rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(out_dir / "report.md", summary_rows, rows, settings)
    _plot(out_dir, summary_rows, rows)


def _settings(args) -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "conditions": [item.strip() for item in args.conditions.split(",") if item.strip()],
        "terrains": [item.strip() for item in args.terrains.split(",") if item.strip()],
        "episodes_per_terrain": int(args.episodes_per_terrain),
        "num_envs": int(args.num_envs),
        "max_steps_per_terrain": int(args.max_steps_per_terrain),
        "cmd_vx": float(args.cmd_vx),
        "cmd_vy": float(args.cmd_vy),
        "cmd_yaw": float(args.cmd_yaw),
        "seed": int(args.seed),
        "checkpoint": str(NSR_METHOD.load_run / f"model_{NSR_METHOD.ckpt}.pt"),
        "terrain_presets": DISTINCTIVE_TERRAIN_PRESETS,
    }


def _run_all(args) -> None:
    out_dir = Path(args.out_dir)
    partials_dir = out_dir / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    terrains = [item.strip() for item in args.terrains.split(",") if item.strip()]

    for condition in conditions:
        for terrain in terrains:
            single_out = partials_dir / f"{condition}_{terrain}"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--condition",
                condition,
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
            print(f"[Run] condition={condition}, terrain={terrain}", flush=True)
            subprocess.run(cmd, check=True)

    _aggregate(out_dir, _settings(args))
    print("Saved:", out_dir / "table_nsr_robustness_summary.csv", flush=True)
    print("Saved:", out_dir / "nsr_five_terrain_robustness_detailed.csv", flush=True)
    print("Saved:", out_dir / "report.md", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NSR robustness on five distinctive terrain types.")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--conditions", type=str, default="clean,push,dr_push")
    parser.add_argument("--condition", type=str, default="clean")
    parser.add_argument("--terrains", type=str, default=",".join(DISTINCTIVE_TERRAIN_ORDER))
    parser.add_argument("--terrain", type=str, default="smooth_slope")
    parser.add_argument("--episodes_per_terrain", type=int, default=64)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_steps_per_terrain", type=int, default=3500)
    parser.add_argument("--cmd_vx", type=float, default=0.8)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_all or args.aggregate_only:
        unknown_conditions = [
            item.strip() for item in args.conditions.split(",") if item.strip() and item.strip() not in CONDITIONS
        ]
        unknown_terrains = [
            item.strip()
            for item in args.terrains.split(",")
            if item.strip() and item.strip() not in DISTINCTIVE_TERRAIN_PRESETS
        ]
    else:
        unknown_conditions = [] if args.condition in CONDITIONS else [args.condition]
        unknown_terrains = [] if args.terrain in DISTINCTIVE_TERRAIN_PRESETS else [args.terrain]
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {unknown_conditions}. Available: {list(CONDITIONS)}")
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

    print(f"[Eval] condition={args.condition}, terrain={args.terrain}, ckpt={NSR_METHOD.ckpt}", flush=True)
    row = evaluate_one(args.terrain, args.condition, args)
    _write_detail_csv(out_dir / "robustness_single_metrics.csv", [row])
    (out_dir / "robustness_single_results.json").write_text(
        json.dumps({"settings": _settings(args), "row": row}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"  success={_pct(row['success_rate'])}, len={row['episode_length']:.1f}, "
        f"vx_rmse={row['speed_tracking_error']:.3f}, fall={_pct(row['fall_rate'])}, "
        f"collision={_pct(row['collision_rate'])}",
        flush=True,
    )
    print("Saved:", out_dir / "robustness_single_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
