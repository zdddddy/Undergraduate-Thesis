import argparse
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "experiments" / "go2" / "training_process_analysis"

STAGE_SWITCH_ITER = 11000
FINAL_ITER = 14000
STAGE_SPLIT_ITER = 8000
PAPER_FIGURE_BASENAME = "two_stage_training_process_paper"
REWARD_TERMS_FIGURE_BASENAME = "two_stage_reward_terms_paper"

RUNS = [
    {
        "name": "stage2a_gt",
        "phase": "stage1_gt_degraded_gt",
        "phase_label": "Stage 1: GT / Degraded-GT",
        "terrain_input": "GT",
        "event_dir": ROOT_DIR / "results/training_logs/go2_stage2/Apr11_00-07-00_stage2a_isaacgym",
        "step_offset": 0,
    },
    {
        "name": "stage2b_gt_hard",
        "phase": "stage1_gt_degraded_gt",
        "phase_label": "Stage 1: GT / Degraded-GT",
        "terrain_input": "GT",
        "event_dir": ROOT_DIR / "results/training_logs/go2_stage2/Apr11_15-47-04_stage2b_isaacgym",
        "step_offset": 0,
    },
    {
        "name": "stage2c_degraded_gt",
        "phase": "stage1_gt_degraded_gt",
        "phase_label": "Stage 1: GT / Degraded-GT",
        "terrain_input": "Degraded-GT",
        "event_dir": ROOT_DIR / "results/training_logs/go2_stage2/Apr16_00-50-38_stage2c_isaacgym",
        "step_offset": 0,
    },
    {
        "name": "stage3_nsr_adapt",
        "phase": "stage2_nsr",
        "phase_label": "Stage 2: NSR adaptation",
        "terrain_input": "NSR prediction",
        "event_dir": ROOT_DIR / "results/training_logs/go2_stage3/May04_23-49-59_stage3_isaacgym",
        "step_offset": STAGE_SWITCH_ITER,
    },
]

METRIC_TAGS = OrderedDict(
    [
        ("mean_episode_reward", "Train/mean_reward"),
        ("mean_episode_length", "Train/mean_episode_length"),
        ("terrain_level", "Episode/terrain_level"),
        ("tracking_lin_reward", "Episode/rew_tracking_lin_vel"),
        ("tracking_ang_reward", "Episode/rew_tracking_ang_vel"),
        ("collision_penalty", "Episode/rew_collision"),
        ("foot_clearance_reward", "Episode/rew_foot_clearance"),
    ]
)

TABLE_METRICS = [
    "mean_episode_reward",
    "mean_episode_length",
    "terrain_level",
    "tracking_reward",
    "collision_penalty",
    "foot_clearance_reward",
]

CHECKPOINTS = [
    {
        "stage": "第一阶段前期",
        "terrain_input": "GT",
        "step": 8000,
        "window_start": 0,
        "window_end": 7999,
        "window_label": "0-8000",
        "checkpoint": ROOT_DIR / "results/training_logs/go2_stage2/Apr11_15-47-04_stage2b_isaacgym/model_8000.pt",
    },
    {
        "stage": "第一阶段后期",
        "terrain_input": "Degraded-GT",
        "step": STAGE_SWITCH_ITER,
        "window_start": 8000,
        "window_end": STAGE_SWITCH_ITER - 1,
        "window_label": "8000-11000",
        "checkpoint": ROOT_DIR / "results/training_logs/go2_stage2/Apr16_00-50-38_stage2c_isaacgym/model_11000.pt",
    },
    {
        "stage": "第二阶段",
        "terrain_input": "NSR prediction",
        "step": FINAL_ITER,
        "window_start": STAGE_SWITCH_ITER,
        "window_end": FINAL_ITER - 1,
        "window_label": "11000-14000",
        "checkpoint": ROOT_DIR / "results/training_logs/go2_stage3/May04_23-49-59_stage3_isaacgym/model_3000.pt",
    },
]


def _find_event_file(event_dir):
    files = sorted(Path(event_dir).glob("events.out.tfevents*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event file found under {event_dir}")
    return files[-1]


def _load_scalars(event_file):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    out = {}
    for metric, tag in METRIC_TAGS.items():
        if tag not in tags:
            continue
        vals = acc.Scalars(tag)
        out[metric] = [(int(v.step), float(v.value)) for v in vals]
    return out


def _build_rows():
    rows_by_step = OrderedDict()
    run_manifest = []
    for run in RUNS:
        event_file = _find_event_file(run["event_dir"])
        scalars = _load_scalars(event_file)
        run_manifest.append(
            {
                "name": run["name"],
                "phase": run["phase"],
                "terrain_input": run["terrain_input"],
                "event_file": str(event_file),
                "step_offset": run["step_offset"],
            }
        )
        for metric, values in scalars.items():
            for raw_step, value in values:
                global_step = raw_step + int(run["step_offset"])
                key = (global_step, run["name"])
                if key not in rows_by_step:
                    rows_by_step[key] = {
                        "global_step": global_step,
                        "raw_step": raw_step,
                        "run": run["name"],
                        "phase": run["phase"],
                        "phase_label": run["phase_label"],
                        "terrain_input": run["terrain_input"],
                    }
                rows_by_step[key][metric] = value

    for row in rows_by_step.values():
        has_lin = "tracking_lin_reward" in row and row["tracking_lin_reward"] != ""
        has_ang = "tracking_ang_reward" in row and row["tracking_ang_reward"] != ""
        if has_lin and has_ang:
            row["tracking_reward"] = float(row["tracking_lin_reward"]) + float(row["tracking_ang_reward"])
        elif has_lin:
            row["tracking_reward"] = float(row["tracking_lin_reward"])
        elif has_ang:
            row["tracking_reward"] = float(row["tracking_ang_reward"])

    rows = list(rows_by_step.values())
    rows.sort(key=lambda r: (r["global_step"], r["run"]))
    return rows, run_manifest


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _value_at_window(rows, metric, start, end):
    vals = [
        float(r[metric])
        for r in rows
        if start <= int(r["global_step"]) <= end and metric in r and r[metric] != "" and np.isfinite(float(r[metric]))
    ]
    if not vals:
        return None
    return float(np.mean(vals))


def _make_checkpoint_table(rows):
    out = []
    for ckpt in CHECKPOINTS:
        row = {
            "training_stage": ckpt["stage"],
            "terrain_input": ckpt["terrain_input"],
            "nominal_iteration": ckpt["step"],
            "metric_window": ckpt.get("window_label", f"{ckpt['window_start']}-{ckpt['window_end']}"),
            "checkpoint": str(ckpt["checkpoint"]),
            "checkpoint_exists": ckpt["checkpoint"].exists(),
        }
        for metric in TABLE_METRICS:
            value = _value_at_window(rows, metric, ckpt["window_start"], ckpt["window_end"])
            row[metric] = "" if value is None else value
        out.append(row)
    return out


def _metric_arrays(rows, metric):
    filtered = [r for r in rows if metric in r and r[metric] != ""]
    return (
        np.asarray([int(r["global_step"]) for r in filtered], dtype=np.float64),
        np.asarray([float(r[metric]) for r in filtered], dtype=np.float64),
        np.asarray([r["terrain_input"] for r in filtered], dtype=object),
    )


def _smooth_series_by_segment(values, segments, window):
    smoothed = np.full_like(values, np.nan, dtype=np.float64)
    for segment in sorted(set(segments)):
        idx = np.where(segments == segment)[0]
        idx = idx[np.isfinite(values[idx])]
        if idx.size == 0:
            continue
        vals = values[idx]
        win = max(1, min(int(window), vals.size))
        if win % 2 == 0:
            win -= 1
        if win <= 1:
            smoothed[idx] = vals
            continue
        kernel = np.ones(win, dtype=np.float64) / float(win)
        pad = win // 2
        padded = np.pad(vals, (pad, pad), mode="edge")
        smoothed[idx] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _plot_paper_figure(rows, out_dir, smooth_window=151):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.transforms import blended_transform_factory

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
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
            "axes.linewidth": 0.7,
        }
    )

    metrics = [
        ("mean_episode_reward", "(a) Episode reward", "Reward", "#2f5597"),
        ("mean_episode_length", "(b) Episode length", "Length", "#2f5597"),
        ("terrain_level", "(c) Terrain level", "Level", "#4f7f45"),
    ]
    stage_regions = [
        (0, STAGE_SPLIT_ITER, "Stage I-A\nGT", "#eaf1fb"),
        (STAGE_SPLIT_ITER, STAGE_SWITCH_ITER, "Stage I-B\nDegraded-GT", "#f7efe2"),
        (STAGE_SWITCH_ITER, FINAL_ITER, "Stage II\nNSR", "#eaf5ec"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.7), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, (metric, title, ylabel, color) in zip(axes, metrics):
        steps, values, segments = _metric_arrays(rows, metric)
        smooth = _smooth_series_by_segment(values, segments, smooth_window)
        for start, end, _, fill in stage_regions:
            ax.axvspan(start, end, color=fill, alpha=0.55, lw=0, zorder=0)
        ax.axvline(STAGE_SPLIT_ITER, color="#6b7280", lw=0.8, ls="--", zorder=1)
        ax.axvline(STAGE_SWITCH_ITER, color="#111827", lw=0.9, ls="--", zorder=1)
        ax.plot(steps, values, color=color, lw=0.45, alpha=0.16, zorder=2)
        ax.plot(steps, smooth, color=color, lw=1.45, zorder=3)
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, FINAL_ITER)
        ax.grid(axis="y", color="#d7d7d7", lw=0.45, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.5, width=0.6)

    top_axis = axes[0]
    transform = blended_transform_factory(top_axis.transData, top_axis.transAxes)
    for start, end, label, _ in stage_regions:
        top_axis.text(
            (start + end) / 2.0,
            1.08,
            label,
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#30343b",
            linespacing=1.1,
        )

    axes[-1].set_xlabel("Training iteration")
    for ax in axes:
        ax.set_xticks([0, 4000, 8000, 11000, 14000])
        ax.set_xticklabels(["0", "4000", "8000", "11000", "14000"])

    handles = [
        plt.Line2D([0], [0], color="#555555", lw=0.45, alpha=0.35, label="raw"),
        plt.Line2D([0], [0], color="#2f5597", lw=1.45, label="smoothed"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="lower right", handlelength=2.2)

    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(Path(out_dir) / f"{PAPER_FIGURE_BASENAME}.{ext}", **kwargs)
    plt.close(fig)


def _plot_reward_terms_figure(rows, out_dir, smooth_window=151):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.transforms import blended_transform_factory

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
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
            "axes.linewidth": 0.7,
        }
    )

    metrics = [
        ("tracking_reward", "(a) Tracking reward", "Reward", "#2f5597"),
        ("collision_penalty", "(b) Collision penalty", "Penalty", "#9a3d3d"),
        ("foot_clearance_reward", "(c) Foot clearance reward", "Reward", "#4f7f45"),
    ]
    stage_regions = [
        (0, STAGE_SPLIT_ITER, "Stage I-A\nGT", "#eaf1fb"),
        (STAGE_SPLIT_ITER, STAGE_SWITCH_ITER, "Stage I-B\nDegraded-GT", "#f7efe2"),
        (STAGE_SWITCH_ITER, FINAL_ITER, "Stage II\nNSR", "#eaf5ec"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.7), sharex=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, (metric, title, ylabel, color) in zip(axes, metrics):
        steps, values, segments = _metric_arrays(rows, metric)
        smooth = _smooth_series_by_segment(values, segments, smooth_window)
        for start, end, _, fill in stage_regions:
            ax.axvspan(start, end, color=fill, alpha=0.55, lw=0, zorder=0)
        ax.axvline(STAGE_SPLIT_ITER, color="#6b7280", lw=0.8, ls="--", zorder=1)
        ax.axvline(STAGE_SWITCH_ITER, color="#111827", lw=0.9, ls="--", zorder=1)
        ax.plot(steps, values, color=color, lw=0.45, alpha=0.16, zorder=2)
        ax.plot(steps, smooth, color=color, lw=1.45, zorder=3)
        ax.set_title(title, loc="left", pad=3)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, FINAL_ITER)
        ax.grid(axis="y", color="#d7d7d7", lw=0.45, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=2.5, width=0.6)

    top_axis = axes[0]
    transform = blended_transform_factory(top_axis.transData, top_axis.transAxes)
    for start, end, label, _ in stage_regions:
        top_axis.text(
            (start + end) / 2.0,
            1.08,
            label,
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#30343b",
            linespacing=1.1,
        )

    axes[-1].set_xlabel("Training iteration")
    for ax in axes:
        ax.set_xticks([0, 4000, 8000, 11000, 14000])
        ax.set_xticklabels(["0", "4000", "8000", "11000", "14000"])

    handles = [
        plt.Line2D([0], [0], color="#555555", lw=0.45, alpha=0.35, label="raw"),
        plt.Line2D([0], [0], color="#2f5597", lw=1.45, label="smoothed"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="best", handlelength=2.2)

    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(Path(out_dir) / f"{REWARD_TERMS_FIGURE_BASENAME}.{ext}", **kwargs)
    plt.close(fig)


def _write_tensorboard_logs(rows, out_dir):
    from torch.utils.tensorboard import SummaryWriter

    tb_dir = Path(out_dir) / "tensorboard_logs"
    if tb_dir.exists():
        for path in tb_dir.glob("**/events.out.tfevents*"):
            path.unlink()
    run_dir = tb_dir / "two_stage_remapped"
    writer = SummaryWriter(log_dir=str(run_dir))

    tag_map = OrderedDict(
        [
            ("Train/mean_reward", "mean_episode_reward"),
            ("Train/mean_episode_length", "mean_episode_length"),
            ("Episode/terrain_level", "terrain_level"),
            ("Episode/tracking_reward", "tracking_reward"),
            ("Episode/rew_collision", "collision_penalty"),
            ("Episode/rew_foot_clearance", "foot_clearance_reward"),
        ]
    )
    for row in rows:
        step = int(row["global_step"])
        writer.add_scalar("Stage/stage_id", 1 if row["phase"] == "stage1_gt_degraded_gt" else 2, step)
        writer.add_scalar("Stage/nsr_switch_iteration", STAGE_SWITCH_ITER, step)
        for tb_tag, metric in tag_map.items():
            if metric not in row or row[metric] == "":
                continue
            value = float(row[metric])
            if np.isfinite(value):
                writer.add_scalar(tb_tag, value, step)

    writer.add_text(
        "stage_switch_info",
        (
            f"Stage 1: GT / Degraded-GT input, iteration 0-{STAGE_SWITCH_ITER}.  \n"
            f"Stage 2: NSR prediction input, iteration {STAGE_SWITCH_ITER}-{FINAL_ITER}.  \n"
            "NSR is frozen in eval mode; PPO continues adapting the control policy."
        ),
        STAGE_SWITCH_ITER,
    )
    writer.flush()
    writer.close()
    return tb_dir


def _round_cell(value, digits=4):
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _write_markdown_report(path, checkpoint_rows, phase_rows, out_dir):
    lines = [
        "# Go2 两阶段越障策略训练过程分析",
        "",
        "本分析按论文中的新阶段定义重排训练日志：第一阶段为 GT / Degraded-GT 地形输入训练，第二阶段为 NSR prediction 地形输入适配训练。",
        "",
        "## 输出文件",
        "",
        "- `tensorboard_logs/`: 按新阶段定义重映射 step 后的 TensorBoard event logs",
        f"- `{PAPER_FIGURE_BASENAME}.pdf/png/svg`: 论文风格训练过程组合图",
        f"- `{REWARD_TERMS_FIGURE_BASENAME}.pdf/png/svg`: tracking / collision / foot clearance 训练过程组合图",
        "- `training_curves.csv`: 合并后的逐 iteration 指标",
        "- `stage_switch_info.csv`: 阶段切换点信息",
        "- `checkpoint_metrics.csv`: 按阶段分段统计的关键指标表",
        "",
        "TensorBoard 启动命令：",
        "",
        "```bash",
        f"/home/zdd/anaconda3/envs/leggedGym/bin/tensorboard --logdir {out_dir / 'tensorboard_logs'} --port 6006 --host 0.0.0.0",
        "```",
        "",
        "## 阶段切换点",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
    ]
    for row in phase_rows:
        lines.append(f"| {row['item']} | {row['value']} |")

    lines.extend(
        [
            "",
            "## 两阶段训练过程中关键指标均值",
            "",
            "| 训练阶段 | 地形输入 | 指标窗口 | 平均 episode reward | 平均 episode length | 平均 terrain level | Tracking reward | Collision penalty | Foot clearance reward |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in checkpoint_rows:
        lines.append(
            "| {stage} | {terrain} | {window} | {reward} | {length} | {level} | {tracking} | {collision} | {clearance} |".format(
                stage=row["training_stage"],
                terrain=row["terrain_input"],
                window=row["metric_window"],
                reward=_round_cell(row["mean_episode_reward"]),
                length=_round_cell(row["mean_episode_length"], 2),
                level=_round_cell(row["terrain_level"], 3),
                tracking=_round_cell(row["tracking_reward"], 4),
                collision=_round_cell(row["collision_penalty"], 4),
                clearance=_round_cell(row["foot_clearance_reward"], 4),
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Go2 two-stage training process from TensorBoard logs.")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, run_manifest = _build_rows()
    fields = [
        "global_step",
        "raw_step",
        "run",
        "phase",
        "phase_label",
        "terrain_input",
        "mean_episode_reward",
        "mean_episode_length",
        "terrain_level",
        "tracking_reward",
        "collision_penalty",
        "foot_clearance_reward",
    ]
    _write_csv(out_dir / "training_curves.csv", rows, fields)
    _write_csv(out_dir / "run_manifest.csv", run_manifest, ["name", "phase", "terrain_input", "event_file", "step_offset"])

    checkpoint_rows = _make_checkpoint_table(rows)
    checkpoint_fields = [
        "training_stage",
        "terrain_input",
        "nominal_iteration",
        "metric_window",
        "mean_episode_reward",
        "mean_episode_length",
        "terrain_level",
        "tracking_reward",
        "collision_penalty",
        "foot_clearance_reward",
    ]
    _write_csv(out_dir / "checkpoint_metrics.csv", checkpoint_rows, checkpoint_fields)

    phase_rows = [
        {"item": "第一阶段训练 iteration", "value": f"0-{STAGE_SWITCH_ITER}"},
        {"item": "第二阶段训练 iteration", "value": f"{STAGE_SWITCH_ITER}-{FINAL_ITER}"},
        {
            "item": "阶段切换 checkpoint",
            "value": str(ROOT_DIR / "results/training_logs/go2_stage2/Apr16_00-50-38_stage2c_isaacgym/model_11000.pt"),
        },
        {"item": "第二阶段输入变化", "value": "GT / Degraded-GT -> NSR prediction"},
        {"item": "actor / critic 是否继续训练", "value": "是，第二阶段继续 PPO 适配 actor 和 critic"},
        {"item": "NSR 网络是否冻结", "value": "是，NSR eval 模式且 requires_grad=False，仅作为地形输入源"},
        {
            "item": "第二阶段 critic 地形输入",
            "value": "critic 保留 GT terrain observation，actor 使用 NSR prediction",
        },
        {
            "item": "第二阶段最终 checkpoint",
            "value": str(ROOT_DIR / "results/training_logs/go2_stage3/May04_23-49-59_stage3_isaacgym/model_3000.pt"),
        },
    ]
    _write_csv(out_dir / "stage_switch_info.csv", phase_rows, ["item", "value"])

    tensorboard_dir = _write_tensorboard_logs(rows, out_dir)
    _plot_paper_figure(rows, out_dir)
    _plot_reward_terms_figure(rows, out_dir)

    summary = {
        "stage_switch_iter": STAGE_SWITCH_ITER,
        "final_iter": FINAL_ITER,
        "tensorboard_logdir": str(tensorboard_dir),
        "paper_figure": str(out_dir / f"{PAPER_FIGURE_BASENAME}.pdf"),
        "reward_terms_figure": str(out_dir / f"{REWARD_TERMS_FIGURE_BASENAME}.pdf"),
        "notes": [
            "Logs are remapped into paper stages: go2_stage2 -> Stage 1, the new single-stage go2_stage3 run -> Stage 2.",
            "The mean table includes episode reward, episode length, terrain level, tracking reward, collision penalty, and foot clearance reward.",
            "Tracking reward is computed as Episode/rew_tracking_lin_vel + Episode/rew_tracking_ang_vel.",
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _write_markdown_report(out_dir / "report.md", checkpoint_rows, phase_rows, out_dir)

    print(f"Output directory: {out_dir}")
    print(f"TensorBoard logdir: {tensorboard_dir}")
    print(out_dir / f"{PAPER_FIGURE_BASENAME}.pdf")
    print(out_dir / f"{REWARD_TERMS_FIGURE_BASENAME}.pdf")
    print(out_dir / "checkpoint_metrics.csv")
    print(out_dir / "stage_switch_info.csv")


if __name__ == "__main__":
    main()
