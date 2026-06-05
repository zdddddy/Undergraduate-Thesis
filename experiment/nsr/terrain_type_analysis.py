import argparse
import csv
import json
import math
import os
import re
from collections import Counter, OrderedDict, defaultdict

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
DEFAULT_RUN_DIR = os.path.join(RESULTS_ROOT, "experiments", "nsr", "full_val_edgeft_v1")
DEFAULT_PRED_DIR = os.path.join(DEFAULT_RUN_DIR, "predictions")
DEFAULT_OUT_DIR = os.path.join(RESULTS_ROOT, "experiments", "nsr", "terrain_type_analysis_edgeft_v1")


STRONG_STRUCTURE_TYPES = {"stairs_up", "stairs_down", "discrete_obstacles"}


class ErrorStats:
    def __init__(self):
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0
        self.count = 0.0
        self.max_abs = 0.0

    def update(self, err, mask):
        valid = mask > 0.5
        count = int(valid.sum())
        if count <= 0:
            return
        vals = err[valid].astype(np.float64, copy=False)
        abs_vals = np.abs(vals)
        self.sum_abs += float(abs_vals.sum())
        self.sum_sq += float((vals * vals).sum())
        self.sum_signed += float(vals.sum())
        self.count += float(count)
        self.max_abs = max(self.max_abs, float(abs_vals.max()))

    def update_sums(self, sum_abs, sum_sq, sum_signed, count, max_abs):
        if count <= 0:
            return
        self.sum_abs += float(sum_abs)
        self.sum_sq += float(sum_sq)
        self.sum_signed += float(sum_signed)
        self.count += float(count)
        self.max_abs = max(self.max_abs, float(max_abs))

    def as_dict(self):
        if self.count <= 0.0:
            return {
                "count": 0,
                "mae": None,
                "rmse": None,
                "mean_error": None,
                "max_abs_error": None,
            }
        return {
            "count": int(round(self.count)),
            "mae": self.sum_abs / self.count,
            "rmse": math.sqrt(self.sum_sq / self.count),
            "mean_error": self.sum_signed / self.count,
            "max_abs_error": self.max_abs,
        }


def _safe_float(value):
    if value is None:
        return ""
    return float(value)


def _metric_sums(err, mask):
    valid = mask > 0.5
    count = int(valid.sum())
    if count <= 0:
        return {
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            "sum_signed": 0.0,
            "count": 0,
            "max_abs": 0.0,
            "mae": None,
            "rmse": None,
            "mean_error": None,
            "max_abs_error": None,
        }
    vals = err[valid].astype(np.float64, copy=False)
    abs_vals = np.abs(vals)
    sum_abs = float(abs_vals.sum())
    sum_sq = float((vals * vals).sum())
    sum_signed = float(vals.sum())
    return {
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "sum_signed": sum_signed,
        "count": count,
        "max_abs": float(abs_vals.max()),
        "mae": sum_abs / count,
        "rmse": math.sqrt(sum_sq / count),
        "mean_error": sum_signed / count,
        "max_abs_error": float(abs_vals.max()),
    }


def _per_frame_mae(err, mask):
    valid = mask > 0.5
    flat_valid = valid.reshape(valid.shape[0], -1)
    count = flat_valid.sum(axis=1)
    flat_abs = np.abs(err).reshape(err.shape[0], -1)
    sum_abs = (flat_abs * flat_valid).sum(axis=1)
    out = np.full((err.shape[0],), np.nan, dtype=np.float64)
    nz = count > 0
    out[nz] = sum_abs[nz] / count[nz]
    return out


def _build_edge_mask(gt_h, gt_m, edge_thresh):
    valid = gt_m > 0.5
    edge = np.zeros_like(valid, dtype=bool)

    dx = np.abs(gt_h[:, :, 1:] - gt_h[:, :, :-1])
    mx = (dx > float(edge_thresh)) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, 1:] |= mx
    edge[:, :, :-1] |= mx

    dy = np.abs(gt_h[:, 1:, :] - gt_h[:, :-1, :])
    my = (dy > float(edge_thresh)) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, 1:, :] |= my
    edge[:, :-1, :] |= my
    return edge.astype(np.float32)


def _sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "unknown"


def _traj_key_from_source(path):
    path = str(path).replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    env = next((p for p in parts if p.startswith("env_")), None)
    traj = next((p for p in parts if p.startswith("traj_")), None)
    if env and traj:
        return f"{env}/{traj}"
    if traj:
        return traj
    return os.path.dirname(path)


def _load_labels_json(path, label_field):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    labels_by_traj = obj.get("labels_by_traj", {})
    out = {}
    for key, rec in labels_by_traj.items():
        if isinstance(rec, dict):
            out[key] = str(rec.get(label_field, "unknown"))
        else:
            out[key] = "unknown"
    return out


def _source_label(source_file, group_field, labels_by_traj, label_cache):
    source_file = str(source_file)
    traj_key = _traj_key_from_source(source_file)
    if labels_by_traj is not None:
        return labels_by_traj.get(traj_key, "unknown")

    cache_key = (source_file, group_field)
    if cache_key in label_cache:
        return label_cache[cache_key]

    with np.load(source_file, allow_pickle=False) as data:
        if group_field in data.files:
            value = str(np.asarray(data[group_field]).reshape(-1)[0])
        else:
            value = "unknown"
    label_cache[cache_key] = value
    return value


def _sequence_label(source_files, group_field, labels_by_traj, label_cache, per_frame_labels):
    if len(source_files) == 0:
        return "unknown"
    if not per_frame_labels:
        return _source_label(source_files[0], group_field, labels_by_traj, label_cache)

    labels = [_source_label(p, group_field, labels_by_traj, label_cache) for p in source_files]
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def _make_stats_bundle():
    return OrderedDict(
        [
            ("overall", ErrorStats()),
            ("seen", ErrorStats()),
            ("hole", ErrorStats()),
            ("edge", ErrorStats()),
        ]
    )


def _flatten_region(prefix, stats):
    obj = stats.as_dict()
    return {
        f"{prefix}_count": obj["count"],
        f"{prefix}_mae": _safe_float(obj["mae"]),
        f"{prefix}_rmse": _safe_float(obj["rmse"]),
        f"{prefix}_mean_error": _safe_float(obj["mean_error"]),
        f"{prefix}_max_abs_error": _safe_float(obj["max_abs_error"]),
    }


def _distribution_summary(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _box_smooth_3x3(arr):
    p = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    return (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 1:-1]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    ) / 9.0


def _fill_and_smooth(h, valid, smooth_passes=1):
    out = h.astype(np.float32).copy()
    if not np.any(valid):
        return np.zeros_like(out, dtype=np.float32)
    out[~valid] = float(out[valid].mean())
    for _ in range(max(0, int(smooth_passes))):
        out = _box_smooth_3x3(out)
    return out


def _gt_frame_features(h, m, resolution):
    valid = m > 0.5
    n_valid = int(valid.sum())
    if n_valid < 32:
        return {
            "valid": False,
            "slope_mag": 0.0,
            "resid_std": 0.0,
            "z_range": 0.0,
            "edge_ratio": 0.0,
            "spike_ratio": 0.0,
            "z_unique_2cm": 0.0,
        }

    hf = _fill_and_smooth(h, valid, smooth_passes=1)
    height, width = hf.shape
    ys, xs = np.meshgrid(
        (np.arange(height, dtype=np.float32) + 0.5) * float(resolution),
        (np.arange(width, dtype=np.float32) + 0.5) * float(resolution),
        indexing="ij",
    )
    a = np.stack([xs[valid], ys[valid], np.ones(n_valid, dtype=np.float32)], axis=1)
    z = hf[valid]
    beta, *_ = np.linalg.lstsq(a, z, rcond=None)
    plane = beta[0] * xs + beta[1] * ys + beta[2]
    resid = hf - plane
    rv = resid[valid]

    gx = np.gradient(hf, axis=1) / float(resolution)
    gy = np.gradient(hf, axis=0) / float(resolution)
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        "valid": True,
        "slope_mag": float(np.linalg.norm(beta[:2])),
        "resid_std": float(np.std(rv)),
        "z_range": float(np.percentile(z, 95) - np.percentile(z, 5)),
        "edge_ratio": float((grad[valid] > 0.25).mean()),
        "spike_ratio": float((np.abs(rv) > 0.06).mean()),
        "z_unique_2cm": float(np.unique(np.round(rv / 0.02)).size),
    }


def _sequence_gt_features(gt, gt_mask, resolution):
    feats = [_gt_frame_features(gt[i], gt_mask[i], resolution) for i in range(gt.shape[0])]
    valid_feats = [f for f in feats if f["valid"]]
    if not valid_feats:
        return _gt_frame_features(gt[0], gt_mask[0], resolution)
    keys = [k for k in valid_feats[0].keys() if k != "valid"]
    out = {"valid": True}
    for key in keys:
        out[key] = float(np.median([f[key] for f in valid_feats]))
    return out


def _classify_local_gt(gt, gt_mask, resolution, source_hint):
    # Keep stair direction from simulator metadata; local GT alone does not
    # reliably distinguish up vs down after yaw/local-frame transforms.
    if source_hint in {"stairs_up", "stairs_down"}:
        return source_hint

    feat = _sequence_gt_features(gt, gt_mask, resolution)
    if not feat.get("valid", False):
        return source_hint if source_hint else "unknown"

    resid = feat["resid_std"]
    edge_ratio = feat["edge_ratio"]
    spike_ratio = feat["spike_ratio"]
    z_unique = feat["z_unique_2cm"]
    z_range = feat["z_range"]
    slope = feat["slope_mag"]

    obstacle_like = (
        (resid > 0.030 and edge_ratio > 0.055)
        or spike_ratio > 0.045
        or (z_unique >= 8 and edge_ratio > 0.075)
    )
    if obstacle_like:
        return "discrete_obstacles"

    rough_like = resid > 0.035 or edge_ratio > 0.18 or z_range > 0.35 or slope > 0.08
    if rough_like:
        return "rough_slope"
    return "smooth_slope"


def _masked_for_display(arr, mask):
    out = arr.astype(np.float32).copy()
    out[mask <= 0.5] = np.nan
    return out


def _axis_ticks(lo, hi, step):
    vals = np.arange(float(lo), float(hi) + 0.5 * float(step), float(step))
    return np.round(vals, 10)


def _save_visualization(
    out_path,
    label,
    seq_idx,
    frame_idx,
    pred_path,
    map_size,
    height_vmin=-0.8,
    height_vmax=0.4,
    error_vmax=0.08,
    error_tick_step=0.01,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(pred_path, allow_pickle=False) as data:
        pred = data["pred_heightmap"][frame_idx]
        gt = data["gt_heightmap"][frame_idx]
        meas = data["measurement_heightmap"][frame_idx]
        gt_m = data["gt_mask"][frame_idx]
        meas_m = data["measurement_mask"][frame_idx]

    abs_err = np.abs(pred - gt)
    vmin = float(height_vmin)
    vmax = float(height_vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-3
    err_vmax = max(float(error_vmax), 1e-6)

    extent = [-0.5 * map_size, 0.5 * map_size, -0.5 * map_size, 0.5 * map_size]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)
    panels = [
        ("Input observation", _masked_for_display(meas, meas_m), "viridis", vmin, vmax),
        ("Prediction", _masked_for_display(pred, gt_m), "viridis", vmin, vmax),
        ("GT heightmap", _masked_for_display(gt, gt_m), "viridis", vmin, vmax),
        ("Absolute error", _masked_for_display(abs_err, gt_m), "magma", 0.0, err_vmax),
    ]
    err_ticks = _axis_ticks(0.0, error_vmax, error_tick_step)
    for panel_idx, (ax, (title, image, cmap, lo, hi)) in enumerate(zip(axes.reshape(-1), panels)):
        im = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        if panel_idx == 3:
            fig.colorbar(im, ax=ax, fraction=0.046, ticks=err_ticks)
        else:
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{label} | seq={seq_idx} frame={frame_idx}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _select_visualization_candidates(label_candidates, count):
    valid = [c for c in label_candidates if c["overall_mae"] is not None]
    if not valid:
        return []
    valid = sorted(valid, key=lambda c: c["overall_mae"])
    count = min(max(1, int(count)), len(valid))
    if count == 1:
        median = len(valid) // 2
        return [valid[median]]
    ranks = np.linspace(0, len(valid) - 1, num=count)
    selected = []
    used = set()
    for rank in ranks:
        idx = int(round(float(rank)))
        while idx in used and idx + 1 < len(valid):
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        if idx not in used:
            used.add(idx)
            selected.append(valid[idx])
    return selected


def _save_boxplots(out_dir, labels, distributions, strong_types):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    metrics = [("overall", "Overall MAE"), ("seen", "Seen MAE"), ("hole", "Hole MAE")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (key, title) in zip(axes, metrics):
        data = [
            np.asarray(distributions[label]["sequence"][key], dtype=np.float64)
            for label in labels
            if distributions[label]["sequence"][key]
        ]
        plot_labels = [label for label in labels if distributions[label]["sequence"][key]]
        ax.boxplot(data, labels=plot_labels, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel("m")
        ax.tick_params(axis="x", rotation=30)
    fig.savefig(os.path.join(out_dir, "boxplot_sequence_mae_by_terrain.png"), dpi=150)
    plt.close(fig)

    edge_labels = [label for label in labels if label in strong_types and distributions[label]["sequence"]["edge"]]
    if edge_labels:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5), constrained_layout=True)
        data = [np.asarray(distributions[label]["sequence"]["edge"], dtype=np.float64) for label in edge_labels]
        ax.boxplot(data, labels=edge_labels, showfliers=False)
        ax.set_title("Edge MAE on strong-structure terrains")
        ax.set_ylabel("m")
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(os.path.join(out_dir, "boxplot_edge_mae_strong_structure.png"), dpi=150)
        plt.close(fig)


def _save_histograms(out_dir, labels, distributions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n = len(labels)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.8 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, label in zip(axes, labels):
        vals = np.asarray(distributions[label]["frame"]["overall"], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            upper = max(float(np.percentile(vals, 99)), 1e-6)
            ax.hist(vals, bins=50, range=(0.0, upper), color="#3b82f6", alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel("Frame overall MAE (m)")
        ax.set_ylabel("Frame count")
    for ax in axes[n:]:
        ax.axis("off")
    fig.savefig(os.path.join(out_dir, "hist_frame_overall_mae_by_terrain.png"), dpi=150)
    plt.close(fig)


def _write_markdown_report(path, rows, out_dir):
    lines = [
        "# NSR Terrain-Type Reconstruction Analysis",
        "",
        "| Terrain | Seq | Frames | Overall MAE | Overall RMSE | Mean Error | Seen MAE | Hole MAE | Edge MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {sequence_count} | {frame_count} | {overall_mae:.6f} | {overall_rmse:.6f} | "
            "{overall_mean_error:.6f} | {seen_mae:.6f} | {hole_mae:.6f} | {edge_mae:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/boxplot_sequence_mae_by_terrain.png`",
            "- `figures/boxplot_edge_mae_strong_structure.png`",
            "- `figures/hist_frame_overall_mae_by_terrain.png`",
            "- `visualizations/<terrain>_seq*_frame*.png`",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def analyze(args):
    pred_dir = os.path.abspath(args.pred_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pred_files = sorted(
        os.path.join(pred_dir, name)
        for name in os.listdir(pred_dir)
        if name.startswith("seq_") and name.endswith(".npz")
    )
    if args.max_sequences > 0:
        pred_files = pred_files[: args.max_sequences]
    if not pred_files:
        raise RuntimeError(f"No prediction npz files found in {pred_dir}")

    labels_by_traj = _load_labels_json(args.labels_json, args.label_field)
    label_cache = {}
    group_stats = defaultdict(_make_stats_bundle)
    group_counts = defaultdict(lambda: {"sequence_count": 0, "frame_count": 0})
    distributions = defaultdict(
        lambda: {
            "sequence": defaultdict(list),
            "frame": defaultdict(list),
        }
    )
    candidates = defaultdict(list)
    sequence_rows = []
    map_size = None

    print(f"[terrain-analysis] pred_dir={pred_dir}")
    print(f"[terrain-analysis] out_dir={out_dir}")
    print(f"[terrain-analysis] files={len(pred_files)} group_field={args.group_field}")

    for idx, pred_path in enumerate(pred_files):
        with np.load(pred_path, allow_pickle=False) as data:
            pred = data["pred_heightmap"].astype(np.float32)
            gt = data["gt_heightmap"].astype(np.float32)
            gt_mask = data["gt_mask"].astype(np.float32)
            visible = data["visible_mask"].astype(np.float32)
            hole = data["unobserved_mask"].astype(np.float32)
            source_files = [str(x) for x in data["source_files"]]
            seq_idx = int(np.asarray(data["seq_idx"]).reshape(-1)[0])
            resolution = 0.05
            if "resolution" in data.files:
                resolution = float(np.asarray(data["resolution"]).reshape(-1)[0])
            if map_size is None and "map_size" in data.files:
                map_size = float(np.asarray(data["map_size"]).reshape(-1)[0])

        source_hint = _sequence_label(
            source_files,
            args.group_field,
            labels_by_traj,
            label_cache,
            args.per_frame_labels,
        )
        if args.group_mode == "local_gt":
            label = _classify_local_gt(gt, gt_mask, resolution, source_hint)
        else:
            label = source_hint
        err = pred - gt
        edge = _build_edge_mask(gt, gt_mask, args.edge_thresh)
        masks = OrderedDict(
            [
                ("overall", gt_mask),
                ("seen", visible),
                ("hole", hole),
                ("edge", edge),
            ]
        )

        group_counts[label]["sequence_count"] += 1
        group_counts[label]["frame_count"] += int(pred.shape[0])

        seq_row = {
            "seq_idx": seq_idx,
            "terrain": label,
            "source_terrain": source_hint,
            "frames": int(pred.shape[0]),
            "pred_file": pred_path,
            "start_source_file": source_files[0] if source_files else "",
        }
        seq_metrics = {}
        for name, mask in masks.items():
            stats = _metric_sums(err, mask)
            group_stats[label][name].update_sums(
                stats["sum_abs"],
                stats["sum_sq"],
                stats["sum_signed"],
                stats["count"],
                stats["max_abs"],
            )
            seq_metrics[name] = stats
            seq_row[f"{name}_count"] = stats["count"]
            seq_row[f"{name}_mae"] = _safe_float(stats["mae"])
            seq_row[f"{name}_rmse"] = _safe_float(stats["rmse"])
            seq_row[f"{name}_mean_error"] = _safe_float(stats["mean_error"])

            if stats["mae"] is not None:
                distributions[label]["sequence"][name].append(float(stats["mae"]))
            frame_vals = _per_frame_mae(err, mask)
            distributions[label]["frame"][name].extend(frame_vals[np.isfinite(frame_vals)].tolist())

        frame_overall = _per_frame_mae(err, gt_mask)
        finite_frames = np.where(np.isfinite(frame_overall))[0]
        if finite_frames.size > 0:
            median_val = float(np.nanmedian(frame_overall))
            frame_idx = int(finite_frames[np.argmin(np.abs(frame_overall[finite_frames] - median_val))])
        else:
            frame_idx = 0
        candidates[label].append(
            {
                "pred_file": pred_path,
                "seq_idx": seq_idx,
                "frame_idx": frame_idx,
                "overall_mae": seq_metrics["overall"]["mae"],
            }
        )
        sequence_rows.append(seq_row)

        if (idx + 1) % args.log_interval == 0 or idx + 1 == len(pred_files):
            print(f"[terrain-analysis] processed {idx + 1}/{len(pred_files)}")

    labels = sorted(group_counts.keys(), key=lambda x: (-group_counts[x]["sequence_count"], x))
    rows = []
    for label in labels:
        row = {
            "label": label,
            "sequence_count": int(group_counts[label]["sequence_count"]),
            "frame_count": int(group_counts[label]["frame_count"]),
            "strong_structure": bool(label in STRONG_STRUCTURE_TYPES),
        }
        for name in ("overall", "seen", "hole", "edge"):
            row.update(_flatten_region(name, group_stats[label][name]))
            for dist_key, dist_value in _distribution_summary(distributions[label]["sequence"][name]).items():
                row[f"seq_{name}_{dist_key}"] = _safe_float(dist_value)
        rows.append(row)

    csv_path = os.path.join(out_dir, "terrain_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    seq_csv_path = os.path.join(out_dir, "per_sequence_terrain_metrics.csv")
    with open(seq_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sequence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sequence_rows)

    payload = {
        "meta": {
            "pred_dir": pred_dir,
            "out_dir": out_dir,
            "group_mode": args.group_mode,
            "group_field": args.group_field,
            "labels_json": os.path.abspath(args.labels_json) if args.labels_json else "",
            "label_field": args.label_field,
            "edge_thresh": float(args.edge_thresh),
            "strong_structure_types": sorted(STRONG_STRUCTURE_TYPES),
            "sequence_files": len(pred_files),
        },
        "rows": rows,
        "distributions": {
            label: {
                "sequence": {
                    name: _distribution_summary(distributions[label]["sequence"][name])
                    for name in ("overall", "seen", "hole", "edge")
                },
                "frame": {
                    name: _distribution_summary(distributions[label]["frame"][name])
                    for name in ("overall", "seen", "hole", "edge")
                },
            }
            for label in labels
        },
    }
    json_path = os.path.join(out_dir, "terrain_metrics_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    vis_dir = os.path.join(out_dir, "visualizations")
    map_size = 3.2 if map_size is None else map_size
    for label in labels:
        chosen_items = _select_visualization_candidates(candidates[label], args.vis_count)
        for rank, chosen in enumerate(chosen_items):
            vis_path = os.path.join(
                vis_dir,
                (
                    f"{_sanitize_name(label)}_rank{rank:02d}_"
                    f"seq{chosen['seq_idx']:06d}_frame{chosen['frame_idx']:03d}.png"
                ),
            )
            _save_visualization(
                vis_path,
                label,
                chosen["seq_idx"],
                chosen["frame_idx"],
                chosen["pred_file"],
                map_size,
                height_vmin=args.vis_height_vmin,
                height_vmax=args.vis_height_vmax,
                error_vmax=args.vis_error_vmax,
                error_tick_step=args.vis_error_tick_step,
            )

    fig_dir = os.path.join(out_dir, "figures")
    _save_boxplots(fig_dir, labels, distributions, STRONG_STRUCTURE_TYPES)
    _save_histograms(fig_dir, labels, distributions)
    _write_markdown_report(os.path.join(out_dir, "terrain_report.md"), rows, out_dir)

    print("[terrain-analysis] done")
    print(f"[terrain-analysis] csv={csv_path}")
    print(f"[terrain-analysis] json={json_path}")
    print(f"[terrain-analysis] per_sequence={seq_csv_path}")
    print(f"[terrain-analysis] visualizations={vis_dir}")
    print(f"[terrain-analysis] figures={fig_dir}")
    for row in rows:
        print(
            f"  - {row['label']}: seq={row['sequence_count']} frames={row['frame_count']} "
            f"overall={row['overall_mae']:.6f} seen={row['seen_mae']:.6f} "
            f"hole={row['hole_mae']:.6f} edge={row['edge_mae']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NSR reconstruction quality by terrain type from saved prediction npz files."
    )
    parser.add_argument("--pred_dir", type=str, default=DEFAULT_PRED_DIR)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--group_mode", type=str, default="source", choices=["source", "local_gt"])
    parser.add_argument("--group_field", type=str, default="terrain_type", help="Source npz metadata field")
    parser.add_argument("--labels_json", type=str, default="", help="Optional labels_by_traj json override")
    parser.add_argument("--label_field", type=str, default="strict_collect_large_terrain_type")
    parser.add_argument("--per_frame_labels", action="store_true", default=False)
    parser.add_argument("--edge_thresh", type=float, default=0.01)
    parser.add_argument("--vis_count", type=int, default=1, help="Number of visualizations to save per terrain type")
    parser.add_argument("--vis_height_vmin", type=float, default=-0.8)
    parser.add_argument("--vis_height_vmax", type=float, default=0.4)
    parser.add_argument("--vis_error_vmax", type=float, default=0.08)
    parser.add_argument("--vis_error_tick_step", type=float, default=0.01)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
