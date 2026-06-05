import argparse
import csv
import os
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_RUN_DIR = ROOT_DIR / "results" / "experiments" / "nsr" / "full_val_edgeft_v1"
DEFAULT_PRED_DIR = DEFAULT_RUN_DIR / "predictions"
DEFAULT_OUT_DIR = ROOT_DIR / "results" / "experiments" / "nsr" / "temporal_blind_completion_long_no_box"


def _masked_for_display(arr, mask):
    out = arr.astype(np.float32).copy()
    out[mask <= 0.5] = np.nan
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
    return edge


def _dilate_bool(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            src_y0 = max(0, dy)
            src_y1 = height + min(0, dy)
            src_x0 = max(0, dx)
            src_x1 = width + min(0, dx)
            dst_y0 = max(0, -dy)
            dst_y1 = height - max(0, dy)
            dst_x0 = max(0, -dx)
            dst_x1 = width - max(0, dx)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out


def _retained_blind_edge_masks(data, edge_thresh, good_error_thresh, dilation_radius):
    gt = data["gt_heightmap"]
    pred = data["pred_heightmap"]
    gt_mask = data["gt_mask"]
    meas_mask = data["measurement_mask"]
    unobserved = data["unobserved_mask"] > 0.5

    edge = _build_edge_mask(gt, gt_mask, edge_thresh)
    err = np.abs(pred - gt)
    retained = np.zeros_like(edge, dtype=bool)
    prev_seen_edge = np.zeros(edge.shape[1:], dtype=bool)

    for frame_idx in range(edge.shape[0]):
        current_blind_good_edge = edge[frame_idx] & unobserved[frame_idx] & (err[frame_idx] <= good_error_thresh)
        retained[frame_idx] = current_blind_good_edge & _dilate_bool(prev_seen_edge, dilation_radius)
        prev_seen_edge |= edge[frame_idx] & (meas_mask[frame_idx] > 0.5)

    return retained, edge, err


def _best_bbox(mask, window_size=16, pad=3):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    height, width = mask.shape
    window_size = min(int(window_size), height, width)
    best = None
    best_count = -1
    for y0 in range(0, height - window_size + 1):
        y1 = y0 + window_size
        for x0 in range(0, width - window_size + 1):
            x1 = x0 + window_size
            count = int(mask[y0:y1, x0:x1].sum())
            if count > best_count:
                best_count = count
                best = (x0, y0, x1, y1)

    x0, y0, x1, y1 = best
    x0 = max(0, x0 - int(pad))
    y0 = max(0, y0 - int(pad))
    x1 = min(width, x1 + int(pad))
    y1 = min(height, y1 + int(pad))
    return x0, y0, x1, y1


def _bbox_to_xy(bbox, map_size, width, height):
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    resolution_x = float(map_size) / float(width)
    resolution_y = float(map_size) / float(height)
    left = -0.5 * float(map_size) + x0 * resolution_x
    bottom = -0.5 * float(map_size) + y0 * resolution_y
    rect_w = (x1 - x0) * resolution_x
    rect_h = (y1 - y0) * resolution_y
    return left, bottom, rect_w, rect_h


def _plot_temporal_sequence(
    out_path,
    label,
    seq_idx,
    frames,
    pred_path,
    height_vmin,
    height_vmax,
    edge_thresh,
    good_error_thresh,
    dilation_radius,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(pred_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "source_files"}
        source_files = np.asarray(data["source_files"]).astype(str)

    retained, edge, err = _retained_blind_edge_masks(
        arrays,
        edge_thresh=edge_thresh,
        good_error_thresh=good_error_thresh,
        dilation_radius=dilation_radius,
    )
    map_size = float(np.asarray(arrays.get("map_size", 3.2)).reshape(-1)[0])
    gt = arrays["gt_heightmap"]
    pred = arrays["pred_heightmap"]
    meas = arrays["measurement_heightmap"]
    gt_mask = arrays["gt_mask"]
    meas_mask = arrays["measurement_mask"]

    frames = [int(frame) for frame in frames]

    extent = [-0.5 * map_size, 0.5 * map_size, -0.5 * map_size, 0.5 * map_size]
    fig, axes = plt.subplots(
        3,
        len(frames),
        figsize=(2.25 * len(frames), 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(frames) == 1:
        axes = axes.reshape(3, 1)

    row_specs = [
        ("Input observation", meas, meas_mask),
        ("Prediction", pred, gt_mask),
        ("GT heightmap", gt, gt_mask),
    ]

    image_handle = None
    for col_idx, frame_idx in enumerate(frames):
        for row_idx, (row_name, values, mask) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            image_handle = ax.imshow(
                _masked_for_display(values[frame_idx], mask[frame_idx]),
                origin="lower",
                extent=extent,
                interpolation="nearest",
                cmap="viridis",
                vmin=height_vmin,
                vmax=height_vmax,
            )
            if row_idx == 0:
                ax.set_title(f"frame {frame_idx}", fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=15)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=10)

    for ax in axes[-1, :]:
        ax.set_xlabel("x (m)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    cbar = fig.colorbar(image_handle, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label("height (m)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    per_frame_rows = []
    for frame_idx in frames:
        blind_edge = edge[frame_idx] & (arrays["unobserved_mask"][frame_idx] > 0.5)
        retained_mask = retained[frame_idx]
        hole = arrays["unobserved_mask"][frame_idx] > 0.5
        per_frame_rows.append(
            {
                "terrain": label,
                "seq_idx": int(seq_idx),
                "frame_idx": int(frame_idx),
                "source_file": source_files[frame_idx],
                "blind_edge_pixels": int(blind_edge.sum()),
                "retained_blind_edge_pixels": int(retained_mask.sum()),
                "hole_mae": float(err[frame_idx][hole].mean()) if np.any(hole) else float("nan"),
            }
        )
    return per_frame_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot temporal NSR blind-area completion examples from saved validation predictions."
    )
    parser.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--height_vmin", type=float, default=-0.8)
    parser.add_argument("--height_vmax", type=float, default=0.4)
    parser.add_argument("--edge_thresh", type=float, default=0.05)
    parser.add_argument("--good_error_thresh", type=float, default=0.03)
    parser.add_argument("--dilation_radius", type=int, default=2)
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help=(
            "Example spec: terrain:seq_idx:start_frame:end_frame[:step]. "
            "Can be repeated."
        ),
    )
    return parser.parse_args()


def _parse_example(spec):
    parts = str(spec).split(":")
    if len(parts) not in {4, 5}:
        raise ValueError(f"Invalid --example '{spec}', expected terrain:seq:start:end[:step]")
    terrain, seq_idx, start, end = parts[:4]
    step = int(parts[4]) if len(parts) == 5 else 1
    start = int(start)
    end = int(end)
    if step <= 0:
        raise ValueError(f"Invalid frame step in --example '{spec}'")
    if end < start:
        raise ValueError(f"Invalid frame range in --example '{spec}'")
    frames = list(range(start, end + 1, step))
    if frames[-1] != end:
        frames.append(end)
    return terrain, int(seq_idx), frames


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = [_parse_example(spec) for spec in args.example]
    if not examples:
        examples = [
            ("stairs_down", 4410, [0, 4, 8, 12, 16, 20, 23]),
            ("discrete_obstacles", 4943, [0, 4, 8, 12, 16, 20, 23]),
        ]

    manifest_rows = []
    frame_rows = []
    for terrain, seq_idx, frames in examples:
        pred_path = pred_dir / f"seq_{seq_idx:06d}.npz"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        out_name = (
            f"{terrain}_seq{seq_idx:06d}_frames{frames[0]:03d}-{frames[-1]:03d}"
            f"_n{len(frames):02d}_temporal_completion.png"
        )
        out_path = out_dir / out_name
        rows = _plot_temporal_sequence(
            str(out_path),
            terrain,
            seq_idx,
            frames,
            str(pred_path),
            height_vmin=args.height_vmin,
            height_vmax=args.height_vmax,
            edge_thresh=args.edge_thresh,
            good_error_thresh=args.good_error_thresh,
            dilation_radius=args.dilation_radius,
        )
        frame_rows.extend(rows)
        manifest_rows.append(
            {
                "terrain": terrain,
                "seq_idx": seq_idx,
                "start_frame": frames[0],
                "end_frame": frames[-1],
                "frame_indices": " ".join(str(frame) for frame in frames),
                "image_file": str(out_path),
                "pred_file": str(pred_path),
            }
        )

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "terrain",
                "seq_idx",
                "start_frame",
                "end_frame",
                "frame_indices",
                "image_file",
                "pred_file",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(out_dir / "frame_blind_completion_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "terrain",
                "seq_idx",
                "frame_idx",
                "source_file",
                "blind_edge_pixels",
                "retained_blind_edge_pixels",
                "hole_mae",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_rows)

    report = [
        "# Temporal Blind-Area Completion",
        "",
        "Height color scale is fixed to [-0.8, 0.4] m. White cells in the input row are currently unobserved.",
        "Each column samples a later frame from the same 24-frame sequence window to show a longer motion span.",
        "",
        "| Terrain | Seq | Frame indices | Figure |",
        "|---|---:|---|---|",
    ]
    for row in manifest_rows:
        report.append(
            f"| {row['terrain']} | {row['seq_idx']} | {row['frame_indices']} | "
            f"`{os.path.basename(row['image_file'])}` |"
        )
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(f"Output directory: {out_dir}")
    for row in manifest_rows:
        print(row["image_file"])


if __name__ == "__main__":
    main()
