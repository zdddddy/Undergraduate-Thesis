import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Allow running from nsr_height/tools/terrain_eval while importing core modules in nsr_height/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NSR_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if NSR_ROOT not in sys.path:
    sys.path.insert(0, NSR_ROOT)

from dataset import HeightMapDataset, collate_fn
from model import HeightRecurrentUNet
from visualize_recon import (
    _binarize_valid,
    build_model_input,
    set_batchnorm_use_batch_stats,
    unpack_model_outputs,
    warp_prev_to_current,
)


def _sample_indices(n_total, max_sequences, seed):
    if max_sequences <= 0 or n_total <= max_sequences:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=max_sequences, replace=False).tolist())


def _traj_key_from_seq(ds, seq_idx):
    file_list, start_idx = ds.samples[seq_idx]
    path = file_list[start_idx].replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    env = next((p for p in parts if p.startswith("env_")), None)
    traj = next((p for p in parts if p.startswith("traj_")), None)
    if env and traj:
        return f"{env}/{traj}"
    if traj:
        return traj
    return os.path.dirname(path)


def _load_labels(labels_json, label_field):
    with open(labels_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    by_traj = obj.get("labels_by_traj", {})
    if not isinstance(by_traj, dict):
        raise ValueError(f"invalid labels_by_traj in {labels_json}")
    labels = {}
    for k, rec in by_traj.items():
        if not isinstance(rec, dict):
            labels[k] = "unknown"
            continue
        labels[k] = str(rec.get(label_field, "unknown"))
    return labels


def _acc(stats, label, e_gt, p_gt, e_hole, p_hole, e_seen, p_seen, n_seq):
    s = stats[label]
    s["sum_gt_err"] += float(e_gt)
    s["sum_gt_pix"] += float(p_gt)
    s["sum_hole_err"] += float(e_hole)
    s["sum_hole_pix"] += float(p_hole)
    s["sum_seen_err"] += float(e_seen)
    s["sum_seen_pix"] += float(p_seen)
    s["seq_count"] += int(n_seq)


def _finalize(stats):
    rows = []
    for label, s in stats.items():
        gt_mae = s["sum_gt_err"] / max(s["sum_gt_pix"], 1.0)
        hole_mae = s["sum_hole_err"] / max(s["sum_hole_pix"], 1.0)
        seen_mae = s["sum_seen_err"] / max(s["sum_seen_pix"], 1.0)
        rows.append(
            {
                "label": label,
                "seq_count": int(s["seq_count"]),
                "gt_mae": float(gt_mae),
                "hole_mae": float(hole_mae),
                "seen_mae": float(seen_mae),
                "gt_pixels": int(s["sum_gt_pix"]),
                "hole_pixels": int(s["sum_hole_pix"]),
                "seen_pixels": int(s["sum_seen_pix"]),
            }
        )
    rows.sort(key=lambda x: (-x["seq_count"], x["label"]))
    return rows


def evaluate(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("train_args", {})

    data_dir = args.data_dir if args.data_dir else train_args.get("data_dir")
    if not data_dir:
        raise ValueError("data_dir must be provided via --data_dir or checkpoint train_args")

    seq_len = int(args.seq_len if args.seq_len > 0 else train_args.get("seq_len", 24))
    min_seq_len = int(args.min_seq_len if args.min_seq_len > 0 else train_args.get("min_seq_len", seq_len))
    map_size = float(args.map_size if args.map_size > 0 else train_args.get("map_size", 3.2))
    resolution = float(args.resolution if args.resolution > 0 else train_args.get("resolution", 0.05))
    fill_value = float(train_args.get("fill_value", 0.0))
    gravity_aligned = bool(train_args.get("gravity_aligned", True))

    in_channels = int(train_args.get("in_channels", 4))
    base_channels = int(train_args.get("base_channels", 32))
    use_edge_head = bool(train_args.get("use_edge_head", False))
    norm_type = str(train_args.get("norm_type", "group"))
    group_norm_groups = int(train_args.get("group_norm_groups", 8))

    align_prev = bool(train_args.get("align_prev", True))
    disable_prev = bool(train_args.get("disable_prev", False))
    prev_valid_threshold = float(train_args.get("prev_valid_threshold", 0.5))
    memory_meas_override = bool(train_args.get("memory_meas_override", True))
    residual_from_base = bool(train_args.get("residual_from_base", False))
    residual_scale = float(train_args.get("residual_scale", 0.2))
    residual_tanh = bool(train_args.get("residual_tanh", True))

    labels_by_traj = _load_labels(args.labels_json, args.label_field)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[group-eval] device={device}")
    print(f"[group-eval] label_field={args.label_field}")

    ds = HeightMapDataset(
        data_dir,
        split=args.split,
        min_seq_len=min_seq_len,
        sequence_length=seq_len,
        map_size=map_size,
        resolution=resolution,
        fill_value=fill_value,
        use_local_frame=True,
        gravity_aligned=gravity_aligned,
    )
    if len(ds) == 0:
        raise RuntimeError("empty dataset")

    seq_indices = _sample_indices(len(ds), args.max_sequences, args.seed)
    seq_labels = []
    kept_indices = []
    for seq_idx in seq_indices:
        key = _traj_key_from_seq(ds, seq_idx)
        label = labels_by_traj.get(key, "unknown")
        if args.exclude_unknown and label == "unknown":
            continue
        kept_indices.append(seq_idx)
        seq_labels.append(label)
    if not kept_indices:
        raise RuntimeError("no sequences left after filtering")

    ds_view = Subset(ds, kept_indices)
    loader = DataLoader(
        ds_view,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = HeightRecurrentUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        out_channels=1,
        use_edge_head=use_edge_head,
        norm_type=norm_type,
        group_norm_groups=group_norm_groups,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(device)
    set_batchnorm_use_batch_stats(model)
    model.eval()

    stats = defaultdict(
        lambda: {
            "sum_gt_err": 0.0,
            "sum_gt_pix": 0.0,
            "sum_hole_err": 0.0,
            "sum_hole_pix": 0.0,
            "sum_seen_err": 0.0,
            "sum_seen_pix": 0.0,
            "seq_count": 0,
        }
    )

    processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            in_height = batch["in_height"].to(device)
            in_mask = batch["in_mask"].to(device)
            gt_height = batch["gt_height"].to(device)
            gt_mask = batch["gt_mask"].to(device)
            pose7 = batch["pose7"].to(device)

            bsz, timesteps = in_height.shape[0], in_height.shape[1]
            local_labels = seq_labels[processed:processed + bsz]

            prev_pred = None
            prev_pose = None
            prev_valid_state = None

            seq_gt_err = torch.zeros(bsz, device=device)
            seq_gt_pix = torch.zeros(bsz, device=device)
            seq_hole_err = torch.zeros(bsz, device=device)
            seq_hole_pix = torch.zeros(bsz, device=device)
            seq_seen_err = torch.zeros(bsz, device=device)
            seq_seen_pix = torch.zeros(bsz, device=device)

            for t in range(timesteps):
                meas_h = in_height[:, t]
                meas_m = in_mask[:, t]
                target_h = gt_height[:, t]
                target_m = gt_mask[:, t]
                cur_pose = pose7[:, t]

                if disable_prev or prev_pred is None:
                    prev_in = meas_h
                    prev_in_valid = meas_m
                else:
                    if align_prev:
                        prev_valid_for_warp = prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                        prev_in, prev_in_valid = warp_prev_to_current(
                            prev_pred,
                            prev_pose,
                            cur_pose,
                            map_size,
                            prev_valid=prev_valid_for_warp,
                            gravity_aligned=gravity_aligned,
                        )
                    else:
                        prev_in = prev_pred
                        prev_in_valid = prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                prev_in_valid = _binarize_valid(prev_in_valid, prev_valid_threshold)

                model_out = model(build_model_input(meas_h, meas_m, prev_in, prev_in_valid, in_channels))
                pred_core, _ = unpack_model_outputs(model_out)

                if residual_from_base:
                    prev_base = torch.where(
                        prev_in_valid > 0.5,
                        prev_in,
                        torch.full_like(prev_in, float(fill_value)),
                    )
                    base_h = torch.where(meas_m > 0.5, meas_h, prev_base)
                    residual = pred_core
                    if residual_tanh:
                        residual = residual_scale * torch.tanh(residual)
                    else:
                        residual = residual_scale * residual
                    pred_h = base_h + residual
                else:
                    pred_h = pred_core
                pred_h = torch.nan_to_num(pred_h, nan=float(fill_value), posinf=1e3, neginf=-1e3)

                hole = ((target_m > 0.5) & (meas_m <= 0.5)).float()
                seen = ((target_m > 0.5) & (meas_m > 0.5)).float()
                err = torch.abs(pred_h - target_h)

                seq_gt_err += (err * target_m).sum(dim=(1, 2, 3))
                seq_gt_pix += target_m.sum(dim=(1, 2, 3))
                seq_hole_err += (err * hole).sum(dim=(1, 2, 3))
                seq_hole_pix += hole.sum(dim=(1, 2, 3))
                seq_seen_err += (err * seen).sum(dim=(1, 2, 3))
                seq_seen_pix += seen.sum(dim=(1, 2, 3))

                if memory_meas_override:
                    prev_pred = torch.where(meas_m > 0.5, meas_h, pred_h)
                else:
                    prev_pred = pred_h
                prev_pose = cur_pose
                prev_valid_state = torch.maximum(meas_m, prev_in_valid)

            seq_gt_err_np = seq_gt_err.detach().cpu().numpy()
            seq_gt_pix_np = seq_gt_pix.detach().cpu().numpy()
            seq_hole_err_np = seq_hole_err.detach().cpu().numpy()
            seq_hole_pix_np = seq_hole_pix.detach().cpu().numpy()
            seq_seen_err_np = seq_seen_err.detach().cpu().numpy()
            seq_seen_pix_np = seq_seen_pix.detach().cpu().numpy()

            for b in range(bsz):
                label = local_labels[b]
                _acc(
                    stats,
                    label,
                    seq_gt_err_np[b],
                    seq_gt_pix_np[b],
                    seq_hole_err_np[b],
                    seq_hole_pix_np[b],
                    seq_seen_err_np[b],
                    seq_seen_pix_np[b],
                    1,
                )
                _acc(
                    stats,
                    "__overall__",
                    seq_gt_err_np[b],
                    seq_gt_pix_np[b],
                    seq_hole_err_np[b],
                    seq_hole_pix_np[b],
                    seq_seen_err_np[b],
                    seq_seen_pix_np[b],
                    1,
                )
            processed += bsz
            if batch_idx % 20 == 0:
                print(f"[group-eval] batch {batch_idx + 1}/{len(loader)}")

    rows = _finalize(stats)
    overall = next((r for r in rows if r["label"] == "__overall__"), None)
    if overall is None:
        raise RuntimeError("overall row missing")

    print(
        "[group-eval] overall "
        f"seq={overall['seq_count']} "
        f"hole_mae={overall['hole_mae']:.6f} "
        f"seen_mae={overall['seen_mae']:.6f} "
        f"gt_mae={overall['gt_mae']:.6f}"
    )
    print("[group-eval] per-group:")
    for row in rows:
        if row["label"] == "__overall__":
            continue
        print(
            f"  - {row['label']}: seq={row['seq_count']} "
            f"hole={row['hole_mae']:.6f} seen={row['seen_mae']:.6f} gt={row['gt_mae']:.6f}"
        )

    if args.out_json:
        payload = {
            "meta": {
                "ckpt": os.path.abspath(args.ckpt),
                "data_dir": os.path.abspath(data_dir),
                "split": args.split,
                "label_field": args.label_field,
                "labels_json": os.path.abspath(args.labels_json),
                "max_sequences": int(args.max_sequences),
                "seed": int(args.seed),
                "exclude_unknown": bool(args.exclude_unknown),
            },
            "rows": rows,
        }
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        print(f"[group-eval] saved: {args.out_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="optional override; default from ckpt train_args")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--labels_json", type=str, required=True)
    parser.add_argument(
        "--label_field",
        type=str,
        default="strict_terrain_type",
        help="field name inside labels_by_traj records, e.g. strict_terrain_type or strict_collect_large_terrain_type",
    )
    parser.add_argument("--out_json", type=str, default="")

    parser.add_argument("--max_sequences", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260307)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--exclude_unknown", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)

    parser.add_argument("--seq_len", type=int, default=0, help="0 means use ckpt train_args")
    parser.add_argument("--min_seq_len", type=int, default=0, help="0 means use ckpt train_args")
    parser.add_argument("--map_size", type=float, default=0.0, help="<=0 means use ckpt train_args")
    parser.add_argument("--resolution", type=float, default=0.0, help="<=0 means use ckpt train_args")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
