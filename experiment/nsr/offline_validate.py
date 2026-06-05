import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
NSR_ROOT = os.path.join(REPO_ROOT, "nsr_height")
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
for path in (NSR_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset import HeightMapDataset, collate_fn
from model import HeightRecurrentUNet
from visualize_recon import (
    _binarize_valid,
    build_model_input,
    set_batchnorm_use_batch_stats,
    unpack_model_outputs,
    warp_prev_to_current,
)


DEFAULT_CKPT = os.path.join(
    RESULTS_ROOT,
    "nsr",
    "checkpoints",
    "nsr_go2_v3_edgeft_v1",
    "checkpoint_best_hole_mae.pth",
)


class ErrorStats:
    def __init__(self):
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0
        self.count = 0.0
        self.max_abs = 0.0

    def update_torch(self, signed_err, mask):
        mask = (mask > 0.5).float()
        count = float(mask.sum().item())
        if count <= 0.0:
            return

        abs_err = torch.abs(signed_err)
        self.sum_abs += float((abs_err * mask).sum().item())
        self.sum_sq += float(((signed_err * signed_err) * mask).sum().item())
        self.sum_signed += float((signed_err * mask).sum().item())
        self.count += count
        masked_max = float(abs_err[mask > 0.5].max().item())
        if masked_max > self.max_abs:
            self.max_abs = masked_max

    def update_values(self, sum_abs, sum_sq, sum_signed, count, max_abs):
        count = float(count)
        if count <= 0.0:
            return
        self.sum_abs += float(sum_abs)
        self.sum_sq += float(sum_sq)
        self.sum_signed += float(sum_signed)
        self.count += count
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


def _region_stats_from_tensors(signed_err, mask):
    mask = (mask > 0.5).float()
    count = float(mask.sum().item())
    if count <= 0.0:
        return {
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            "sum_signed": 0.0,
            "count": 0.0,
            "max_abs": 0.0,
            "mae": None,
            "rmse": None,
            "mean_error": None,
            "max_abs_error": None,
        }

    abs_err = torch.abs(signed_err)
    sum_abs = float((abs_err * mask).sum().item())
    sum_sq = float(((signed_err * signed_err) * mask).sum().item())
    sum_signed = float((signed_err * mask).sum().item())
    max_abs = float(abs_err[mask > 0.5].max().item())
    return {
        "sum_abs": sum_abs,
        "sum_sq": sum_sq,
        "sum_signed": sum_signed,
        "count": count,
        "max_abs": max_abs,
        "mae": sum_abs / count,
        "rmse": math.sqrt(sum_sq / count),
        "mean_error": sum_signed / count,
        "max_abs_error": max_abs,
    }


def _build_edge_mask_torch(height, mask, edge_thresh):
    valid = mask > 0.5
    edge = torch.zeros_like(mask, dtype=torch.bool)
    dx = torch.abs(height[:, :, :, 1:] - height[:, :, :, :-1])
    mx = (dx > float(edge_thresh)) & valid[:, :, :, 1:] & valid[:, :, :, :-1]
    edge[:, :, :, 1:] |= mx
    edge[:, :, :, :-1] |= mx
    dy = torch.abs(height[:, :, 1:, :] - height[:, :, :-1, :])
    my = (dy > float(edge_thresh)) & valid[:, :, 1:, :] & valid[:, :, :-1, :]
    edge[:, :, 1:, :] |= my
    edge[:, :, :-1, :] |= my
    return edge.float()


def _flatten_metric(prefix, stats):
    return {
        f"{prefix}_count": int(round(stats["count"])),
        f"{prefix}_mae": stats["mae"],
        f"{prefix}_rmse": stats["rmse"],
        f"{prefix}_mean_error": stats["mean_error"],
        f"{prefix}_max_abs_error": stats["max_abs_error"],
    }


def _select_indices(total, start_seq, max_sequences, random_sample, seed):
    start_seq = max(0, int(start_seq))
    if start_seq >= total:
        return []
    candidates = np.arange(start_seq, total, dtype=np.int64)
    if max_sequences > 0 and max_sequences < len(candidates):
        if random_sample:
            rng = np.random.default_rng(seed)
            candidates = np.sort(rng.choice(candidates, size=max_sequences, replace=False))
        else:
            candidates = candidates[:max_sequences]
    return [int(x) for x in candidates.tolist()]


def _get_train_args(ckpt):
    train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    if train_args is None:
        return {}
    if not isinstance(train_args, dict):
        return vars(train_args)
    return train_args


def _resolve_eval_config(args, train_args):
    data_dir = args.data_dir or train_args.get("data_dir", "")
    if not data_dir:
        raise ValueError("Missing data_dir. Pass --data_dir or use a checkpoint with train_args['data_dir'].")

    seq_len = int(args.seq_len if args.seq_len > 0 else train_args.get("seq_len", 24))
    min_seq_len = int(args.min_seq_len if args.min_seq_len > 0 else train_args.get("min_seq_len", seq_len))

    world_frame = args.world_frame
    if world_frame is None:
        world_frame = bool(train_args.get("world_frame", False))

    gravity_aligned = args.gravity_aligned
    if gravity_aligned is None:
        gravity_aligned = bool(train_args.get("gravity_aligned", True))

    align_prev = args.align_prev
    if align_prev is None:
        align_prev = bool(train_args.get("align_prev", True))

    disable_prev = args.disable_prev
    if disable_prev is None:
        disable_prev = bool(train_args.get("disable_prev", False))

    return {
        "data_dir": data_dir,
        "split": args.split,
        "seq_len": seq_len,
        "min_seq_len": min_seq_len,
        "map_size": float(args.map_size if args.map_size > 0 else train_args.get("map_size", 3.2)),
        "resolution": float(args.resolution if args.resolution > 0 else train_args.get("resolution", 0.05)),
        "fill_value": float(args.fill_value if args.fill_value is not None else train_args.get("fill_value", 0.0)),
        "world_frame": bool(world_frame),
        "gravity_aligned": bool(gravity_aligned),
        "in_channels": int(args.in_channels if args.in_channels > 0 else train_args.get("in_channels", 4)),
        "base_channels": int(args.base_channels if args.base_channels > 0 else train_args.get("base_channels", 32)),
        "use_edge_head": bool(train_args.get("use_edge_head", False)),
        "norm_type": str(train_args.get("norm_type", "group")),
        "group_norm_groups": int(train_args.get("group_norm_groups", 8)),
        "align_prev": bool(align_prev),
        "disable_prev": bool(disable_prev),
        "prev_valid_threshold": float(train_args.get("prev_valid_threshold", 0.5)),
        "memory_meas_override": bool(train_args.get("memory_meas_override", True)),
        "residual_from_base": bool(train_args.get("residual_from_base", False)),
        "residual_scale": float(train_args.get("residual_scale", 0.2)),
        "residual_tanh": bool(train_args.get("residual_tanh", True)),
    }


def _source_files_for_seq(dataset, seq_idx, seq_len):
    file_list, start_idx = dataset.samples[seq_idx]
    return file_list[start_idx:start_idx + seq_len]


def _make_default_out_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_ROOT, "experiments", "nsr", f"offline_val_{stamp}")


def _jsonable_config(config):
    out = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save_sequence_npz(path, arrays, compress):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    saver = np.savez_compressed if compress else np.savez
    saver(path, **arrays)


def _build_model(config, ckpt, device, allow_partial_load):
    model = HeightRecurrentUNet(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        out_channels=1,
        use_edge_head=config["use_edge_head"],
        norm_type=config["norm_type"],
        group_norm_groups=config["group_norm_groups"],
    )
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing or unexpected:
        msg = (
            f"checkpoint key mismatch: missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_example={missing[:3]} unexpected_example={unexpected[:3]}"
        )
        if allow_partial_load:
            print(f"[offline-val] warning: {msg}")
        else:
            raise RuntimeError(msg + ". Pass --allow_partial_load only for debugging.")
    else:
        print("[offline-val] checkpoint loaded with full key match")

    model.to(device)
    return model


def run_offline_validation(args):
    ckpt_path = os.path.abspath(args.ckpt)
    out_dir = os.path.abspath(args.out_dir or _make_default_out_dir())
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = _load_checkpoint(ckpt_path)
    train_args = _get_train_args(ckpt)
    config = _resolve_eval_config(args, train_args)

    print(f"[offline-val] device={device}")
    print(f"[offline-val] checkpoint={ckpt_path}")
    print(f"[offline-val] data_dir={config['data_dir']} split={config['split']}")
    print(f"[offline-val] out_dir={out_dir}")

    dataset = HeightMapDataset(
        config["data_dir"],
        split=config["split"],
        min_seq_len=config["min_seq_len"],
        sequence_length=config["seq_len"],
        map_size=config["map_size"],
        resolution=config["resolution"],
        fill_value=config["fill_value"],
        use_local_frame=not config["world_frame"],
        gravity_aligned=config["gravity_aligned"],
    )
    if len(dataset) == 0:
        raise RuntimeError("empty validation dataset")

    seq_indices = _select_indices(
        len(dataset),
        args.start_seq,
        args.max_sequences,
        args.random_sample,
        args.seed,
    )
    if not seq_indices:
        raise RuntimeError(f"no sequences selected from dataset size={len(dataset)}")

    subset = Subset(dataset, seq_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = _build_model(config, ckpt, device, args.allow_partial_load)
    if args.bn_use_batch_stats:
        set_batchnorm_use_batch_stats(model)
    model.eval()

    pred_dir = os.path.join(out_dir, "predictions")
    if args.save_predictions:
        os.makedirs(pred_dir, exist_ok=True)

    meta = {
        "ckpt": ckpt_path,
        "ckpt_epoch": ckpt.get("epoch", None) if isinstance(ckpt, dict) else None,
        "best_val_hole_mae": ckpt.get("best_val_hole_mae", None) if isinstance(ckpt, dict) else None,
        "selected_sequences": len(seq_indices),
        "dataset_sequences": len(dataset),
        "start_seq": int(args.start_seq),
        "max_sequences": int(args.max_sequences),
        "random_sample": bool(args.random_sample),
        "seed": int(args.seed),
        "save_predictions": bool(args.save_predictions),
        "save_frame_stride": int(args.save_frame_stride),
        "edge_thresh": float(args.edge_thresh),
        "config": _jsonable_config(config),
    }
    _write_json(os.path.join(out_dir, "run_config.json"), meta)

    aggregate = OrderedDict(
        [
            ("overall", ErrorStats()),
            ("visible", ErrorStats()),
            ("unobserved", ErrorStats()),
            ("edge", ErrorStats()),
        ]
    )

    frame_csv_path = os.path.join(out_dir, "per_frame_metrics.csv")
    seq_csv_path = os.path.join(out_dir, "per_sequence_metrics.csv")
    frame_csv = open(frame_csv_path, "w", newline="", encoding="utf-8")
    seq_csv = open(seq_csv_path, "w", newline="", encoding="utf-8")
    frame_writer = None
    seq_writer = None

    processed = 0
    total_frames = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                in_height = batch["in_height"].to(device)
                in_mask = batch["in_mask"].to(device)
                gt_height = batch["gt_height"].to(device)
                gt_mask = batch["gt_mask"].to(device)
                pose7 = batch["pose7"].to(device)

                batch_size = in_height.shape[0]
                timesteps = in_height.shape[1]
                original_indices = seq_indices[processed:processed + batch_size]

                seq_stats = [
                    OrderedDict(
                        [
                            ("overall", ErrorStats()),
                            ("visible", ErrorStats()),
                            ("unobserved", ErrorStats()),
                            ("edge", ErrorStats()),
                        ]
                    )
                    for _ in range(batch_size)
                ]
                if args.save_predictions:
                    seq_pred = [[] for _ in range(batch_size)]
                    seq_gt = [[] for _ in range(batch_size)]
                    seq_meas = [[] for _ in range(batch_size)]
                    seq_gt_mask = [[] for _ in range(batch_size)]
                    seq_meas_mask = [[] for _ in range(batch_size)]
                    seq_visible = [[] for _ in range(batch_size)]
                    seq_unobserved = [[] for _ in range(batch_size)]
                    seq_abs_err = [[] for _ in range(batch_size)]

                prev_pred = None
                prev_pose = None
                prev_valid_state = None

                for t in range(timesteps):
                    meas_h = in_height[:, t]
                    meas_m = in_mask[:, t]
                    target_h = gt_height[:, t]
                    target_m = gt_mask[:, t]
                    cur_pose = pose7[:, t]

                    if config["disable_prev"] or prev_pred is None:
                        prev_in = meas_h
                        prev_in_valid = meas_m
                    else:
                        if config["align_prev"]:
                            prev_valid_for_warp = (
                                prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                            )
                            prev_in, prev_in_valid = warp_prev_to_current(
                                prev_pred,
                                prev_pose,
                                cur_pose,
                                config["map_size"],
                                prev_valid=prev_valid_for_warp,
                                gravity_aligned=config["gravity_aligned"],
                            )
                        else:
                            prev_in = prev_pred
                            prev_in_valid = (
                                prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                            )
                    prev_in_valid = _binarize_valid(prev_in_valid, config["prev_valid_threshold"])

                    model_input = build_model_input(
                        meas_h,
                        meas_m,
                        prev_in,
                        prev_in_valid,
                        config["in_channels"],
                    )
                    model_out = model(model_input)
                    pred_core, _pred_edge_logits = unpack_model_outputs(model_out)
                    if config["residual_from_base"]:
                        prev_base = torch.where(
                            prev_in_valid > 0.5,
                            prev_in,
                            torch.full_like(prev_in, float(config["fill_value"])),
                        )
                        base_h = torch.where(meas_m > 0.5, meas_h, prev_base)
                        residual = pred_core
                        if config["residual_tanh"]:
                            residual = config["residual_scale"] * torch.tanh(residual)
                        else:
                            residual = config["residual_scale"] * residual
                        pred_h = base_h + residual
                    else:
                        pred_h = pred_core
                    pred_h = torch.nan_to_num(
                        pred_h,
                        nan=float(config["fill_value"]),
                        posinf=1e3,
                        neginf=-1e3,
                    )

                    visible_m = ((target_m > 0.5) & (meas_m > 0.5)).float()
                    unobserved_m = ((target_m > 0.5) & (meas_m <= 0.5)).float()
                    edge_m = _build_edge_mask_torch(target_h, target_m, args.edge_thresh)
                    signed_err = pred_h - target_h

                    region_masks = OrderedDict(
                        [
                            ("overall", target_m),
                            ("visible", visible_m),
                            ("unobserved", unobserved_m),
                            ("edge", edge_m),
                        ]
                    )

                    if args.save_predictions and (t % args.save_frame_stride == 0):
                        pred_np = pred_h[:, 0].detach().cpu().numpy().astype(np.float32)
                        gt_np = target_h[:, 0].detach().cpu().numpy().astype(np.float32)
                        meas_np = meas_h[:, 0].detach().cpu().numpy().astype(np.float32)
                        gt_mask_np = target_m[:, 0].detach().cpu().numpy().astype(np.float32)
                        meas_mask_np = meas_m[:, 0].detach().cpu().numpy().astype(np.float32)
                        visible_np = visible_m[:, 0].detach().cpu().numpy().astype(np.float32)
                        unobserved_np = unobserved_m[:, 0].detach().cpu().numpy().astype(np.float32)
                        abs_err_np = torch.abs(signed_err[:, 0]).detach().cpu().numpy().astype(np.float32)
                        for b in range(batch_size):
                            seq_pred[b].append(pred_np[b])
                            seq_gt[b].append(gt_np[b])
                            seq_meas[b].append(meas_np[b])
                            seq_gt_mask[b].append(gt_mask_np[b])
                            seq_meas_mask[b].append(meas_mask_np[b])
                            seq_visible[b].append(visible_np[b])
                            seq_unobserved[b].append(unobserved_np[b])
                            seq_abs_err[b].append(abs_err_np[b])

                    for b in range(batch_size):
                        row = {
                            "seq_idx": original_indices[b],
                            "frame": t,
                            "source_file": _source_files_for_seq(dataset, original_indices[b], timesteps)[t],
                        }
                        for name, mask in region_masks.items():
                            stats = _region_stats_from_tensors(
                                signed_err[b:b + 1],
                                mask[b:b + 1],
                            )
                            aggregate[name].update_values(
                                stats["sum_abs"],
                                stats["sum_sq"],
                                stats["sum_signed"],
                                stats["count"],
                                stats["max_abs"],
                            )
                            seq_stats[b][name].update_values(
                                stats["sum_abs"],
                                stats["sum_sq"],
                                stats["sum_signed"],
                                stats["count"],
                                stats["max_abs"],
                            )
                            row.update(_flatten_metric(name, stats))

                        if frame_writer is None:
                            frame_writer = csv.DictWriter(frame_csv, fieldnames=list(row.keys()))
                            frame_writer.writeheader()
                        frame_writer.writerow(row)
                        total_frames += 1

                    if config["memory_meas_override"]:
                        prev_pred = torch.where(meas_m > 0.5, meas_h, pred_h)
                    else:
                        prev_pred = pred_h
                    prev_pose = cur_pose
                    prev_valid_state = torch.maximum(meas_m, prev_in_valid)

                for b, seq_idx in enumerate(original_indices):
                    source_files = _source_files_for_seq(dataset, seq_idx, timesteps)
                    seq_row = {
                        "seq_idx": seq_idx,
                        "start_file": source_files[0] if source_files else "",
                        "end_file": source_files[-1] if source_files else "",
                        "frames": timesteps,
                    }
                    for name, stats_obj in seq_stats[b].items():
                        seq_row.update(_flatten_metric(name, stats_obj.as_dict()))
                    if seq_writer is None:
                        seq_writer = csv.DictWriter(seq_csv, fieldnames=list(seq_row.keys()))
                        seq_writer.writeheader()
                    seq_writer.writerow(seq_row)

                    if args.save_predictions:
                        out_npz = os.path.join(pred_dir, f"seq_{seq_idx:06d}.npz")
                        arrays = {
                            "pred_heightmap": np.stack(seq_pred[b], axis=0).astype(np.float32),
                            "gt_heightmap": np.stack(seq_gt[b], axis=0).astype(np.float32),
                            "measurement_heightmap": np.stack(seq_meas[b], axis=0).astype(np.float32),
                            "gt_mask": np.stack(seq_gt_mask[b], axis=0).astype(np.float32),
                            "measurement_mask": np.stack(seq_meas_mask[b], axis=0).astype(np.float32),
                            "visible_mask": np.stack(seq_visible[b], axis=0).astype(np.float32),
                            "unobserved_mask": np.stack(seq_unobserved[b], axis=0).astype(np.float32),
                            "abs_error": np.stack(seq_abs_err[b], axis=0).astype(np.float32),
                            "source_files": np.asarray(source_files[::args.save_frame_stride], dtype=str),
                            "seq_idx": np.asarray(seq_idx, dtype=np.int64),
                            "map_size": np.asarray(config["map_size"], dtype=np.float32),
                            "resolution": np.asarray(config["resolution"], dtype=np.float32),
                        }
                        _save_sequence_npz(out_npz, arrays, compress=not args.no_compress)

                processed += batch_size
                if (batch_idx + 1) % args.log_interval == 0 or processed >= len(seq_indices):
                    current = aggregate["overall"].as_dict()
                    mae = current["mae"]
                    mae_s = "nan" if mae is None else f"{mae:.6f}"
                    print(
                        f"[offline-val] batch {batch_idx + 1}/{len(loader)} "
                        f"seq={processed}/{len(seq_indices)} overall_mae={mae_s}"
                    )
    finally:
        frame_csv.close()
        seq_csv.close()

    summary = {
        "meta": {
            **meta,
            "processed_sequences": int(processed),
            "processed_frames": int(total_frames),
            "out_dir": out_dir,
        },
        "regions": {name: stats.as_dict() for name, stats in aggregate.items()},
    }
    _write_json(os.path.join(out_dir, "metrics_summary.json"), summary)

    regions = summary["regions"]
    def _fmt_metric(value):
        return "nan" if value is None else f"{value:.6f}"

    print(
        "[offline-val] done "
        f"seq={processed} frames={total_frames} "
        f"overall_mae={_fmt_metric(regions['overall']['mae'])} "
        f"visible_mae={_fmt_metric(regions['visible']['mae'])} "
        f"unobserved_mae={_fmt_metric(regions['unobserved']['mae'])} "
        f"edge_mae={_fmt_metric(regions['edge']['mae'])}"
    )
    print(f"[offline-val] summary={os.path.join(out_dir, 'metrics_summary.json')}")
    print(f"[offline-val] per-frame={frame_csv_path}")
    print(f"[offline-val] per-sequence={seq_csv_path}")
    if args.save_predictions:
        print(f"[offline-val] predictions={pred_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Offline NSR validation: run validation samples through the NSR network, "
            "save predicted height maps, and compare them with GT height maps grid by grid."
        )
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", type=str, default="", help="Override data dir; default comes from checkpoint train_args")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out_dir", type=str, default="", help="Default: results/experiments/nsr/offline_val_<timestamp>")

    parser.add_argument("--seq_len", type=int, default=0, help="0 means use checkpoint train_args")
    parser.add_argument("--min_seq_len", type=int, default=0, help="0 means use checkpoint train_args")
    parser.add_argument("--map_size", type=float, default=0.0, help="<=0 means use checkpoint train_args")
    parser.add_argument("--resolution", type=float, default=0.0, help="<=0 means use checkpoint train_args")
    parser.add_argument("--fill_value", type=float, default=None, help="Default comes from checkpoint train_args")
    parser.add_argument("--in_channels", type=int, default=0, help="0 means use checkpoint train_args")
    parser.add_argument("--base_channels", type=int, default=0, help="0 means use checkpoint train_args")

    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument("--world_frame", action="store_true", dest="world_frame", default=None)
    frame_group.add_argument("--local_frame", action="store_false", dest="world_frame")

    gravity_group = parser.add_mutually_exclusive_group()
    gravity_group.add_argument("--gravity_aligned", action="store_true", dest="gravity_aligned", default=None)
    gravity_group.add_argument("--full_6dof_frame", action="store_false", dest="gravity_aligned")

    align_group = parser.add_mutually_exclusive_group()
    align_group.add_argument("--align_prev", action="store_true", dest="align_prev", default=None)
    align_group.add_argument("--no_align_prev", action="store_false", dest="align_prev")

    prev_group = parser.add_mutually_exclusive_group()
    prev_group.add_argument("--disable_prev", action="store_true", dest="disable_prev", default=None)
    prev_group.add_argument("--enable_prev", action="store_false", dest="disable_prev")

    parser.add_argument("--start_seq", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=0, help="<=0 evaluates all selected split sequences")
    parser.add_argument("--random_sample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", default=False)

    parser.add_argument("--save_predictions", action="store_true", default=True)
    parser.add_argument("--no_save_predictions", action="store_false", dest="save_predictions")
    parser.add_argument("--save_frame_stride", type=int, default=1, help="Save every Nth frame into sequence npz files")
    parser.add_argument("--edge_thresh", type=float, default=0.01, help="Height-gradient threshold (m) for edge-region metrics")
    parser.add_argument("--no_compress", action="store_true", default=False, help="Use np.savez instead of np.savez_compressed")
    parser.add_argument("--allow_partial_load", action="store_true", default=False)
    parser.add_argument("--bn_use_batch_stats", action="store_true", default=True)
    parser.add_argument("--bn_use_running_stats", action="store_false", dest="bn_use_batch_stats")
    parser.add_argument("--log_interval", type=int, default=20)

    args = parser.parse_args()
    if args.save_frame_stride <= 0:
        raise ValueError("--save_frame_stride must be positive")
    run_offline_validation(args)


if __name__ == "__main__":
    main()
