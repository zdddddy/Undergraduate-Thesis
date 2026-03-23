import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader


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


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stats(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {
            "count": 0,
            "ms_mean": 0.0,
            "ms_std": 0.0,
            "ms_p50": 0.0,
            "ms_p90": 0.0,
            "ms_p95": 0.0,
            "ms_p99": 0.0,
            "hz_mean": 0.0,
            "hz_p50": 0.0,
        }
    x = np.asarray(arr, dtype=np.float64)
    mean_ms = float(x.mean())
    p50 = float(np.percentile(x, 50))
    return {
        "count": int(x.size),
        "ms_mean": mean_ms,
        "ms_std": float(x.std()),
        "ms_p50": p50,
        "ms_p90": float(np.percentile(x, 90)),
        "ms_p95": float(np.percentile(x, 95)),
        "ms_p99": float(np.percentile(x, 99)),
        "hz_mean": float(1000.0 / max(mean_ms, 1e-9)),
        "hz_p50": float(1000.0 / max(p50, 1e-9)),
    }


def _pick_device(mode: str) -> torch.device:
    mode = str(mode).strip().lower()
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    train_args = ckpt.get("train_args", {})

    data_dir = args.data_dir if args.data_dir else train_args.get("data_dir", "")
    if not data_dir:
        raise ValueError("Missing data_dir. Pass --data_dir or use ckpt with train_args['data_dir'].")

    seq_len = int(args.seq_len if args.seq_len > 0 else train_args.get("seq_len", 24))
    min_seq_len = int(args.min_seq_len if args.min_seq_len > 0 else train_args.get("min_seq_len", seq_len))
    map_size = float(args.map_size if args.map_size > 0 else train_args.get("map_size", 3.2))
    resolution = float(args.resolution if args.resolution > 0 else train_args.get("resolution", 0.05))
    fill_value = float(train_args.get("fill_value", 0.0))

    in_channels = int(train_args.get("in_channels", 4))
    base_channels = int(train_args.get("base_channels", 32))
    use_edge_head = bool(train_args.get("use_edge_head", False))
    norm_type = str(train_args.get("norm_type", "group"))
    group_norm_groups = int(train_args.get("group_norm_groups", 8))

    gravity_aligned = bool(train_args.get("gravity_aligned", True))
    align_prev = bool(train_args.get("align_prev", True))
    disable_prev = bool(train_args.get("disable_prev", False))
    prev_valid_threshold = float(train_args.get("prev_valid_threshold", 0.5))
    memory_meas_override = bool(train_args.get("memory_meas_override", True))
    residual_from_base = bool(train_args.get("residual_from_base", False))
    residual_scale = float(train_args.get("residual_scale", 0.2))
    residual_tanh = bool(train_args.get("residual_tanh", True))

    device = _pick_device(args.device)

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

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(args.num_workers),
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

    total_ms = []
    warp_ms = []
    model_ms = []
    steps = 0
    warmup_steps = max(0, int(args.warmup_steps))
    max_steps = max(1, int(args.max_steps))

    with torch.no_grad():
        for batch in loader:
            in_height = batch["in_height"].to(device)
            in_mask = batch["in_mask"].to(device)
            pose7 = batch["pose7"].to(device)

            bsz, timesteps = in_height.shape[0], in_height.shape[1]
            prev_pred = None
            prev_pose = None
            prev_valid_state = None

            for t in range(timesteps):
                if steps >= max_steps:
                    break

                meas_h = in_height[:, t]
                meas_m = in_mask[:, t]
                cur_pose = pose7[:, t]

                _sync_if_cuda(device)
                t0 = _now_ms()

                _sync_if_cuda(device)
                tw0 = _now_ms()
                if disable_prev or prev_pred is None:
                    prev_in = meas_h
                    prev_in_valid = meas_m
                else:
                    if align_prev:
                        prev_valid_for_warp = (
                            prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                        )
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
                        prev_in_valid = (
                            prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                        )
                prev_in_valid = _binarize_valid(prev_in_valid, prev_valid_threshold)
                _sync_if_cuda(device)
                tw1 = _now_ms()

                _sync_if_cuda(device)
                tm0 = _now_ms()
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
                _sync_if_cuda(device)
                tm1 = _now_ms()

                if memory_meas_override:
                    prev_pred = torch.where(meas_m > 0.5, meas_h, pred_h)
                else:
                    prev_pred = pred_h
                prev_pose = cur_pose
                prev_valid_state = torch.maximum(meas_m, prev_in_valid)

                _sync_if_cuda(device)
                t1 = _now_ms()

                if steps >= warmup_steps:
                    total_ms.append(float(t1 - t0))
                    warp_ms.append(float(tw1 - tw0))
                    model_ms.append(float(tm1 - tm0))
                steps += bsz

            if steps >= max_steps:
                break

    total_stat = _stats(total_ms)
    warp_stat = _stats(warp_ms)
    model_stat = _stats(model_ms)

    out = {
        "ckpt": os.path.abspath(args.ckpt),
        "data_dir": os.path.abspath(data_dir),
        "split": args.split,
        "device": str(device),
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "config": {
            "seq_len": seq_len,
            "min_seq_len": min_seq_len,
            "map_size": map_size,
            "resolution": resolution,
            "align_prev": align_prev,
            "gravity_aligned": gravity_aligned,
            "in_channels": in_channels,
            "base_channels": base_channels,
        },
        "metrics": {
            "total_ms": total_stat,
            "warp_ms": warp_stat,
            "model_ms": model_stat,
        },
    }

    if args.out_json:
        out_dir = os.path.dirname(os.path.abspath(args.out_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=True)
        print(f"[bench] saved: {args.out_json}")

    print(
        "[bench] total "
        f"count={total_stat['count']} "
        f"mean={total_stat['ms_mean']:.3f}ms "
        f"p95={total_stat['ms_p95']:.3f}ms "
        f"hz_mean={total_stat['hz_mean']:.2f}"
    )
    print(
        "[bench] warp  "
        f"mean={warp_stat['ms_mean']:.3f}ms "
        f"p95={warp_stat['ms_p95']:.3f}ms"
    )
    print(
        "[bench] model "
        f"mean={model_stat['ms_mean']:.3f}ms "
        f"p95={model_stat['ms_p95']:.3f}ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="", help="Optional override; default from ckpt train_args")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=0, help="0 means use ckpt train_args")
    parser.add_argument("--min_seq_len", type=int, default=0, help="0 means use ckpt train_args")
    parser.add_argument("--map_size", type=float, default=0.0, help="<=0 means use ckpt train_args")
    parser.add_argument("--resolution", type=float, default=0.0, help="<=0 means use ckpt train_args")
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
