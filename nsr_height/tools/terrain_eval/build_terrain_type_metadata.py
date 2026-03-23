import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def _resolve_split_root(data_dir, split):
    split_root = data_dir
    explicit = os.path.join(data_dir, split)
    if os.path.isdir(explicit):
        split_root = explicit
    return split_root


def _find_traj_dirs(split_root):
    traj_dirs = sorted(glob.glob(os.path.join(split_root, "traj_*")))
    if traj_dirs:
        return traj_dirs
    return sorted(glob.glob(os.path.join(split_root, "env_*", "traj_*")))


def _traj_key(traj_dir, split_root):
    rel = os.path.relpath(traj_dir, split_root).replace("\\", "/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) >= 2 and parts[-2].startswith("env_") and parts[-1].startswith("traj_"):
        return f"{parts[-2]}/{parts[-1]}"
    if parts and parts[-1].startswith("traj_"):
        return parts[-1]
    return rel


def _sample_indices(length, max_items):
    if length <= 0:
        return []
    if max_items <= 0 or length <= max_items:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, num=max_items, dtype=int).tolist()))


def _load_gt_heightmap(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        if "gt_heightmap" not in data.files:
            raise KeyError(f"{npz_path}: missing gt_heightmap")
        h = data["gt_heightmap"].astype(np.float32)
        if h.ndim == 3 and h.shape[0] == 1:
            h = h[0]
        if h.ndim != 2:
            raise ValueError(f"{npz_path}: invalid gt_heightmap shape {h.shape}")

        if "gt_mask" in data.files:
            m = data["gt_mask"].astype(np.float32)
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            if m.shape != h.shape:
                raise ValueError(f"{npz_path}: gt_mask shape mismatch {m.shape} vs {h.shape}")
            m = (m > 0.5).astype(np.float32)
        else:
            m = np.isfinite(h).astype(np.float32)

        if "gt_resolution" in data.files:
            res = float(np.asarray(data["gt_resolution"]).reshape(-1)[0])
        else:
            res = 0.05

    h = h.astype(np.float32)
    h[m <= 0.5] = 0.0
    return h, m, res


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


def _fill_and_smooth(h, valid, smooth_passes):
    out = h.astype(np.float32).copy()
    if not np.any(valid):
        return np.zeros_like(out, dtype=np.float32)
    mean_z = float(out[valid].mean())
    out[~valid] = mean_z
    for _ in range(max(0, int(smooth_passes))):
        out = _box_smooth_3x3(out)
    return out


def _norm(v, lo, hi):
    if hi <= lo:
        return 0.0
    return float(max(0.0, min(1.0, (float(v) - lo) / (hi - lo))))


def _clamp01(v):
    return float(max(0.0, min(1.0, v)))


def _compute_features_frame(h, m, resolution, smooth_passes=2):
    valid = m > 0.5
    n_valid = int(valid.sum())
    if n_valid < 32:
        return {
            "is_valid": False,
            "valid_pixels": n_valid,
            "z_range": 0.0,
            "slope_mag": 0.0,
            "plane_resid_std": 0.0,
            "edge_ratio_g25": 0.0,
            "edge_ratio_g40": 0.0,
            "edge_dir_coherence": 0.0,
            "flat_ratio_g08": 0.0,
            "z_unique_2cm": 0,
            "spike_ratio_abs06": 0.0,
            "spike_ratio_abs10": 0.0,
            "center_minus_outer": 0.0,
        }

    hf = _fill_and_smooth(h, valid, smooth_passes=smooth_passes)
    h_h, h_w = hf.shape
    ys, xs = np.meshgrid(
        (np.arange(h_h, dtype=np.float32) + 0.5) * float(resolution),
        (np.arange(h_w, dtype=np.float32) + 0.5) * float(resolution),
        indexing="ij",
    )

    a = np.stack([xs[valid], ys[valid], np.ones(n_valid, dtype=np.float32)], axis=1)
    zv = hf[valid]
    beta, *_ = np.linalg.lstsq(a, zv, rcond=None)
    plane = beta[0] * xs + beta[1] * ys + beta[2]
    resid = hf - plane
    rv = resid[valid]

    slope_mag = float(np.linalg.norm(beta[:2]))
    plane_resid_std = float(np.std(rv))
    z_range = float(np.quantile(zv, 0.95) - np.quantile(zv, 0.05))

    gx = np.gradient(hf, axis=1) / float(resolution)
    gy = np.gradient(hf, axis=0) / float(resolution)
    grad = np.sqrt(gx * gx + gy * gy)
    edge_ratio_g25 = float((grad[valid] > 0.25).mean())
    edge_ratio_g40 = float((grad[valid] > 0.40).mean())
    flat_ratio_g08 = float((grad[valid] < 0.08).mean())

    edge_mask = (grad > 0.25) & valid
    if np.any(edge_mask):
        theta = np.mod(np.arctan2(gy[edge_mask], gx[edge_mask]), np.pi)
        hist, _ = np.histogram(theta, bins=18, range=(0.0, np.pi))
        edge_dir_coherence = float(hist.max() / max(int(hist.sum()), 1))
    else:
        edge_dir_coherence = 0.0

    z_unique_2cm = int(np.unique(np.round(rv / 0.02)).size)
    spike_ratio_abs06 = float((np.abs(rv) > 0.06).mean())
    spike_ratio_abs10 = float((np.abs(rv) > 0.10).mean())

    x0 = xs - float(xs.mean())
    y0 = ys - float(ys.mean())
    rr = np.sqrt(x0 * x0 + y0 * y0)
    center = (rr <= 0.45) & valid
    outer = (rr >= 1.10) & valid
    if np.any(center) and np.any(outer):
        center_minus_outer = float(hf[center].mean() - hf[outer].mean())
    else:
        center_minus_outer = 0.0

    return {
        "is_valid": True,
        "valid_pixels": n_valid,
        "z_range": z_range,
        "slope_mag": slope_mag,
        "plane_resid_std": plane_resid_std,
        "edge_ratio_g25": edge_ratio_g25,
        "edge_ratio_g40": edge_ratio_g40,
        "edge_dir_coherence": edge_dir_coherence,
        "flat_ratio_g08": flat_ratio_g08,
        "z_unique_2cm": z_unique_2cm,
        "spike_ratio_abs06": spike_ratio_abs06,
        "spike_ratio_abs10": spike_ratio_abs10,
        "center_minus_outer": center_minus_outer,
    }


def _aggregate_features(per_frame_features):
    valid_feats = [f for f in per_frame_features if f.get("is_valid", False)]
    if not valid_feats:
        return {
            "is_valid": False,
            "frames_used": len(per_frame_features),
            "valid_frame_count": 0,
        }

    keys = [k for k in valid_feats[0].keys() if k not in {"is_valid"}]
    out = {
        "is_valid": True,
        "frames_used": len(per_frame_features),
        "valid_frame_count": len(valid_feats),
    }
    for k in keys:
        vals = [float(f[k]) for f in valid_feats]
        out[k] = float(np.median(vals))
    return out


def _classify_collect_large(features, strict_conf):
    if not features.get("is_valid", False):
        return "unknown", 0.0, "unknown", "unknown"

    slope = float(features["slope_mag"])
    resid = float(features["plane_resid_std"])
    edge = float(features["edge_ratio_g25"])
    coh = float(features["edge_dir_coherence"])
    flat = float(features["flat_ratio_g08"])
    levels = float(features["z_unique_2cm"])
    spike06 = float(features["spike_ratio_abs06"])
    center_delta = float(features["center_minus_outer"])

    slope_base = (
        0.32
        + 0.43 * _norm(slope, 0.06, 0.35)
        + 0.12 * (1.0 - _norm(coh, 0.40, 0.80))
        + 0.13 * (1.0 - _norm(levels, 12.0, 45.0))
    )
    conf_smooth = slope_base + 0.20 * (1.0 - _norm(resid, 0.02, 0.08)) + 0.10 * (
        1.0 - _norm(edge, 0.15, 0.45)
    )
    conf_rough = slope_base + 0.24 * _norm(resid, 0.03, 0.14) + 0.06 * _norm(edge, 0.10, 0.50)

    conf_stairs = (
        0.30
        + 0.30 * _norm(edge, 0.18, 0.70)
        + 0.28 * _norm(coh, 0.40, 0.80)
        + 0.12 * _norm(levels, 12.0, 45.0)
    )
    if coh < 0.30 or edge < 0.12:
        conf_stairs -= 0.22

    conf_discrete = (
        0.30
        + 0.24 * (1.0 - _norm(slope, 0.03, 0.14))
        + 0.24 * _norm(spike06, 0.02, 0.18)
        + 0.12 * (1.0 - _norm(coh, 0.30, 0.70))
        + 0.10 * _norm(resid, 0.02, 0.12)
        + 0.10 * _norm(flat, 0.35, 0.95)
    )
    if slope > 0.12:
        conf_discrete -= 0.22
    if coh > 0.50:
        conf_discrete -= 0.22

    conf_smooth = _clamp01(conf_smooth)
    conf_rough = _clamp01(conf_rough)
    conf_stairs = _clamp01(conf_stairs)
    conf_discrete = _clamp01(conf_discrete)

    if center_delta >= 0.01:
        stairs_label = "stairs_up"
    elif center_delta <= -0.01:
        stairs_label = "stairs_down"
    else:
        stairs_label = "stairs_up" if center_delta >= 0.0 else "stairs_down"
        conf_stairs = _clamp01(conf_stairs - 0.04)

    candidates = {
        "smooth_slope": conf_smooth,
        "rough_slope": conf_rough,
        stairs_label: conf_stairs,
        "discrete_obstacles": conf_discrete,
    }
    terrain, conf = max(candidates.items(), key=lambda kv: kv[1])

    if terrain in {"smooth_slope", "rough_slope"}:
        family = "slope"
    elif terrain in {"stairs_up", "stairs_down"}:
        family = "stairs"
    elif terrain == "discrete_obstacles":
        family = "discrete_obstacles"
    else:
        family = "unknown"

    strict = terrain if conf >= float(strict_conf) else "unknown"
    return terrain, conf, strict, family


def _classify_generic_from_collect(collect_t, collect_conf, strict_conf):
    if collect_t in {"stairs_up", "stairs_down"}:
        terrain_type = "stairs_like"
    elif collect_t in {"smooth_slope", "rough_slope"}:
        terrain_type = "slope_like"
    elif collect_t == "discrete_obstacles":
        terrain_type = "mixed_like"
    else:
        terrain_type = "unknown"
    strict_type = terrain_type if collect_conf >= float(strict_conf) else "unknown"
    return terrain_type, collect_conf, strict_type


def build_metadata(args):
    split_root = _resolve_split_root(args.data_dir, args.split)
    traj_dirs = _find_traj_dirs(split_root)
    if not traj_dirs:
        raise RuntimeError(f"no trajectories found under {split_root}")

    out = {
        "meta": {
            "data_dir": os.path.abspath(args.data_dir),
            "split": args.split,
            "split_root": os.path.abspath(split_root),
            "strict_conf_threshold": float(args.strict_conf_threshold),
            "label_source": "heuristic_from_gt_heightmap_multi_frame",
            "frames_per_traj": int(args.frames_per_traj),
            "smooth_passes": int(args.smooth_passes),
            "version": 2,
        },
        "labels_by_traj": {},
        "summary": {},
    }

    soft_counter = Counter()
    strict_counter = Counter()
    collect_soft_counter = Counter()
    collect_strict_counter = Counter()
    collect_family_counter = Counter()

    for traj_dir in traj_dirs:
        npz_files = sorted(glob.glob(os.path.join(traj_dir, "*.npz")))
        if not npz_files:
            continue
        key = _traj_key(traj_dir, split_root)
        idxs = _sample_indices(len(npz_files), int(args.frames_per_traj))
        sample_paths = [npz_files[i] for i in idxs]

        try:
            frame_features = []
            resolution = 0.05
            for p in sample_paths:
                h, m, resolution = _load_gt_heightmap(p)
                frame_features.append(
                    _compute_features_frame(
                        h=h,
                        m=m,
                        resolution=resolution,
                        smooth_passes=args.smooth_passes,
                    )
                )

            features = _aggregate_features(frame_features)
            collect_t, collect_conf, collect_strict, collect_family = _classify_collect_large(
                features, args.strict_conf_threshold
            )
            terrain_type, conf, strict_type = _classify_generic_from_collect(
                collect_t, collect_conf, args.strict_conf_threshold
            )

            rec = {
                "terrain_type": terrain_type,
                "terrain_type_confidence": conf,
                "strict_terrain_type": strict_type,
                "collect_large_terrain_type": collect_t,
                "collect_large_terrain_type_confidence": collect_conf,
                "strict_collect_large_terrain_type": collect_strict,
                "collect_large_terrain_family": collect_family,
                "features": features,
                "num_frames_sampled": len(sample_paths),
                "example_npz": os.path.abspath(sample_paths[0]),
            }
        except Exception as e:
            rec = {
                "terrain_type": "unknown",
                "terrain_type_confidence": 0.0,
                "strict_terrain_type": "unknown",
                "collect_large_terrain_type": "unknown",
                "collect_large_terrain_type_confidence": 0.0,
                "strict_collect_large_terrain_type": "unknown",
                "collect_large_terrain_family": "unknown",
                "features": {"is_valid": False},
                "num_frames_sampled": len(sample_paths),
                "example_npz": os.path.abspath(sample_paths[0]),
                "error": str(e),
            }

        out["labels_by_traj"][key] = rec
        soft_counter[rec["terrain_type"]] += 1
        strict_counter[rec["strict_terrain_type"]] += 1
        collect_soft_counter[rec["collect_large_terrain_type"]] += 1
        collect_strict_counter[rec["strict_collect_large_terrain_type"]] += 1
        collect_family_counter[rec["collect_large_terrain_family"]] += 1

    out["summary"]["num_traj"] = int(sum(soft_counter.values()))
    out["summary"]["soft_counts"] = dict(sorted(soft_counter.items()))
    out["summary"]["strict_counts"] = dict(sorted(strict_counter.items()))
    out["summary"]["collect_large_soft_counts"] = dict(sorted(collect_soft_counter.items()))
    out["summary"]["collect_large_strict_counts"] = dict(sorted(collect_strict_counter.items()))
    out["summary"]["collect_large_family_counts"] = dict(sorted(collect_family_counter.items()))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)

    print(f"[terrain-meta] saved: {args.out_json}")
    print(f"[terrain-meta] trajectories: {out['summary']['num_traj']}")
    print(f"[terrain-meta] soft counts: {out['summary']['soft_counts']}")
    print(f"[terrain-meta] strict counts: {out['summary']['strict_counts']}")
    print(f"[terrain-meta] collect_large soft: {out['summary']['collect_large_soft_counts']}")
    print(f"[terrain-meta] collect_large strict: {out['summary']['collect_large_strict_counts']}")
    print(f"[terrain-meta] collect_large family: {out['summary']['collect_large_family_counts']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument(
        "--strict_conf_threshold",
        type=float,
        default=0.65,
        help="labels with confidence below this threshold are mapped to strict_terrain_type=unknown",
    )
    parser.add_argument(
        "--frames_per_traj",
        type=int,
        default=8,
        help="number of frames sampled per trajectory for feature aggregation",
    )
    parser.add_argument(
        "--smooth_passes",
        type=int,
        default=2,
        help="box-smoothing passes before feature extraction",
    )
    args = parser.parse_args()
    build_metadata(args)


if __name__ == "__main__":
    main()
