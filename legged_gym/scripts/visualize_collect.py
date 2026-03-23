# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import argparse
import glob
import os
import time

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")


def _resolve_data_dir(data_dir, traj_idx=None):
    paths = sorted(glob.glob(os.path.join(data_dir, "frame_*.npz")))
    if paths:
        return data_dir

    traj_dirs = sorted(glob.glob(os.path.join(data_dir, "traj_*")))
    if traj_dirs:
        if traj_idx is None:
            return traj_dirs[0]
        desired = os.path.join(data_dir, f"traj_{traj_idx:04d}")
        if os.path.isdir(desired):
            return desired
        raise FileNotFoundError(f"traj_{traj_idx:04d} not found in {data_dir}")

    raise FileNotFoundError(f"No frame_*.npz or traj_* found in {data_dir}")


def _load_frames(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "frame_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No frame_*.npz found in {data_dir}")
    return paths


def _load_clouds(path):
    data = np.load(path)
    if "cam_points" in data:
        cam_pts = data["cam_points"]
    else:
        cams = []
        for k in ["cam0", "cam1", "cam2", "cam3"]:
            if k in data:
                cams.append(data[k])
        cam_pts = np.concatenate(cams, axis=0) if cams else np.zeros((0, 3), dtype=np.float32)
    terrain_pts = data["terrain"] if "terrain" in data else np.zeros((0, 3), dtype=np.float32)
    if "robot_pose7" in data:
        robot_pos = data["robot_pose7"][:3]
        robot_quat = data["robot_pose7"][3:]
    else:
        robot_pos = data["robot_pos"] if "robot_pos" in data else None
        robot_quat = None
    return cam_pts, terrain_pts, robot_pos, robot_quat


def _quat_to_rot(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    q = q / n
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_to_yaw_rot(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / n
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _to_local(points, pos, quat, gravity_aligned=True):
    if points is None or points.shape[0] == 0 or pos is None or quat is None:
        return points
    rot = _quat_to_yaw_rot(quat) if gravity_aligned else _quat_to_rot(quat)
    return (rot.T @ (points - pos).T).T


def _rasterize_points(points, map_size, resolution, fill_value=0.0):
    grid_size = int(round(float(map_size) / float(resolution)))
    half_extent = 0.5 * float(map_size)

    hmap = np.full((grid_size, grid_size), -np.inf, dtype=np.float32)
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    if points is None or points.size == 0:
        hmap.fill(fill_value)
        return hmap, mask

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ix = np.floor((x + half_extent) / resolution).astype(np.int32)
    iy = np.floor((y + half_extent) / resolution).astype(np.int32)

    valid = (
        (ix >= 0)
        & (ix < grid_size)
        & (iy >= 0)
        & (iy < grid_size)
        & np.isfinite(z)
    )
    if not np.any(valid):
        hmap.fill(fill_value)
        return hmap, mask

    ix = ix[valid]
    iy = iy[valid]
    z = z[valid]
    linear = ix * grid_size + iy
    flat_hmap = hmap.reshape(-1)
    np.maximum.at(flat_hmap, linear, z)
    flat_mask = mask.reshape(-1)
    flat_mask[linear] = 1.0
    hmap[mask <= 0.5] = fill_value
    return hmap, mask


def _load_maps_for_frame(path, default_map_size=3.2, default_resolution=0.05, fill_value=0.0):
    with np.load(path) as data:
        map_size = float(np.asarray(data["gt_map_size"]).reshape(-1)[0]) if "gt_map_size" in data else float(default_map_size)
        resolution = float(np.asarray(data["gt_resolution"]).reshape(-1)[0]) if "gt_resolution" in data else float(default_resolution)

        if "gt_heightmap" in data:
            gt_h = data["gt_heightmap"].astype(np.float32)
            if gt_h.ndim == 3 and gt_h.shape[0] == 1:
                gt_h = gt_h[0]
            if "gt_mask" in data:
                gt_m = data["gt_mask"].astype(np.float32)
                if gt_m.ndim == 3 and gt_m.shape[0] == 1:
                    gt_m = gt_m[0]
            else:
                gt_m = np.isfinite(gt_h).astype(np.float32)
        else:
            terrain = data["terrain"].astype(np.float32) if "terrain" in data else np.zeros((0, 3), dtype=np.float32)
            gt_h, gt_m = _rasterize_points(terrain, map_size, resolution, fill_value=fill_value)

        gt_m = (gt_m > 0.5).astype(np.float32)
        gt_h[gt_m <= 0.5] = fill_value

        pose7 = data["robot_pose7"].astype(np.float32) if "robot_pose7" in data else None
        meas = data["cam_points"].astype(np.float32) if "cam_points" in data else np.zeros((0, 3), dtype=np.float32)

        gt_is_local = bool(int(np.asarray(data["gt_local_frame"]).reshape(-1)[0])) if "gt_local_frame" in data else True
        gt_gravity_aligned = (
            bool(int(np.asarray(data["gt_gravity_aligned"]).reshape(-1)[0]))
            if "gt_gravity_aligned" in data
            else True
        )
        if gt_is_local and pose7 is not None and pose7.shape == (7,):
            meas = _to_local(meas, pose7[:3], pose7[3:7], gravity_aligned=gt_gravity_aligned).astype(np.float32)

    meas_h, meas_m = _rasterize_points(meas, map_size, resolution, fill_value=fill_value)
    return {
        "meas_h": meas_h,
        "meas_m": meas_m,
        "gt_h": gt_h,
        "gt_m": gt_m,
        "map_size": float(map_size),
        "resolution": float(resolution),
    }


def _find_traj_dirs(data_dir, split="train"):
    if os.path.isdir(os.path.join(data_dir, split)):
        base = os.path.join(data_dir, split)
    else:
        base = data_dir

    direct = sorted(glob.glob(os.path.join(base, "traj_*")))
    nested = sorted(glob.glob(os.path.join(base, "env_*", "traj_*")))
    traj_dirs = [d for d in (direct + nested) if os.path.isdir(d)]

    if not traj_dirs and os.path.basename(base).startswith("traj_"):
        traj_dirs = [base]
    return traj_dirs


def _connected_components(binary):
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=np.uint8)
    comp_count = 0
    max_area = 0
    for i in range(h):
        for j in range(w):
            if not binary[i, j] or visited[i, j]:
                continue
            comp_count += 1
            stack = [(i, j)]
            visited[i, j] = 1
            area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= nx < h and 0 <= ny < w and binary[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = 1
                        stack.append((nx, ny))
            if area > max_area:
                max_area = area
    return comp_count, max_area


def _terrain_features(gt_h, gt_m):
    valid = gt_m > 0.5
    n_valid = int(valid.sum())
    if n_valid < 64:
        return None

    vals = gt_h[valid]
    median = float(np.median(vals))
    h = gt_h.copy()
    h[~valid] = median
    gx, gy = np.gradient(h)
    grad = np.sqrt(gx * gx + gy * gy)
    gradv = grad[valid]
    edge_thr = float(max(0.015, np.percentile(gradv, 75)))
    edge_mask = (grad > edge_thr) & valid
    edge_ratio = float(edge_mask.sum() / max(1, n_valid))

    ii, jj = np.nonzero(valid)
    x = (ii / max(1, gt_h.shape[0] - 1)) * 2.0 - 1.0
    y = (jj / max(1, gt_h.shape[1] - 1)) * 2.0 - 1.0
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    coeff, *_ = np.linalg.lstsq(a, vals, rcond=None)
    plane_vals = a @ coeff
    plane_resid = float(np.mean(np.abs(vals - plane_vals)))

    plane_map = np.zeros_like(gt_h, dtype=np.float32)
    plane_map[valid] = plane_vals.astype(np.float32)
    resid_map = np.abs(h - plane_map)
    hard_thr = float(max(0.03, np.percentile(resid_map[valid], 85)))
    hard = (resid_map > hard_thr) & valid
    comp_count, max_area = _connected_components(hard)

    if edge_mask.any():
        ang = np.arctan2(gy[edge_mask], gx[edge_mask])
        c = float(np.mean(np.cos(2.0 * ang)))
        s = float(np.mean(np.sin(2.0 * ang)))
        coherence = float(np.sqrt(c * c + s * s))
    else:
        coherence = 0.0

    zq = np.round((vals - float(np.mean(vals))) / 0.03).astype(np.int32)
    num_levels = int(np.unique(zq).size)

    comp_density = float(comp_count / 20.0)
    max_area_ratio = float(max_area / max(1, n_valid))
    levels_norm = float(min(1.0, num_levels / 20.0))

    rough_flat_score = (
        1.8 * plane_resid
        + 0.9 * edge_ratio
        + 0.7 * comp_density
        - 1.2 * max_area_ratio
        - 0.6 * coherence
    )
    scores = {
        "slope": -2.5 * plane_resid - 1.5 * edge_ratio + 0.8 * coherence,
        "rough_flat": rough_flat_score,
        "rough": rough_flat_score,  # backward-compatible alias
        "stairs": 1.5 * coherence + 0.8 * edge_ratio + 1.0 * levels_norm - 0.8 * plane_resid,
        "obstacles": 1.8 * plane_resid + 0.7 * edge_ratio + 1.0 * max_area_ratio - 0.7 * coherence,
    }
    return scores


def _sample_window_starts(num_frames, frames_per_gif, max_windows_per_traj):
    if num_frames <= 0:
        return []
    if num_frames <= frames_per_gif:
        return [0]
    last_start = num_frames - frames_per_gif
    count = int(max(1, min(max_windows_per_traj, last_start + 1)))
    starts = np.linspace(0, last_start, num=count, dtype=np.int32)
    return sorted({int(x) for x in starts.tolist()})


def _read_traj_direct_label(traj_dir):
    frames = _load_frames(traj_dir)
    if not frames:
        return None
    try:
        with np.load(frames[0]) as data:
            terrain_type = str(np.asarray(data["terrain_type"]).reshape(-1)[0]) if "terrain_type" in data else ""
            terrain_family = str(np.asarray(data["terrain_family"]).reshape(-1)[0]) if "terrain_family" in data else ""
    except Exception:
        return None
    if not terrain_type and not terrain_family:
        return None
    return {"terrain_type": terrain_type, "terrain_family": terrain_family}


def _direct_label_matches_request(label_info, requested_type):
    if not label_info:
        return None
    terrain_type = label_info.get("terrain_type", "")
    terrain_family = label_info.get("terrain_family", "")
    req = str(requested_type).strip().lower()
    if req == "obstacles":
        req = "discrete_obstacles"
    if req == "rough":
        req = "rough_flat"
    if req == "stairs":
        return terrain_family == "stairs"
    if req == "slope":
        return terrain_type == "smooth_slope"
    if req == "rough_flat":
        return terrain_type == "rough_slope"
    return terrain_type == req or terrain_family == req


def _score_window(frame_paths, start_idx, frames_per_gif, window_eval_frames):
    end_idx = min(start_idx + frames_per_gif, len(frame_paths))
    window_paths = frame_paths[start_idx:end_idx]
    if not window_paths:
        return None

    eval_count = int(max(1, min(window_eval_frames, len(window_paths))))
    eval_ids = np.linspace(0, len(window_paths) - 1, num=eval_count, dtype=np.int32)
    feats_list = []
    for i in eval_ids.tolist():
        maps = _load_maps_for_frame(window_paths[i])
        feats = _terrain_features(maps["gt_h"], maps["gt_m"])
        if feats is not None:
            feats_list.append(feats)
    if not feats_list:
        return None

    keys = feats_list[0].keys()
    return {k: float(np.median([f[k] for f in feats_list])) for k in keys}


def _select_trajectories_by_type(
    traj_dirs,
    terrain_types,
    scan_trajs=500,
    seed=0,
    frames_per_gif=30,
    max_windows_per_traj=6,
    window_eval_frames=4,
):
    rng = np.random.default_rng(seed)
    traj_dirs = list(traj_dirs)
    rng.shuffle(traj_dirs)
    if scan_trajs is not None and scan_trajs > 0:
        traj_dirs = traj_dirs[: min(int(scan_trajs), len(traj_dirs))]

    candidates = {t: [] for t in terrain_types}  # (score, traj_dir, start_idx, score_dict)
    for traj_i, traj_dir in enumerate(traj_dirs):
        frames = _load_frames(traj_dir)
        if not frames:
            continue
        direct_label = _read_traj_direct_label(traj_dir)
        starts = _sample_window_starts(len(frames), frames_per_gif, max_windows_per_traj)
        if not starts:
            continue

        best_in_traj = {t: (-np.inf, None, None) for t in terrain_types}
        for s in starts:
            score_dict = _score_window(frames, s, frames_per_gif, window_eval_frames)
            if score_dict is None:
                continue
            for t in terrain_types:
                matches = _direct_label_matches_request(direct_label, t)
                if matches is False:
                    continue
                sc = float(score_dict.get(t, -np.inf))
                if sc > best_in_traj[t][0]:
                    best_in_traj[t] = (sc, s, score_dict)

        for t in terrain_types:
            sc, s, score_dict = best_in_traj[t]
            if np.isfinite(sc) and s is not None:
                candidates[t].append((sc, traj_dir, s, score_dict))

        if (traj_i + 1) % 100 == 0:
            print(f"[scan] processed {traj_i+1}/{len(traj_dirs)} trajectories")

    selected = {}  # t -> (score, traj_dir, start_idx, score_dict)
    used_traj = set()
    for t in terrain_types:
        ranked = sorted(candidates[t], key=lambda x: x[0], reverse=True)
        if not ranked:
            continue

        picked = None
        for cand in ranked:
            _, traj, _, _ = cand
            if traj not in used_traj:
                picked = cand
                break
        if picked is None:
            picked = ranked[0]

        selected[t] = picked
        used_traj.add(picked[1])

        topk = ranked[:3]
        print(f"[select:{t}] top candidates:")
        for rank_i, (sc, tr, st, _) in enumerate(topk, start=1):
            print(f"  #{rank_i}: score={sc:.4f} traj={tr} start={st}")
    return selected


def _select_random_trajectories(traj_dirs, num_gifs, frames_per_gif, seed=0):
    rng = np.random.default_rng(seed)
    traj_dirs = list(traj_dirs)
    rng.shuffle(traj_dirs)
    traj_dirs = traj_dirs[: min(int(num_gifs), len(traj_dirs))]

    selected = []
    for traj_dir in traj_dirs:
        frame_paths = _load_frames(traj_dir)
        if not frame_paths:
            continue
        n_take = min(int(frames_per_gif), len(frame_paths))
        max_start = max(0, len(frame_paths) - n_take)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        selected.append((traj_dir, start))
    return selected


def _render_pair_frame(meas_h, meas_m, gt_h, gt_m, map_size, title="", vmin=None, vmax=None):
    half = 0.5 * map_size
    extent = (-half, half, -half, half)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="lightgray")

    m = meas_h.copy()
    g = gt_h.copy()
    m[meas_m <= 0.5] = np.nan
    g[gt_m <= 0.5] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=12)

    im0 = axes[0].imshow(
        m,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[0].set_title("Measurement")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        g,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


def _load_pointcloud_frame(path, max_points=20000):
    with np.load(path) as data:
        if "cam_points" in data:
            cam_pts = data["cam_points"].astype(np.float32)
        else:
            cams = []
            for k in ["cam0", "cam1", "cam2", "cam3"]:
                if k in data:
                    cams.append(data[k].astype(np.float32))
            cam_pts = np.concatenate(cams, axis=0) if cams else np.zeros((0, 3), dtype=np.float32)

        terrain_pts = data["terrain"].astype(np.float32) if "terrain" in data else np.zeros((0, 3), dtype=np.float32)
        pose7 = data["robot_pose7"].astype(np.float32) if "robot_pose7" in data else None
        map_size = float(np.asarray(data["gt_map_size"]).reshape(-1)[0]) if "gt_map_size" in data else 3.2
        gt_is_local = bool(int(np.asarray(data["gt_local_frame"]).reshape(-1)[0])) if "gt_local_frame" in data else True
        gt_gravity_aligned = (
            bool(int(np.asarray(data["gt_gravity_aligned"]).reshape(-1)[0]))
            if "gt_gravity_aligned" in data
            else True
        )

    if gt_is_local and pose7 is not None and pose7.shape == (7,):
        pos = pose7[:3]
        quat = pose7[3:7]
        cam_pts = _to_local(cam_pts, pos, quat, gravity_aligned=gt_gravity_aligned).astype(np.float32)
        terrain_pts = _to_local(terrain_pts, pos, quat, gravity_aligned=gt_gravity_aligned).astype(np.float32)

    # Crop to map extent for cleaner animation.
    half = 0.5 * map_size
    if cam_pts.size > 0:
        keep = (np.abs(cam_pts[:, 0]) <= half) & (np.abs(cam_pts[:, 1]) <= half) & np.isfinite(cam_pts[:, 2])
        cam_pts = cam_pts[keep]
    if terrain_pts.size > 0:
        keep = (np.abs(terrain_pts[:, 0]) <= half) & (np.abs(terrain_pts[:, 1]) <= half) & np.isfinite(terrain_pts[:, 2])
        terrain_pts = terrain_pts[keep]

    def _decimate(points):
        if points.shape[0] <= max_points:
            return points
        idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        return points[idx]

    cam_pts = _decimate(cam_pts)
    terrain_pts = _decimate(terrain_pts)
    return cam_pts, terrain_pts, map_size


def _render_measurement_pointcloud_frame(
    cam_pts,
    map_size,
    title="",
    zmin=None,
    zmax=None,
    point_size=2.0,
):
    half = 0.5 * map_size
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.2), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=12)

    if zmin is None or zmax is None or not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        zmin, zmax = -1.0, 1.0

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Measurement Point Cloud")
    if cam_pts is not None and cam_pts.shape[0] > 0:
        sc = ax.scatter(
            cam_pts[:, 0],
            cam_pts[:, 1],
            c=cam_pts[:, 2],
            s=float(point_size),
            cmap="viridis",
            vmin=zmin,
            vmax=zmax,
            marker=".",
            linewidths=0.0,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


def _save_gif(imgs, out_path, fps=4, frame_duration_ms=None):
    if not imgs:
        raise ValueError("No frames to save.")
    if frame_duration_ms is not None:
        duration_ms = int(max(20, round(float(frame_duration_ms))))
    else:
        duration_ms = int(max(20, round(1000.0 / max(1, int(fps)))))

    pil_frames = [Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in imgs]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,  # milliseconds per frame
        loop=0,
        optimize=False,
    )
    return duration_ms / 1000.0


def export_terrain_pointcloud_gifs(
    data_dir,
    out_dir,
    split="train",
    terrain_types=("slope", "rough_flat", "stairs", "obstacles"),
    frames_per_gif=30,
    fps=4,
    frame_duration_ms=1200,
    scan_trajs=800,
    seed=0,
    max_points=20000,
    point_size=2.0,
    random_num_gifs=0,
):
    traj_dirs = _find_traj_dirs(data_dir, split=split)
    if not traj_dirs:
        raise FileNotFoundError(f"No traj_* found under: {data_dir}")

    os.makedirs(out_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.abspath(data_dir.rstrip("/")))
    jobs = []  # (label, traj_dir, start, score)
    if int(random_num_gifs) > 0:
        random_selected = _select_random_trajectories(
            traj_dirs=traj_dirs,
            num_gifs=int(random_num_gifs),
            frames_per_gif=frames_per_gif,
            seed=seed,
        )
        if not random_selected:
            raise RuntimeError("Failed to select random trajectories.")
        for i, (traj_dir, start) in enumerate(random_selected):
            jobs.append((f"random_{i:02d}", traj_dir, start, None))
    else:
        selected = _select_trajectories_by_type(
            traj_dirs=traj_dirs,
            terrain_types=terrain_types,
            scan_trajs=scan_trajs,
            seed=seed,
            frames_per_gif=frames_per_gif,
        )
        if not selected:
            raise RuntimeError("No trajectories selected for requested terrain types.")
        for terrain_type in terrain_types:
            picked = selected.get(terrain_type)
            if picked is None:
                print(f"[warn] no trajectory found for terrain type: {terrain_type}")
                continue
            score, traj_dir, start, score_dict = picked
            jobs.append((terrain_type, traj_dir, start, float(score_dict.get(terrain_type, score))))

    for label, traj_dir, start, maybe_score in jobs:
        frame_paths = _load_frames(traj_dir)
        n_take = min(int(frames_per_gif), len(frame_paths))
        start = min(max(0, int(start)), max(0, len(frame_paths) - n_take))
        frame_paths = frame_paths[start:start + n_take]

        frames_data = [_load_pointcloud_frame(p, max_points=max_points) for p in frame_paths]
        z_values = []
        for cam_pts, _, _ in frames_data:
            if cam_pts.size > 0:
                z_values.append(cam_pts[:, 2])
        if z_values:
            zcat = np.concatenate(z_values, axis=0)
            zmin = float(np.percentile(zcat, 1))
            zmax = float(np.percentile(zcat, 99))
            if zmax <= zmin:
                zmin = float(np.min(zcat))
                zmax = float(np.max(zcat) + 1e-3)
        else:
            zmin, zmax = -1.0, 1.0

        imgs = []
        traj_name = os.path.basename(traj_dir)
        env_name = os.path.basename(os.path.dirname(traj_dir)) if os.path.basename(os.path.dirname(traj_dir)).startswith("env_") else "env_unknown"
        for i, (cam_pts, _terrain_pts, map_size) in enumerate(frames_data):
            title = f"{label} | {env_name}/{traj_name} | frame={start+i}"
            img = _render_measurement_pointcloud_frame(
                cam_pts=cam_pts,
                map_size=map_size,
                title=title,
                zmin=zmin,
                zmax=zmax,
                point_size=point_size,
            )
            imgs.append(img)

        out_name = f"{dataset_name}_{label}_{env_name}_{traj_name}_pointcloud.gif"
        out_path = os.path.join(out_dir, out_name)
        duration = _save_gif(
            imgs=imgs,
            out_path=out_path,
            fps=fps,
            frame_duration_ms=frame_duration_ms,
        )
        if maybe_score is None:
            print(f"[gif-pc] {label}: {out_path} ({len(imgs)} frames, duration={duration:.3f}s)")
        else:
            print(
                f"[gif-pc] {label}: {out_path} ({len(imgs)} frames, "
                f"duration={duration:.3f}s, score={maybe_score:.4f})"
            )


def export_terrain_gifs(
    data_dir,
    out_dir,
    split="train",
    terrain_types=("slope", "stairs", "obstacles"),
    frames_per_gif=30,
    fps=6,
    frame_duration_ms=None,
    scan_trajs=800,
    seed=0,
):
    traj_dirs = _find_traj_dirs(data_dir, split=split)
    if not traj_dirs:
        raise FileNotFoundError(f"No traj_* found under: {data_dir}")

    selected = _select_trajectories_by_type(
        traj_dirs=traj_dirs,
        terrain_types=terrain_types,
        scan_trajs=scan_trajs,
        seed=seed,
        frames_per_gif=frames_per_gif,
    )
    if not selected:
        raise RuntimeError("No trajectories selected for requested terrain types.")

    os.makedirs(out_dir, exist_ok=True)
    dataset_name = os.path.basename(os.path.abspath(data_dir.rstrip("/")))
    for terrain_type in terrain_types:
        picked = selected.get(terrain_type)
        if picked is None:
            print(f"[warn] no trajectory found for terrain type: {terrain_type}")
            continue
        _, traj_dir, start, score_dict = picked
        frame_paths = _load_frames(traj_dir)
        n_take = min(int(frames_per_gif), len(frame_paths))
        start = min(max(0, int(start)), max(0, len(frame_paths) - n_take))
        frame_paths = frame_paths[start:start + n_take]

        frame_maps = [_load_maps_for_frame(p) for p in frame_paths]
        valid_vals = []
        for d in frame_maps:
            vm = d["meas_h"][d["meas_m"] > 0.5]
            vg = d["gt_h"][d["gt_m"] > 0.5]
            if vm.size > 0:
                valid_vals.append(vm)
            if vg.size > 0:
                valid_vals.append(vg)
        if valid_vals:
            all_vals = np.concatenate(valid_vals, axis=0)
            vmin = float(np.percentile(all_vals, 1))
            vmax = float(np.percentile(all_vals, 99))
            if vmax <= vmin:
                vmin = float(np.min(all_vals))
                vmax = float(np.max(all_vals) + 1e-3)
        else:
            vmin, vmax = -1.0, 1.0

        imgs = []
        traj_name = os.path.basename(traj_dir)
        env_name = os.path.basename(os.path.dirname(traj_dir)) if os.path.basename(os.path.dirname(traj_dir)).startswith("env_") else "env_unknown"
        for i, d in enumerate(frame_maps):
            title = f"{terrain_type} | {env_name}/{traj_name} | frame={start+i}"
            img = _render_pair_frame(
                d["meas_h"],
                d["meas_m"],
                d["gt_h"],
                d["gt_m"],
                d["map_size"],
                title=title,
                vmin=vmin,
                vmax=vmax,
            )
            imgs.append(img)

        out_name = f"{dataset_name}_{terrain_type}_{env_name}_{traj_name}.gif"
        out_path = os.path.join(out_dir, out_name)
        duration = _save_gif(
            imgs=imgs,
            out_path=out_path,
            fps=fps,
            frame_duration_ms=frame_duration_ms,
        )
        print(
            f"[gif] {terrain_type}: {out_path} ({len(imgs)} frames, "
            f"duration={duration:.3f}s, score={score_dict.get(terrain_type, float('nan')):.4f})"
        )


def visualize(
    data_dir,
    frame_idx=None,
    auto=True,
    pause_ms=200,
    max_frames=None,
    single_window=False,
    follow_robot=True,
    traj_idx=None,
    local_frame=True,
    point_size=2.0,
    show_axis=True,
):
    import open3d as o3d

    resolved_dir = _resolve_data_dir(data_dir, traj_idx=traj_idx)
    paths = _load_frames(resolved_dir)
    if frame_idx is not None:
        if frame_idx < 0 or frame_idx >= len(paths):
            raise IndexError(f"frame_idx {frame_idx} out of range (0..{len(paths)-1})")
        paths = [paths[frame_idx]]

    if max_frames is not None:
        paths = paths[:max_frames]

    cam_pts, terrain_pts, robot_pos, robot_quat = _load_clouds(paths[0])

    def _quat_to_rot(q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-8:
            return np.eye(3, dtype=np.float64)
        q = q / n
        x, y, z, w = q
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    def _to_local(points, pos, quat):
        if points is None or points.shape[0] == 0 or pos is None or quat is None:
            return points
        R = _quat_to_rot(quat)
        return (R.T @ (points - pos).T).T

    if local_frame:
        cam_pts = _to_local(cam_pts, robot_pos, robot_quat)
        terrain_pts = _to_local(terrain_pts, robot_pos, robot_quat)

    cam_pc = o3d.geometry.PointCloud()
    cam_pc.points = o3d.utility.Vector3dVector(cam_pts)
    if cam_pts.shape[0] > 0:
        cam_pc.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.1, 0.6, 1.0]]), (cam_pts.shape[0], 1))
        )

    terrain_pc = o3d.geometry.PointCloud()
    terrain_pc.points = o3d.utility.Vector3dVector(terrain_pts)
    if terrain_pts.shape[0] > 0:
        terrain_pc.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[1.0, 0.6, 0.1]]), (terrain_pts.shape[0], 1))
        )

    follow_initialized = False

    def _update_view(vis, lookat):
        nonlocal follow_initialized
        if local_frame:
            return
        if not follow_robot or lookat is None:
            return
        vc = vis.get_view_control()
        if not follow_initialized:
            front = np.array([0.0, -1.0, -0.8], dtype=np.float32)
            front = front / (np.linalg.norm(front) + 1e-8)
            vc.set_front(front.tolist())
            vc.set_up([0.0, 0.0, 1.0])
            vc.set_zoom(0.5)
            follow_initialized = True
        vc.set_lookat(lookat.tolist())

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    if single_window:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="cams + terrain", width=960, height=720)
        vis.get_render_option().point_size = float(point_size)
        if cam_pts.shape[0] > 0:
            vis.add_geometry(cam_pc)
        if terrain_pts.shape[0] > 0:
            vis.add_geometry(terrain_pc)
        if show_axis:
            vis.add_geometry(axis)
        _update_view(vis, robot_pos)
        vis.reset_view_point(True)
        for path in paths:
            cam_pts, terrain_pts, robot_pos, robot_quat = _load_clouds(path)
            if local_frame:
                cam_pts = _to_local(cam_pts, robot_pos, robot_quat)
                terrain_pts = _to_local(terrain_pts, robot_pos, robot_quat)
            cam_pc.points = o3d.utility.Vector3dVector(cam_pts)
            if cam_pts.shape[0] > 0:
                cam_pc.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[0.1, 0.6, 1.0]]), (cam_pts.shape[0], 1))
                )
            terrain_pc.points = o3d.utility.Vector3dVector(terrain_pts)
            if terrain_pts.shape[0] > 0:
                terrain_pc.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[1.0, 0.6, 0.1]]), (terrain_pts.shape[0], 1))
                )
            if cam_pts.shape[0] > 0:
                vis.update_geometry(cam_pc)
            if terrain_pts.shape[0] > 0:
                vis.update_geometry(terrain_pc)
            if show_axis:
                vis.update_geometry(axis)
            _update_view(vis, robot_pos)
            vis.poll_events()
            vis.update_renderer()
            if auto:
                time.sleep(pause_ms / 1000.0)
            else:
                input("Press Enter for next frame...")
        vis.destroy_window()
        return

    vis_cam = o3d.visualization.Visualizer()
    vis_terrain = o3d.visualization.Visualizer()
    vis_cam.create_window(window_name="cams", width=960, height=720, left=50, top=50)
    vis_terrain.create_window(window_name="terrain", width=960, height=720, left=1050, top=50)
    vis_cam.get_render_option().point_size = float(point_size)
    vis_terrain.get_render_option().point_size = float(point_size)
    if cam_pts.shape[0] > 0:
        vis_cam.add_geometry(cam_pc)
    if terrain_pts.shape[0] > 0:
        vis_terrain.add_geometry(terrain_pc)
    if show_axis:
        vis_cam.add_geometry(axis)
        vis_terrain.add_geometry(axis)
    _update_view(vis_cam, robot_pos)
    _update_view(vis_terrain, robot_pos)
    vis_cam.reset_view_point(True)
    vis_terrain.reset_view_point(True)

    for path in paths:
        cam_pts, terrain_pts, robot_pos, robot_quat = _load_clouds(path)
        if local_frame:
            cam_pts = _to_local(cam_pts, robot_pos, robot_quat)
            terrain_pts = _to_local(terrain_pts, robot_pos, robot_quat)
        cam_pc.points = o3d.utility.Vector3dVector(cam_pts)
        if cam_pts.shape[0] > 0:
            cam_pc.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0.1, 0.6, 1.0]]), (cam_pts.shape[0], 1))
            )
        terrain_pc.points = o3d.utility.Vector3dVector(terrain_pts)
        if terrain_pts.shape[0] > 0:
            terrain_pc.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[1.0, 0.6, 0.1]]), (terrain_pts.shape[0], 1))
            )
        if cam_pts.shape[0] > 0:
            vis_cam.update_geometry(cam_pc)
        if terrain_pts.shape[0] > 0:
            vis_terrain.update_geometry(terrain_pc)
        if show_axis:
            vis_cam.update_geometry(axis)
            vis_terrain.update_geometry(axis)
        _update_view(vis_cam, robot_pos)
        _update_view(vis_terrain, robot_pos)
        vis_cam.poll_events()
        vis_terrain.poll_events()
        vis_cam.update_renderer()
        vis_terrain.update_renderer()
        if auto:
            time.sleep(pause_ms / 1000.0)
        else:
            input("Press Enter for next frame...")

    vis_cam.destroy_window()
    vis_terrain.destroy_window()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing frame_*.npz")
    parser.add_argument("--traj", type=int, default=None, help="Trajectory index (if data_dir is env_*)")
    parser.add_argument("--frame", type=int, default=None, help="Only visualize a single frame index")
    parser.add_argument("--auto", action="store_true", default=False, help="Auto-advance frames")
    parser.add_argument("--pause_ms", type=int, default=200, help="Pause between frames when --auto")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames")
    parser.add_argument("--single_window", action="store_true", default=False, help="Show cams+terrain in one window")
    parser.add_argument("--follow_robot", action="store_true", default=True, help="Follow robot position (default on)")
    parser.add_argument("--no_follow_robot", action="store_false", dest="follow_robot", help="Disable follow")
    parser.add_argument("--local_frame", action="store_true", default=True, help="Show points in robot local frame (default on)")
    parser.add_argument("--world_frame", action="store_false", dest="local_frame", help="Show points in world frame")
    parser.add_argument("--point_size", type=float, default=2.0, help="Point size for rendering")
    parser.add_argument("--no_axis", action="store_false", dest="show_axis", help="Hide coordinate axis")
    parser.add_argument(
        "--export_terrain_gifs",
        action="store_true",
        default=False,
        help="Export side-by-side Measurement/GT GIFs for multiple terrain types.",
    )
    parser.add_argument(
        "--export_pointcloud_gifs",
        action="store_true",
        default=False,
        help="Export side-by-side cam_points/terrain point cloud GIFs for typical terrain types.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split to scan when exporting GIFs: train/val")
    parser.add_argument(
        "--terrain_types",
        type=str,
        default="slope,rough_flat,stairs,obstacles",
        help="Comma-separated terrain types to export: slope,rough_flat,stairs,obstacles (rough is alias of rough_flat).",
    )
    parser.add_argument("--frames_per_gif", type=int, default=30, help="Number of continuous frames in each GIF")
    parser.add_argument("--fps", type=int, default=4, help="GIF frame rate (used when --frame_duration_ms is not set)")
    parser.add_argument(
        "--frame_duration_ms",
        type=int,
        default=1200,
        help="Frame duration for GIF in ms. Use this to slow down playback directly.",
    )
    parser.add_argument("--scan_trajs", type=int, default=800, help="How many trajectories to scan for type selection")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when scanning trajectories")
    parser.add_argument("--max_points", type=int, default=20000, help="Max points rendered per cloud per frame for pointcloud GIF mode")
    parser.add_argument(
        "--random_num_gifs",
        type=int,
        default=0,
        help="If >0 in --export_pointcloud_gifs mode, skip terrain-type selection and export N random trajectory GIFs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for GIFs. Default: parent folder of dataset root (usually /home/zdd/RL/dataset).",
    )
    args = parser.parse_args()

    if args.export_terrain_gifs or args.export_pointcloud_gifs:
        terrain_types = [x.strip() for x in args.terrain_types.split(",") if x.strip()]
        if args.out_dir is None:
            data_abs = os.path.abspath(args.data_dir)
            parent = os.path.dirname(data_abs)
            if os.path.basename(data_abs) in ("train", "val"):
                parent = os.path.dirname(parent)
            out_dir = parent
        else:
            out_dir = args.out_dir
        if args.export_pointcloud_gifs:
            export_terrain_pointcloud_gifs(
                data_dir=args.data_dir,
                out_dir=out_dir,
                split=args.split,
                terrain_types=terrain_types,
                frames_per_gif=args.frames_per_gif,
                fps=args.fps,
                frame_duration_ms=args.frame_duration_ms,
                scan_trajs=args.scan_trajs,
                seed=args.seed,
                max_points=args.max_points,
                point_size=args.point_size,
                random_num_gifs=args.random_num_gifs,
            )
        else:
            export_terrain_gifs(
                data_dir=args.data_dir,
                out_dir=out_dir,
                split=args.split,
                terrain_types=terrain_types,
                frames_per_gif=args.frames_per_gif,
                fps=args.fps,
                frame_duration_ms=args.frame_duration_ms,
                scan_trajs=args.scan_trajs,
                seed=args.seed,
            )
        return

    visualize(
        data_dir=args.data_dir,
        frame_idx=args.frame,
        auto=args.auto,
        pause_ms=args.pause_ms,
        max_frames=args.max_frames,
        single_window=args.single_window,
        follow_robot=args.follow_robot,
        local_frame=args.local_frame,
        point_size=args.point_size,
        show_axis=args.show_axis,
        traj_idx=args.traj,
    )


if __name__ == "__main__":
    main()
