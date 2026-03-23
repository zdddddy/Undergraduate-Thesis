# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.camera_profiles import apply_pointcloud_camera_profile, get_pointcloud_camera_profile
from legged_gym.utils.terrain_labels import (
    build_proportions_cumsum,
    make_traj_label_record,
    terrain_metadata_from_indices,
)
import os
import sys
import time
import json

import isaacgym
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import task_registry

import numpy as np
import torch


def get_collect_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Task name."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume from checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Experiment name override"},
        {"name": "--run_name", "type": str, "help": "Run name override"},
        {"name": "--load_run", "type": str, "help": "Run to load when resume=True. -1 loads last run"},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint number. -1 loads last checkpoint"},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Disable viewer"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device"},
        {"name": "--num_envs", "type": int, "default": 100, "help": "Number of environments"},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--frames_per_env", "type": int, "default": 2000, "help": "Number of frames to save per env"},
        {"name": "--frame_interval", "type": int, "default": 10, "help": "Steps between captures"},
        {"name": "--out_dir", "type": str, "default": "dataset/urban_200k", "help": "Output directory"},
        {"name": "--terrain_patch_size", "type": float, "default": 3.2, "help": "Patch side length (m)"},
        {"name": "--max_terrain_points", "type": int, "default": 15000, "help": "Max terrain points saved"},
        {"name": "--gt_map_resolution", "type": float, "default": 0.05, "help": "GT heightmap resolution (m/cell)"},
        {"name": "--gt_world_frame", "action": "store_true", "default": False, "help": "Store GT heightmap in world-aligned patch"},
        {"name": "--gt_gravity_aligned", "action": "store_true", "default": True, "help": "For local-frame GT, use yaw-only gravity-aligned frame (default on)"},
        {"name": "--gt_full_6dof_frame", "action": "store_false", "dest": "gt_gravity_aligned", "help": "For local-frame GT, use full 6DoF local frame (legacy)"},
        {
            "name": "--gt_sampler_method",
            "type": str,
            "default": "mesh_ray",
            "help": "GT sampler: mesh_ray (recommended) or heightfield_min (legacy).",
        },
        {
            "name": "--gt_ray_neighbor_radius",
            "type": int,
            "default": 2,
            "help": "Neighbor cell radius for mesh_ray candidate triangles.",
        },
        {
            "name": "--gt_ray_bary_eps",
            "type": float,
            "default": 1e-6,
            "help": "Barycentric epsilon for point-in-triangle test in mesh_ray sampler.",
        },
    ]
    args = gymutil.parse_arguments(description="Collect large pointcloud dataset", custom_parameters=custom_parameters)
    # Keep default behavior deterministic: use gravity-aligned local GT unless explicitly overridden.
    if "--gt_full_6dof_frame" in sys.argv:
        args.gt_gravity_aligned = False
    elif "--gt_gravity_aligned" in sys.argv:
        args.gt_gravity_aligned = True
    else:
        args.gt_gravity_aligned = True
    args.gt_sampler_method = str(args.gt_sampler_method).strip().lower()
    if args.gt_sampler_method not in {"mesh_ray", "heightfield_min"}:
        raise ValueError(
            f"Unsupported --gt_sampler_method={args.gt_sampler_method}; "
            "expected one of: mesh_ray, heightfield_min"
        )
    args.gt_ray_neighbor_radius = max(0, int(args.gt_ray_neighbor_radius))
    args.gt_ray_bary_eps = float(max(args.gt_ray_bary_eps, 1e-9))
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def _build_terrain_vertices_world(env, env_cfg):
    if env_cfg.terrain.mesh_type == "trimesh":
        vertices = env.terrain.vertices
    elif env_cfg.terrain.mesh_type == "heightfield":
        from isaacgym import terrain_utils
        vertices, _ = terrain_utils.convert_heightfield_to_trimesh(
            env.terrain.height_field_raw,
            env_cfg.terrain.horizontal_scale,
            env_cfg.terrain.vertical_scale,
            env_cfg.terrain.slope_treshold,
        )
    else:
        return None
    vertices_world = np.asarray(vertices, dtype=np.float32).copy()
    vertices_world[:, 0] -= env_cfg.terrain.border_size
    vertices_world[:, 1] -= env_cfg.terrain.border_size
    return vertices_world


def _filter_pointcloud(pointcloud, cam_pos, far_plane):
    pts = pointcloud.reshape(-1, 3)
    cam = cam_pos.reshape(1, 3)
    dist = np.linalg.norm(pts - cam, axis=1)
    mask = dist < (far_plane - 1e-3)
    return pts[mask]


def _quat_to_rotmat_xyzw(q):
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        return np.eye(3, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / n
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
        dtype=np.float32,
    )


def _quat_to_yaw_rotmat_xyzw(q):
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        return np.eye(3, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / n
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _build_heightfield_mesh_vertex_grids(height_samples, horizontal_scale, vertical_scale, slope_threshold, border_size):
    hf = height_samples.astype(np.float32)
    rows, cols = hf.shape

    y = np.linspace(0.0, (cols - 1) * horizontal_scale, cols, dtype=np.float32)
    x = np.linspace(0.0, (rows - 1) * horizontal_scale, rows, dtype=np.float32)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold_scaled = float(slope_threshold) * float(horizontal_scale) / max(float(vertical_scale), 1e-8)
        move_x = np.zeros((rows, cols), dtype=np.float32)
        move_y = np.zeros((rows, cols), dtype=np.float32)
        move_corners = np.zeros((rows, cols), dtype=np.float32)

        move_x[:rows - 1, :] += (hf[1:rows, :] - hf[:rows - 1, :] > slope_threshold_scaled)
        move_x[1:rows, :] -= (hf[:rows - 1, :] - hf[1:rows, :] > slope_threshold_scaled)
        move_y[:, :cols - 1] += (hf[:, 1:cols] - hf[:, :cols - 1] > slope_threshold_scaled)
        move_y[:, 1:cols] -= (hf[:, :cols - 1] - hf[:, 1:cols] > slope_threshold_scaled)
        move_corners[:rows - 1, :cols - 1] += (hf[1:rows, 1:cols] - hf[:rows - 1, :cols - 1] > slope_threshold_scaled)
        move_corners[1:rows, 1:cols] -= (hf[:rows - 1, :cols - 1] - hf[1:rows, 1:cols] > slope_threshold_scaled)

        xx = xx + (move_x + move_corners * (move_x == 0.0)) * float(horizontal_scale)
        yy = yy + (move_y + move_corners * (move_y == 0.0)) * float(horizontal_scale)

    vx = xx.astype(np.float32) - float(border_size)
    vy = yy.astype(np.float32) - float(border_size)
    vz = hf.astype(np.float32) * float(vertical_scale)
    return vx, vy, vz


def _heightfield_min_world_z(world_x, world_y, sampler):
    height_samples = sampler['height_samples']
    border_size = sampler['border_size']
    hscale = sampler['horizontal_scale']
    vscale = sampler['vertical_scale']
    rows, cols = height_samples.shape

    px_f = (world_x + border_size) / hscale
    py_f = (world_y + border_size) / hscale
    px0 = np.floor(px_f).astype(np.int64)
    py0 = np.floor(py_f).astype(np.int64)
    valid = (px0 >= 0) & (px0 < rows - 1) & (py0 >= 0) & (py0 < cols - 1)

    px = np.clip(px0, 0, rows - 2)
    py = np.clip(py0, 0, cols - 2)
    h1 = height_samples[px, py]
    h2 = height_samples[px + 1, py]
    h3 = height_samples[px, py + 1]
    world_z = np.minimum(np.minimum(h1, h2), h3) * vscale
    return world_z.astype(np.float32), valid


def _update_hit_with_triangle(
    qx,
    qy,
    tri_x0,
    tri_y0,
    tri_z0,
    tri_x1,
    tri_y1,
    tri_z1,
    tri_x2,
    tri_y2,
    tri_z2,
    bary_eps,
    hit_z,
    hit_valid,
):
    det = (tri_y1 - tri_y2) * (tri_x0 - tri_x2) + (tri_x2 - tri_x1) * (tri_y0 - tri_y2)
    non_deg = np.abs(det) > bary_eps
    if not np.any(non_deg):
        return

    inv_det = np.zeros_like(det, dtype=np.float32)
    inv_det[non_deg] = 1.0 / det[non_deg]

    w0 = ((tri_y1 - tri_y2) * (qx - tri_x2) + (tri_x2 - tri_x1) * (qy - tri_y2)) * inv_det
    w1 = ((tri_y2 - tri_y0) * (qx - tri_x2) + (tri_x0 - tri_x2) * (qy - tri_y2)) * inv_det
    w2 = 1.0 - w0 - w1

    inside = (
        non_deg
        & (w0 >= -bary_eps)
        & (w1 >= -bary_eps)
        & (w2 >= -bary_eps)
        & (w0 <= 1.0 + bary_eps)
        & (w1 <= 1.0 + bary_eps)
        & (w2 <= 1.0 + bary_eps)
    )
    if not np.any(inside):
        return

    z = w0 * tri_z0 + w1 * tri_z1 + w2 * tri_z2
    better = inside & ((~hit_valid) | (z > hit_z))
    if not np.any(better):
        return
    hit_z[better] = z[better]
    hit_valid[better] = True


def _mesh_ray_world_z(world_x, world_y, sampler):
    vx = sampler['mesh_vx']
    vy = sampler['mesh_vy']
    vz = sampler['mesh_vz']
    rows = int(sampler['rows'])
    cols = int(sampler['cols'])
    border_size = float(sampler['border_size'])
    hscale = float(sampler['horizontal_scale'])
    neighbor_offsets = sampler['ray_neighbor_offsets']
    bary_eps = float(sampler['ray_bary_eps'])

    qx = world_x.reshape(-1).astype(np.float32)
    qy = world_y.reshape(-1).astype(np.float32)
    n = qx.shape[0]

    base_i = np.floor((qx + border_size) / hscale).astype(np.int64)
    base_j = np.floor((qy + border_size) / hscale).astype(np.int64)
    in_domain = (base_i >= -int(sampler['ray_neighbor_radius'])) & (base_i < rows - 1 + int(sampler['ray_neighbor_radius'])) & (
        base_j >= -int(sampler['ray_neighbor_radius'])
    ) & (base_j < cols - 1 + int(sampler['ray_neighbor_radius']))

    hit_z = np.full((n,), -np.inf, dtype=np.float32)
    hit_valid = np.zeros((n,), dtype=bool)

    for di, dj in neighbor_offsets:
        ci = base_i + int(di)
        cj = base_j + int(dj)
        cell_valid = in_domain & (ci >= 0) & (ci < rows - 1) & (cj >= 0) & (cj < cols - 1)
        if not np.any(cell_valid):
            continue

        i_sel = ci[cell_valid]
        j_sel = cj[cell_valid]
        qx_sel = qx[cell_valid]
        qy_sel = qy[cell_valid]

        x00 = vx[i_sel, j_sel]
        y00 = vy[i_sel, j_sel]
        z00 = vz[i_sel, j_sel]
        x01 = vx[i_sel, j_sel + 1]
        y01 = vy[i_sel, j_sel + 1]
        z01 = vz[i_sel, j_sel + 1]
        x10 = vx[i_sel + 1, j_sel]
        y10 = vy[i_sel + 1, j_sel]
        z10 = vz[i_sel + 1, j_sel]
        x11 = vx[i_sel + 1, j_sel + 1]
        y11 = vy[i_sel + 1, j_sel + 1]
        z11 = vz[i_sel + 1, j_sel + 1]

        local_hit_z = hit_z[cell_valid].copy()
        local_hit_valid = hit_valid[cell_valid].copy()

        # Triangle A: (i,j) -> (i+1,j+1) -> (i,j+1)
        _update_hit_with_triangle(
            qx_sel,
            qy_sel,
            x00,
            y00,
            z00,
            x11,
            y11,
            z11,
            x01,
            y01,
            z01,
            bary_eps,
            local_hit_z,
            local_hit_valid,
        )
        # Triangle B: (i,j) -> (i+1,j) -> (i+1,j+1)
        _update_hit_with_triangle(
            qx_sel,
            qy_sel,
            x00,
            y00,
            z00,
            x10,
            y10,
            z10,
            x11,
            y11,
            z11,
            bary_eps,
            local_hit_z,
            local_hit_valid,
        )

        hit_z[cell_valid] = local_hit_z
        hit_valid[cell_valid] = local_hit_valid

    world_z = hit_z.reshape(world_x.shape)
    valid = hit_valid.reshape(world_x.shape)
    return world_z.astype(np.float32), valid


def _prepare_gt_sampler(
    env,
    env_cfg,
    map_size,
    resolution,
    world_frame=False,
    gravity_aligned=True,
    sampler_method='mesh_ray',
    ray_neighbor_radius=2,
    ray_bary_eps=1e-6,
):
    if resolution <= 0.0:
        raise ValueError("gt_map_resolution must be positive.")
    grid_size = int(round(float(map_size) / float(resolution)))
    if grid_size <= 0:
        raise ValueError("Invalid GT heightmap grid_size.")
    centers = (np.arange(grid_size, dtype=np.float32) + 0.5) * float(resolution) - 0.5 * float(map_size)
    grid_x, grid_y = np.meshgrid(centers, centers, indexing='ij')

    height_samples = env.height_samples.detach().cpu().numpy().astype(np.float32)
    sampler_method = str(sampler_method).strip().lower()
    if sampler_method not in {"mesh_ray", "heightfield_min"}:
        raise ValueError(f"Unsupported gt_sampler_method={sampler_method}")

    sampler = {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'height_samples': height_samples,
        'horizontal_scale': float(env_cfg.terrain.horizontal_scale),
        'vertical_scale': float(env_cfg.terrain.vertical_scale),
        'border_size': float(env_cfg.terrain.border_size),
        'world_frame': bool(world_frame),
        'gravity_aligned': bool(gravity_aligned),
        'resolution': float(resolution),
        'map_size': float(map_size),
        'sampler_method': sampler_method,
        'rows': int(height_samples.shape[0]),
        'cols': int(height_samples.shape[1]),
    }

    if sampler_method == 'mesh_ray':
        vx, vy, vz = _build_heightfield_mesh_vertex_grids(
            height_samples=height_samples,
            horizontal_scale=sampler['horizontal_scale'],
            vertical_scale=sampler['vertical_scale'],
            slope_threshold=float(env_cfg.terrain.slope_treshold),
            border_size=sampler['border_size'],
        )
        radius = max(0, int(ray_neighbor_radius))
        offsets = [(di, dj) for di in range(-radius, radius + 1) for dj in range(-radius, radius + 1)]
        sampler.update(
            {
                'mesh_vx': vx,
                'mesh_vy': vy,
                'mesh_vz': vz,
                'ray_neighbor_radius': radius,
                'ray_neighbor_offsets': offsets,
                'ray_bary_eps': float(max(ray_bary_eps, 1e-9)),
            }
        )

    return sampler


def _sample_gt_heightmap(robot_pos, robot_quat, sampler):
    grid_x = sampler['grid_x']
    grid_y = sampler['grid_y']
    world_frame = sampler['world_frame']

    if world_frame:
        world_x = robot_pos[0] + grid_x
        world_y = robot_pos[1] + grid_y
        rot = None
    else:
        if sampler.get('gravity_aligned', True):
            rot = _quat_to_yaw_rotmat_xyzw(robot_quat)
        else:
            rot = _quat_to_rotmat_xyzw(robot_quat)
        local_xy0 = np.stack(
            [grid_x, grid_y, np.zeros_like(grid_x, dtype=np.float32)],
            axis=-1,
        ).reshape(-1, 3)
        world_xy0 = (rot @ local_xy0.T).T + robot_pos.reshape(1, 3)
        world_x = world_xy0[:, 0].reshape(grid_x.shape)
        world_y = world_xy0[:, 1].reshape(grid_y.shape)

    if sampler['sampler_method'] == 'mesh_ray':
        world_z, valid = _mesh_ray_world_z(world_x, world_y, sampler)
        # Robust fallback for rare misses near near-vertical facets.
        if not np.all(valid):
            world_z_fallback, valid_fallback = _heightfield_min_world_z(world_x, world_y, sampler)
            use_fb = (~valid) & valid_fallback
            world_z[use_fb] = world_z_fallback[use_fb]
            valid = valid | valid_fallback
    else:
        world_z, valid = _heightfield_min_world_z(world_x, world_y, sampler)

    world_z_safe = world_z.astype(np.float32).copy()
    world_z_safe[~valid] = 0.0

    if world_frame:
        gt_h = world_z_safe
    else:
        world_pts = np.stack([world_x, world_y, world_z_safe], axis=-1).reshape(-1, 3)
        local_pts = (rot.T @ (world_pts - robot_pos.reshape(1, 3)).T).T
        gt_h = local_pts[:, 2].reshape(grid_x.shape).astype(np.float32)

    gt_m = valid.astype(np.float32)
    gt_h[~valid] = 0.0
    return gt_h, gt_m


def collect(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.load_run is not None:
        try:
            load_run_int = int(args.load_run)
            if str(load_run_int) == str(args.load_run).strip():
                args.load_run = load_run_int
        except (ValueError, TypeError):
            pass

    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.use_warp = True
    env_cfg.sensor.add_depth = True
    apply_pointcloud_camera_profile(env_cfg.sensor, get_pointcloud_camera_profile(args.task))
    env_cfg.terrain.mesh_type = "trimesh"

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = bool(args.resume)
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.run_name is not None:
        train_cfg.runner.run_name = args.run_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    out_dir = args.out_dir
    if out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "pc_collect_large", ts)
    os.makedirs(out_dir, exist_ok=True)

    terrain_vertices_world = _build_terrain_vertices_world(env, env_cfg)
    if terrain_vertices_world is None:
        raise RuntimeError("Terrain mesh_type must be 'trimesh' or 'heightfield' to collect terrain patch.")

    far_plane = float(env_cfg.sensor.depth_camera_config.far_plane)
    half = 0.5 * float(args.terrain_patch_size)
    gt_sampler = _prepare_gt_sampler(
        env=env,
        env_cfg=env_cfg,
        map_size=args.terrain_patch_size,
        resolution=args.gt_map_resolution,
        world_frame=args.gt_world_frame,
        gravity_aligned=args.gt_gravity_aligned,
        sampler_method=args.gt_sampler_method,
        ray_neighbor_radius=args.gt_ray_neighbor_radius,
        ray_bary_eps=args.gt_ray_bary_eps,
    )
    print(
        f"[collect_large] GT sampler method={gt_sampler['sampler_method']} "
        f"(ray_radius={gt_sampler.get('ray_neighbor_radius', 0)}, "
        f"ray_bary_eps={gt_sampler.get('ray_bary_eps', 0.0):.2e})"
    )

    terrain_props = getattr(env_cfg.terrain, "terrain_proportions", [1.0])
    terrain_props_cumsum = build_proportions_cumsum(terrain_props)
    terrain_num_cols = int(getattr(env_cfg.terrain, "num_cols", 1))
    terrain_num_rows = int(getattr(env_cfg.terrain, "num_rows", 1))
    labels_by_traj = {}

    def _traj_key(env_id, traj_id):
        return f"env_{env_id:04d}/traj_{traj_id:04d}"

    def _write_labels_json():
        out_path = os.path.join(out_dir, "terrain_labels.json")
        payload = {
            "meta": {
                "task": str(args.task),
                "label_source": "collect_large_env_terrain_types",
                "terrain_num_rows": terrain_num_rows,
                "terrain_num_cols": terrain_num_cols,
                "terrain_proportions": [float(x) for x in terrain_props],
                "frames_per_env": int(args.frames_per_env),
                "frame_interval": int(args.frame_interval),
            },
            "labels_by_traj": labels_by_traj,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        return out_path

    num_envs = env_cfg.env.num_envs
    frames_per_env = int(args.frames_per_env)

    env_dirs = []
    traj_dirs = []
    traj_idx = np.zeros(num_envs, dtype=np.int64)
    traj_frame_idx = np.zeros(num_envs, dtype=np.int64)
    frames_saved = np.zeros(num_envs, dtype=np.int64)

    for env_id in range(num_envs):
        env_dir = os.path.join(out_dir, f"env_{env_id:04d}")
        os.makedirs(env_dir, exist_ok=True)
        env_dirs.append(env_dir)
        tdir = os.path.join(env_dir, f"traj_{traj_idx[env_id]:04d}")
        os.makedirs(tdir, exist_ok=True)
        traj_dirs.append(tdir)

    step = 0
    labels_json_path = ""
    try:
        while np.any(frames_saved < frames_per_env):
            with torch.no_grad():
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())

            # handle resets first to avoid cross-trajectory contamination
            done_envs = torch.nonzero(dones).flatten().tolist()
            for env_id in done_envs:
                if frames_saved[env_id] < frames_per_env:
                    traj_idx[env_id] += 1
                    traj_frame_idx[env_id] = 0
                    new_dir = os.path.join(env_dirs[env_id], f"traj_{traj_idx[env_id]:04d}")
                    os.makedirs(new_dir, exist_ok=True)
                    traj_dirs[env_id] = new_dir

            if step % args.frame_interval == 0:
                # batch transfer to CPU to avoid per-env syncs
                all_robot_pos = env.root_states[:, :3].detach().cpu().numpy()
                all_robot_quat = env.root_states[:, 3:7].detach().cpu().numpy()
                all_depth_images = env.depth_images[:, :, 0].detach().cpu().numpy()
                all_sensor_pos = env.sensor_pos_tensor.detach().cpu().numpy()
                all_terrain_cols = env.terrain_types.detach().cpu().numpy().astype(np.int64)
                all_terrain_rows = env.terrain_levels.detach().cpu().numpy().astype(np.int64)

                for env_id in range(num_envs):
                    if frames_saved[env_id] >= frames_per_env:
                        continue
                    robot_pos = all_robot_pos[env_id]
                    robot_quat = all_robot_quat[env_id]

                    mask = (
                        (np.abs(terrain_vertices_world[:, 0] - robot_pos[0]) <= half)
                        & (np.abs(terrain_vertices_world[:, 1] - robot_pos[1]) <= half)
                    )
                    terrain_patch = terrain_vertices_world[mask]
                    if args.max_terrain_points and terrain_patch.shape[0] > args.max_terrain_points:
                        stride = int(np.ceil(terrain_patch.shape[0] / args.max_terrain_points))
                        terrain_patch = terrain_patch[::stride]

                    cam_points_list = []
                    for sensor_id in range(env_cfg.sensor.num_sensors):
                        pc = all_depth_images[env_id, sensor_id].astype(np.float32)
                        cam_pos = all_sensor_pos[env_id, sensor_id].astype(np.float32)
                        filtered = _filter_pointcloud(pc, cam_pos, far_plane)
                        if filtered.size > 0:
                            cam_points_list.append(filtered)
                    if cam_points_list:
                        cam_points_merged = np.concatenate(cam_points_list, axis=0)
                    else:
                        cam_points_merged = np.zeros((0, 3), dtype=np.float32)
                    gt_h, gt_m = _sample_gt_heightmap(robot_pos, robot_quat, gt_sampler)

                    terrain_meta = terrain_metadata_from_indices(
                        col_idx=int(all_terrain_cols[env_id]),
                        row_idx=int(all_terrain_rows[env_id]),
                        num_cols=terrain_num_cols,
                        num_rows=terrain_num_rows,
                        proportions_cumsum=terrain_props_cumsum,
                    )
                    traj_key = _traj_key(env_id, traj_idx[env_id])
                    if traj_key not in labels_by_traj:
                        rec = make_traj_label_record(terrain_meta, num_frames=0)
                        rec["env_id"] = int(env_id)
                        rec["traj_id"] = int(traj_idx[env_id])
                        labels_by_traj[traj_key] = rec
                    labels_by_traj[traj_key]["num_frames"] += 1

                    save_path = os.path.join(traj_dirs[env_id], f"frame_{traj_frame_idx[env_id]:06d}.npz")
                    np.savez(
                        save_path,
                        terrain=terrain_patch.astype(np.float32),
                        robot_pose7=np.concatenate([robot_pos, robot_quat]).astype(np.float32),
                        cam_points=cam_points_merged.astype(np.float32),
                        gt_heightmap=gt_h.astype(np.float32),
                        gt_mask=gt_m.astype(np.float32),
                        gt_map_size=np.array(gt_sampler['map_size'], dtype=np.float32),
                        gt_resolution=np.array(gt_sampler['resolution'], dtype=np.float32),
                        gt_local_frame=np.array(0 if args.gt_world_frame else 1, dtype=np.uint8),
                        gt_gravity_aligned=np.array(1 if ((not args.gt_world_frame) and args.gt_gravity_aligned) else 0, dtype=np.uint8),
                        gt_sampler_method=np.asarray(gt_sampler['sampler_method']),
                        terrain_col=np.array(terrain_meta["terrain_col"], dtype=np.int32),
                        terrain_row=np.array(terrain_meta["terrain_row"], dtype=np.int32),
                        terrain_choice=np.array(terrain_meta["terrain_choice"], dtype=np.float32),
                        terrain_difficulty=np.array(terrain_meta["terrain_difficulty"], dtype=np.float32),
                        terrain_type=np.asarray(terrain_meta["terrain_type"]),
                        terrain_family=np.asarray(terrain_meta["terrain_family"]),
                    )
                    frames_saved[env_id] += 1
                    traj_frame_idx[env_id] += 1

            step += 1
    except KeyboardInterrupt:
        print("[collect_large] interrupted by user")
    finally:
        labels_json_path = _write_labels_json()

    total_saved = int(frames_saved.sum())
    print(f"[collect_large] saved {total_saved} frames to {out_dir} across {num_envs} envs")
    print(f"[collect_large] terrain labels: {labels_json_path}")


if __name__ == "__main__":
    args = get_collect_args()
    collect(args)
