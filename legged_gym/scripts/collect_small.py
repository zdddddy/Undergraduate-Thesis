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
        {"name": "--headless", "action": "store_true", "default": False, "help": "Disable viewer"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device"},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments"},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--frame_interval", "type": int, "default": 1, "help": "Steps between captures"},
        {"name": "--out_dir", "type": str, "default": None, "help": "Output directory"},
        {"name": "--terrain_patch_size", "type": float, "default": 3.2, "help": "Patch side length (m)"},
        {"name": "--max_terrain_points", "type": int, "default": 15000, "help": "Max terrain points saved"},
        {"name": "--gt_map_resolution", "type": float, "default": 0.05, "help": "GT heightmap resolution (m/cell)"},
        {"name": "--gt_world_frame", "action": "store_true", "default": False, "help": "Store GT heightmap in world-aligned patch"},
        {"name": "--gt_gravity_aligned", "action": "store_true", "default": True, "help": "For local-frame GT, use yaw-only gravity-aligned frame (default on)"},
        {"name": "--gt_full_6dof_frame", "action": "store_false", "dest": "gt_gravity_aligned", "help": "For local-frame GT, use full 6DoF local frame (legacy)"},
    ]
    args = gymutil.parse_arguments(description="Collect small pointcloud dataset", custom_parameters=custom_parameters)
    # Keep default behavior deterministic: use gravity-aligned local GT unless explicitly overridden.
    if "--gt_full_6dof_frame" in sys.argv:
        args.gt_gravity_aligned = False
    elif "--gt_gravity_aligned" in sys.argv:
        args.gt_gravity_aligned = True
    else:
        args.gt_gravity_aligned = True
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


def _prepare_gt_sampler(env, env_cfg, map_size, resolution, world_frame=False, gravity_aligned=True):
    if resolution <= 0.0:
        raise ValueError("gt_map_resolution must be positive.")
    grid_size = int(round(float(map_size) / float(resolution)))
    if grid_size <= 0:
        raise ValueError("Invalid GT heightmap grid_size.")
    centers = (np.arange(grid_size, dtype=np.float32) + 0.5) * float(resolution) - 0.5 * float(map_size)
    grid_x, grid_y = np.meshgrid(centers, centers, indexing='ij')

    height_samples = env.height_samples.detach().cpu().numpy().astype(np.float32)
    return {
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
    }


def _sample_gt_heightmap(robot_pos, robot_quat, sampler):
    grid_x = sampler['grid_x']
    grid_y = sampler['grid_y']
    world_frame = sampler['world_frame']
    height_samples = sampler['height_samples']
    border_size = sampler['border_size']
    hscale = sampler['horizontal_scale']
    vscale = sampler['vertical_scale']
    rows, cols = height_samples.shape

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

    if world_frame:
        gt_h = world_z.astype(np.float32)
    else:
        world_pts = np.stack([world_x, world_y, world_z], axis=-1).reshape(-1, 3)
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

    if args.seed is not None:
        env_cfg.seed = args.seed
    else:
        env_cfg.seed = -1

    # override for small collection (mirrors play.py defaults)
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
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
        out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "pc_collect", ts)
    os.makedirs(out_dir, exist_ok=True)

    env_dir = os.path.join(out_dir, "env_0000")
    os.makedirs(env_dir, exist_ok=True)

    traj_idx = 0
    traj_dir = os.path.join(env_dir, f"traj_{traj_idx:04d}")
    os.makedirs(traj_dir, exist_ok=True)
    traj_frame_idx = 0

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
    )

    terrain_props = getattr(env_cfg.terrain, "terrain_proportions", [1.0])
    terrain_props_cumsum = build_proportions_cumsum(terrain_props)
    terrain_num_cols = int(getattr(env_cfg.terrain, "num_cols", 1))
    terrain_num_rows = int(getattr(env_cfg.terrain, "num_rows", 1))
    labels_by_traj = {}

    def _traj_key(traj_id):
        return f"env_0000/traj_{traj_id:04d}"

    def _write_labels_json():
        out_path = os.path.join(out_dir, "terrain_labels.json")
        payload = {
            "meta": {
                "task": str(args.task),
                "label_source": "collect_small_env_terrain_types",
                "terrain_num_rows": terrain_num_rows,
                "terrain_num_cols": terrain_num_cols,
                "terrain_proportions": [float(x) for x in terrain_props],
                "frame_interval": int(args.frame_interval),
            },
            "labels_by_traj": labels_by_traj,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        return out_path

    step = 0
    saved = 0
    labels_json_path = ""
    try:
        while True:
            with torch.no_grad():
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())

            if env.viewer is not None:
                env.draw_debug_depth_images()

            # handle resets before saving to avoid cross-trajectory contamination
            if bool(dones[0].item()):
                traj_idx += 1
                traj_dir = os.path.join(env_dir, f"traj_{traj_idx:04d}")
                os.makedirs(traj_dir, exist_ok=True)
                traj_frame_idx = 0

            if step % args.frame_interval == 0:
                robot_pos = env.root_states[0, :3].detach().cpu().numpy()
                robot_quat = env.root_states[0, 3:7].detach().cpu().numpy()
                terrain_col = int(env.terrain_types[0].detach().cpu().item())
                terrain_row = int(env.terrain_levels[0].detach().cpu().item())

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
                    pc = env.depth_images[0, sensor_id, 0].detach().cpu().numpy().astype(np.float32)
                    cam_pos = env.sensor_pos_tensor[0, sensor_id].detach().cpu().numpy().astype(np.float32)
                    filtered = _filter_pointcloud(pc, cam_pos, far_plane)
                    if filtered.size > 0:
                        cam_points_list.append(filtered)
                if cam_points_list:
                    cam_points_merged = np.concatenate(cam_points_list, axis=0)
                else:
                    cam_points_merged = np.zeros((0, 3), dtype=np.float32)
                gt_h, gt_m = _sample_gt_heightmap(robot_pos, robot_quat, gt_sampler)
                terrain_meta = terrain_metadata_from_indices(
                    col_idx=terrain_col,
                    row_idx=terrain_row,
                    num_cols=terrain_num_cols,
                    num_rows=terrain_num_rows,
                    proportions_cumsum=terrain_props_cumsum,
                )
                traj_key = _traj_key(traj_idx)
                if traj_key not in labels_by_traj:
                    rec = make_traj_label_record(terrain_meta, num_frames=0)
                    rec["env_id"] = 0
                    rec["traj_id"] = int(traj_idx)
                    labels_by_traj[traj_key] = rec
                labels_by_traj[traj_key]["num_frames"] += 1

                save_path = os.path.join(traj_dir, f"frame_{traj_frame_idx:06d}.npz")
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
                    terrain_col=np.array(terrain_meta["terrain_col"], dtype=np.int32),
                    terrain_row=np.array(terrain_meta["terrain_row"], dtype=np.int32),
                    terrain_choice=np.array(terrain_meta["terrain_choice"], dtype=np.float32),
                    terrain_difficulty=np.array(terrain_meta["terrain_difficulty"], dtype=np.float32),
                    terrain_type=np.asarray(terrain_meta["terrain_type"]),
                    terrain_family=np.asarray(terrain_meta["terrain_family"]),
                )
                traj_frame_idx += 1
                saved += 1

            step += 1
    except KeyboardInterrupt:
        print("[collect_small] interrupted by user")
    finally:
        labels_json_path = _write_labels_json()

    print(f"[collect_small] saved {saved} frames to {out_dir} across {traj_idx + 1} trajectories")
    print(f"[collect_small] terrain labels: {labels_json_path}")


if __name__ == "__main__":
    args = get_collect_args()
    collect(args)
