# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import argparse
import glob
import os

import numpy as np


def _find_env_dirs(root):
    env_dirs = sorted(glob.glob(os.path.join(root, "env_*")))
    if env_dirs:
        return env_dirs
    return [root]


def _find_traj_dirs(env_dir):
    traj_dirs = sorted(glob.glob(os.path.join(env_dir, "traj_*")))
    if traj_dirs:
        return traj_dirs
    frames = glob.glob(os.path.join(env_dir, "frame_*.npz"))
    if frames:
        return [env_dir]
    return []


def _cam_points_count(data):
    if "cam_points" in data:
        return int(data["cam_points"].shape[0])
    count = 0
    for k in ["cam0", "cam1", "cam2", "cam3"]:
        if k in data:
            count += int(data[k].shape[0])
    return count


def _robot_z(data):
    if "robot_pose7" in data:
        pose7 = data["robot_pose7"]
        if pose7.shape == (7,):
            return float(pose7[2])
    if "robot_pos" in data:
        pos = data["robot_pos"]
        if pos.shape == (3,):
            return float(pos[2])
    return None


def analyze(data_dir, z_threshold=0.35, min_points=1000, max_envs=None, max_frames_per_env=None, stride=1):
    env_dirs = _find_env_dirs(data_dir)
    if max_envs is not None:
        env_dirs = env_dirs[:max_envs]

    total_frames = 0
    total_points = 0
    total_low_points = 0
    total_empty = 0
    total_low_z = 0
    total_with_z = 0
    total_trajs = 0
    traj_lengths_all = []

    for env_dir in env_dirs:
        traj_dirs = _find_traj_dirs(env_dir)
        if not traj_dirs:
            print(f"[skip] no trajs/frames in {env_dir}")
            continue

        env_frames = 0
        env_points = 0
        env_low_points = 0
        env_empty = 0
        env_low_z = 0
        env_with_z = 0
        env_traj_lengths = []

        for traj_dir in traj_dirs:
            frames = sorted(glob.glob(os.path.join(traj_dir, "frame_*.npz")))
            if not frames:
                continue
            env_traj_lengths.append(len(frames))
            total_trajs += 1

            for idx, frame_path in enumerate(frames):
                if stride > 1 and (idx % stride) != 0:
                    continue
                if max_frames_per_env is not None and env_frames >= max_frames_per_env:
                    break
                with np.load(frame_path) as data:
                    npts = _cam_points_count(data)
                    env_points += npts
                    if npts == 0:
                        env_empty += 1
                    if npts < min_points:
                        env_low_points += 1
                    z = _robot_z(data)
                    if z is not None:
                        env_with_z += 1
                        if z < z_threshold:
                            env_low_z += 1
                env_frames += 1

            if max_frames_per_env is not None and env_frames >= max_frames_per_env:
                break

        if env_frames == 0:
            print(f"[skip] no frames in {env_dir}")
            continue

        env_mean_pts = env_points / max(env_frames, 1)
        env_low_pts_ratio = env_low_points / max(env_frames, 1)
        env_empty_ratio = env_empty / max(env_frames, 1)
        env_low_z_ratio = env_low_z / max(env_with_z, 1) if env_with_z > 0 else 0.0

        traj_lengths_all.extend(env_traj_lengths)

        total_frames += env_frames
        total_points += env_points
        total_low_points += env_low_points
        total_empty += env_empty
        total_low_z += env_low_z
        total_with_z += env_with_z

        if env_traj_lengths:
            tl = np.array(env_traj_lengths, dtype=np.int64)
            print(
                f"[env] {os.path.basename(env_dir)} "
                f"frames={env_frames} trajs={len(env_traj_lengths)} "
                f"traj_len(mean/min/max)={tl.mean():.1f}/{tl.min()}/{tl.max()} "
                f"cam_pts_mean={env_mean_pts:.1f} "
                f"low_pts<{min_points}={env_low_pts_ratio:.2%} empty={env_empty_ratio:.2%} "
                f"low_z<{z_threshold}={env_low_z_ratio:.2%}"
            )
        else:
            print(
                f"[env] {os.path.basename(env_dir)} frames={env_frames} "
                f"cam_pts_mean={env_mean_pts:.1f} "
                f"low_pts<{min_points}={env_low_pts_ratio:.2%} empty={env_empty_ratio:.2%} "
                f"low_z<{z_threshold}={env_low_z_ratio:.2%}"
            )

    if total_frames == 0:
        print("[summary] no frames processed")
        return

    overall_mean_pts = total_points / max(total_frames, 1)
    overall_low_pts_ratio = total_low_points / max(total_frames, 1)
    overall_empty_ratio = total_empty / max(total_frames, 1)
    overall_low_z_ratio = total_low_z / max(total_with_z, 1) if total_with_z > 0 else 0.0

    if traj_lengths_all:
        tl = np.array(traj_lengths_all, dtype=np.int64)
        traj_stats = f"{tl.mean():.1f}/{tl.min()}/{tl.max()}"
    else:
        traj_stats = "n/a"

    print(
        f"[summary] frames={total_frames} trajs={total_trajs} "
        f"traj_len(mean/min/max)={traj_stats} "
        f"cam_pts_mean={overall_mean_pts:.1f} "
        f"low_pts<{min_points}={overall_low_pts_ratio:.2%} empty={overall_empty_ratio:.2%} "
        f"low_z<{z_threshold}={overall_low_z_ratio:.2%}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root or env_* directory")
    parser.add_argument("--z_threshold", type=float, default=0.35, help="Robot z threshold for low height")
    parser.add_argument("--min_points", type=int, default=1000, help="Min cam_points for quality")
    parser.add_argument("--max_envs", type=int, default=None, help="Limit number of envs")
    parser.add_argument("--max_frames_per_env", type=int, default=None, help="Limit frames per env")
    parser.add_argument("--stride", type=int, default=1, help="Sample every N frames")
    args = parser.parse_args()

    analyze(
        data_dir=args.data_dir,
        z_threshold=args.z_threshold,
        min_points=args.min_points,
        max_envs=args.max_envs,
        max_frames_per_env=args.max_frames_per_env,
        stride=max(1, args.stride),
    )


if __name__ == "__main__":
    main()
