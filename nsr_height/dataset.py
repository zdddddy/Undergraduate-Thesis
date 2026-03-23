import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


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


def _world_to_local(points, pose7, gravity_aligned=True):
    if points.size == 0 or pose7 is None:
        return points
    pos = pose7[:3].astype(np.float32)
    quat = pose7[3:7].astype(np.float32)
    rot = _quat_to_yaw_rotmat_xyzw(quat) if gravity_aligned else _quat_to_rotmat_xyzw(quat)
    return (rot.T @ (points - pos).T).T.astype(np.float32)


class HeightMapDataset(Dataset):
    def __init__(
        self,
        data_root,
        min_seq_len=12,
        sequence_length=12,
        split='train',
        map_size=3.2,
        resolution=0.05,
        fill_value=0.0,
        use_local_frame=True,
        gravity_aligned=True,
        augment_dropout=0.1,
        augment_jitter_xy=0.03,
        augment_jitter_z=0.02,
        augment_tilt_deg=2.0,
    ):
        self.data_root = data_root
        self.min_seq_len = int(min_seq_len)
        self.sequence_length = int(sequence_length)
        self.split = split
        self.map_size = float(map_size)
        self.resolution = float(resolution)
        self.fill_value = float(fill_value)
        self.use_local_frame = bool(use_local_frame)
        self.gravity_aligned = bool(gravity_aligned)
        self.augment_dropout = float(augment_dropout)
        self.augment_jitter_xy = float(augment_jitter_xy)
        self.augment_jitter_z = float(augment_jitter_z)
        self.augment_tilt_deg = float(augment_tilt_deg)

        self.grid_size = int(round(self.map_size / self.resolution))
        if self.grid_size <= 0:
            raise ValueError('grid_size must be positive')
        self.half_extent = 0.5 * self.map_size

        self.samples = []
        split_root = self.data_root
        explicit_split_root = os.path.join(self.data_root, self.split)
        if os.path.isdir(explicit_split_root):
            split_root = explicit_split_root

        traj_dirs = sorted(glob.glob(os.path.join(split_root, 'traj_*')))
        if not traj_dirs:
            traj_dirs = sorted(glob.glob(os.path.join(split_root, 'env_*', 'traj_*')))
        for traj_dir in traj_dirs:
            if not os.path.isdir(traj_dir):
                continue
            npz_files = sorted(glob.glob(os.path.join(traj_dir, '*.npz')))
            if len(npz_files) < self.min_seq_len:
                continue
            num_valid_starts = len(npz_files) - self.sequence_length + 1
            if num_valid_starts <= 0:
                continue
            for i in range(num_valid_starts):
                self.samples.append((npz_files, i))

    def __len__(self):
        return len(self.samples)

    def augment_points(self, points):
        if points.size == 0:
            return points

        if self.augment_tilt_deg > 0.0:
            tilt_rad = np.deg2rad(self.augment_tilt_deg)
            roll = np.random.uniform(-tilt_rad, tilt_rad)
            pitch = np.random.uniform(-tilt_rad, tilt_rad)
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            rot_x = np.array(
                [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
                dtype=np.float32,
            )
            rot_y = np.array(
                [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
                dtype=np.float32,
            )
            points = (points @ (rot_y @ rot_x).T).astype(np.float32)

        jitter = np.zeros_like(points, dtype=np.float32)
        jitter[:, :2] = np.random.uniform(
            -self.augment_jitter_xy, self.augment_jitter_xy, size=(points.shape[0], 2)
        ).astype(np.float32)
        jitter[:, 2] = np.random.uniform(
            -self.augment_jitter_z, self.augment_jitter_z, size=(points.shape[0],)
        ).astype(np.float32)
        points = points + jitter

        if points.shape[0] > 1 and self.augment_dropout > 0.0:
            keep = np.random.rand(points.shape[0]) > self.augment_dropout
            if not np.any(keep):
                keep[np.random.randint(0, points.shape[0])] = True
            points = points[keep]
        return points

    def points_to_heightmap(self, points):
        hmap = np.full((self.grid_size, self.grid_size), -np.inf, dtype=np.float32)
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if points.size == 0:
            hmap.fill(self.fill_value)
            return hmap[None], mask[None]

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        ix = np.floor((x + self.half_extent) / self.resolution).astype(np.int32)
        iy = np.floor((y + self.half_extent) / self.resolution).astype(np.int32)

        valid = (
            (ix >= 0)
            & (ix < self.grid_size)
            & (iy >= 0)
            & (iy < self.grid_size)
            & np.isfinite(z)
        )
        if not np.any(valid):
            hmap.fill(self.fill_value)
            return hmap[None], mask[None]

        ix = ix[valid]
        iy = iy[valid]
        z = z[valid]
        linear = ix * self.grid_size + iy

        flat_hmap = hmap.reshape(-1)
        np.maximum.at(flat_hmap, linear, z)
        flat_mask = mask.reshape(-1)
        flat_mask[linear] = 1.0

        hmap[mask == 0] = self.fill_value
        return hmap[None], mask[None]

    def _resize_2d(self, arr, mode):
        t = torch.from_numpy(arr.astype(np.float32))[None, None]
        if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            out = F.interpolate(
                t,
                size=(self.grid_size, self.grid_size),
                mode=mode,
                align_corners=False,
            )
        else:
            out = F.interpolate(
                t,
                size=(self.grid_size, self.grid_size),
                mode=mode,
            )
        return out[0, 0].cpu().numpy().astype(np.float32)

    def _load_direct_gt_heightmap(self, data):
        if 'gt_heightmap' not in data:
            return None

        gt_h = data['gt_heightmap'].astype(np.float32)
        if gt_h.ndim == 3 and gt_h.shape[0] == 1:
            gt_h = gt_h[0]
        if gt_h.ndim != 2:
            raise ValueError(f'Invalid gt_heightmap shape: {gt_h.shape}')

        if 'gt_mask' in data:
            gt_m = data['gt_mask'].astype(np.float32)
            if gt_m.ndim == 3 and gt_m.shape[0] == 1:
                gt_m = gt_m[0]
        else:
            gt_m = np.isfinite(gt_h).astype(np.float32)
        if gt_m.ndim != 2:
            raise ValueError(f'Invalid gt_mask shape: {gt_m.shape}')

        if 'gt_local_frame' in data:
            gt_is_local = bool(int(np.asarray(data['gt_local_frame']).reshape(-1)[0]))
            if gt_is_local != self.use_local_frame:
                frame_name = 'local' if gt_is_local else 'world'
                expected_name = 'local' if self.use_local_frame else 'world'
                raise ValueError(
                    f'GT heightmap frame mismatch: data={frame_name}, dataset expects {expected_name}.'
                )
            if gt_is_local and 'gt_gravity_aligned' in data:
                gt_gravity_aligned = bool(int(np.asarray(data['gt_gravity_aligned']).reshape(-1)[0]))
                if gt_gravity_aligned != self.gravity_aligned:
                    frame_name = 'gravity-aligned' if gt_gravity_aligned else 'full-6dof-local'
                    expected_name = 'gravity-aligned' if self.gravity_aligned else 'full-6dof-local'
                    raise ValueError(
                        f'GT local-frame convention mismatch: data={frame_name}, dataset expects {expected_name}.'
                    )

        if gt_h.shape != (self.grid_size, self.grid_size):
            gt_h = self._resize_2d(gt_h, mode='bilinear')
            gt_m = self._resize_2d(gt_m, mode='nearest')

        gt_m = (gt_m > 0.5).astype(np.float32)
        gt_h = gt_h.astype(np.float32)
        gt_h[gt_m <= 0.5] = self.fill_value
        return gt_h[None], gt_m[None]

    def __getitem__(self, idx):
        file_list, start_idx = self.samples[idx]

        seq_in_height = []
        seq_in_mask = []
        seq_gt_height = []
        seq_gt_mask = []
        seq_pose7 = []

        for t in range(self.sequence_length):
            path = file_list[start_idx + t]
            with np.load(path) as data:
                measurement = data['cam_points'].astype(np.float32)
                pose7 = data['robot_pose7'].astype(np.float32) if 'robot_pose7' in data else None
                direct_gt = self._load_direct_gt_heightmap(data)
                if direct_gt is None:
                    ground_truth = data['terrain'].astype(np.float32)
                else:
                    ground_truth = None
            if pose7 is None or pose7.shape != (7,):
                pose7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            if self.use_local_frame:
                measurement = _world_to_local(
                    measurement,
                    pose7,
                    gravity_aligned=self.gravity_aligned,
                )
                if ground_truth is not None:
                    ground_truth = _world_to_local(
                        ground_truth,
                        pose7,
                        gravity_aligned=self.gravity_aligned,
                    )

            if self.split == 'train':
                measurement = self.augment_points(measurement)

            in_height, in_mask = self.points_to_heightmap(measurement)
            if direct_gt is None:
                gt_height, gt_mask = self.points_to_heightmap(ground_truth)
            else:
                gt_height, gt_mask = direct_gt

            seq_in_height.append(in_height)
            seq_in_mask.append(in_mask)
            seq_gt_height.append(gt_height)
            seq_gt_mask.append(gt_mask)
            seq_pose7.append(pose7.astype(np.float32))

        return {
            'seq_in_height': seq_in_height,
            'seq_in_mask': seq_in_mask,
            'seq_gt_height': seq_gt_height,
            'seq_gt_mask': seq_gt_mask,
            'seq_pose7': seq_pose7,
        }


def collate_fn(batch):
    seq_in_height = np.stack([np.stack(item['seq_in_height'], axis=0) for item in batch], axis=0)
    seq_in_mask = np.stack([np.stack(item['seq_in_mask'], axis=0) for item in batch], axis=0)
    seq_gt_height = np.stack([np.stack(item['seq_gt_height'], axis=0) for item in batch], axis=0)
    seq_gt_mask = np.stack([np.stack(item['seq_gt_mask'], axis=0) for item in batch], axis=0)
    seq_pose7 = np.stack([np.stack(item['seq_pose7'], axis=0) for item in batch], axis=0)

    in_height = torch.from_numpy(seq_in_height).float()
    in_mask = torch.from_numpy(seq_in_mask).float()
    gt_height = torch.from_numpy(seq_gt_height).float()
    gt_mask = torch.from_numpy(seq_gt_mask).float()
    pose7 = torch.from_numpy(seq_pose7).float()

    return {
        'in_height': in_height,
        'in_mask': in_mask,
        'gt_height': gt_height,
        'gt_mask': gt_mask,
        'pose7': pose7,
        'metadata': {
            'B': int(in_height.shape[0]),
            'T': int(in_height.shape[1]),
            'H': int(in_height.shape[3]),
            'W': int(in_height.shape[4]),
        },
    }


# Backward-compatible alias.
PointCloudDataset = HeightMapDataset
