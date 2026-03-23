import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import MinkowskiEngine as ME


class PointCloudDataset(Dataset):
    def __init__(self, data_root, voxel_size=0.05, min_seq_len=12, sequence_length=12, split='train'):
        self.data_root = data_root
        self.voxel_size = float(voxel_size)
        self.min_seq_len = int(min_seq_len)
        self.sequence_length = int(sequence_length)
        self.split = split
        self.samples = []

        traj_dirs = sorted(glob.glob(os.path.join(self.data_root, 'traj_*')))
        if not traj_dirs:
            traj_dirs = sorted(glob.glob(os.path.join(self.data_root, 'env_*', 'traj_*')))
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

    def augment(self, points):
        if points.size == 0:
            return points

        # Uniform noise (position jitter)
        jitter = np.random.uniform(-0.05, 0.05, size=points.shape).astype(np.float32)
        points = points + jitter

        # Random dropout
        if points.shape[0] > 1:
            mask = np.random.rand(points.shape[0]) > 0.1
            if not np.any(mask):
                mask[np.random.randint(0, points.shape[0])] = True
            points = points[mask]

        return points

    def sparse_quantize(self, points):
        if points.size == 0:
            empty_coords = np.zeros((0, 3), dtype=np.int32)
            empty_feats = np.zeros((0, 3), dtype=np.float32)
            return empty_coords, empty_feats

        discrete_coords = np.floor(points / self.voxel_size).astype(np.int32)
        coords, inverse = np.unique(discrete_coords, axis=0, return_inverse=True)

        sums = np.zeros((coords.shape[0], 3), dtype=np.float32)
        np.add.at(sums, inverse, points.astype(np.float32))
        counts = np.bincount(inverse).astype(np.float32)
        centroids = sums / counts[:, None]
        voxel_centers = (coords.astype(np.float32) + 0.5) * self.voxel_size
        feats = centroids - voxel_centers

        return coords, feats

    def __getitem__(self, idx):
        file_list, start_idx = self.samples[idx]

        seq_coords = []
        seq_feats = []
        seq_gt_coords = []
        seq_gt_offsets = []

        for t in range(self.sequence_length):
            path = file_list[start_idx + t]
            with np.load(path) as data:
                measurement = data['cam_points'].astype(np.float32)
                ground_truth = data['terrain'].astype(np.float32)

            if self.split == 'train':
                measurement = self.augment(measurement)

            in_coords, in_feats = self.sparse_quantize(measurement)
            gt_coords, gt_offsets = self.sparse_quantize(ground_truth)

            seq_coords.append(in_coords)
            seq_feats.append(in_feats)
            seq_gt_coords.append(gt_coords)
            seq_gt_offsets.append(gt_offsets)

        return {
            'seq_coords': seq_coords,
            'seq_feats': seq_feats,
            'seq_gt_coords': seq_gt_coords,
            'seq_gt_offsets': seq_gt_offsets,
        }


def collate_fn(batch):
    B = len(batch)
    T = len(batch[0]['seq_coords']) if B > 0 else 0

    seq = []

    for t in range(T):
        coords_list = []
        feats_list = []
        gt_coords_list = []
        gt_offsets_list = []

        for b in range(B):
            coords_list.append(torch.from_numpy(batch[b]['seq_coords'][t]).int())
            feats_list.append(torch.from_numpy(batch[b]['seq_feats'][t]).float())
            gt_coords_list.append(torch.from_numpy(batch[b]['seq_gt_coords'][t]).int())
            gt_offsets_list.append(torch.from_numpy(batch[b]['seq_gt_offsets'][t]).float())

        coords_batch = ME.utils.batched_coordinates(coords_list)
        gt_coords_batch = ME.utils.batched_coordinates(gt_coords_list)
        feats_batch = torch.cat(feats_list, dim=0) if feats_list else torch.empty((0, 3), dtype=torch.float32)
        gt_offsets_batch = (
            torch.cat(gt_offsets_list, dim=0) if gt_offsets_list else torch.empty((0, 3), dtype=torch.float32)
        )

        seq.append(
            {
                'coords': coords_batch,
                'feats': feats_batch,
                'gt_coords': gt_coords_batch,
                'gt_offsets': gt_offsets_batch,
            }
        )

    return {
        'seq': seq,
        'metadata': {'B': B, 'T': T},
    }
