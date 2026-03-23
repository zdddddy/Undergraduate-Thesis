import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import HeightMapDataset, collate_fn
from model import HeightRecurrentUNet

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def set_batchnorm_use_batch_stats(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None


def quat_to_rotmat_xyzw_torch(q):
    n = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)
    qn = q / n
    x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    row0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=1)
    row1 = torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], dim=1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], dim=1)
    return torch.stack([row0, row1, row2], dim=1)


def quat_to_yaw_rotmat_xyzw_torch(q):
    n = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)
    qn = q / n
    x, y, z, w = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    yaw = torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    zeros = torch.zeros_like(cy)
    ones = torch.ones_like(cy)
    row0 = torch.stack([cy, -sy, zeros], dim=1)
    row1 = torch.stack([sy, cy, zeros], dim=1)
    row2 = torch.stack([zeros, zeros, ones], dim=1)
    return torch.stack([row0, row1, row2], dim=1)


def warp_prev_to_current(prev_map, prev_pose7, cur_pose7, map_size, prev_valid=None, gravity_aligned=True):
    bsz, _, h, w = prev_map.shape
    if prev_valid is None:
        prev_valid = torch.ones_like(prev_map)
    if h < 2 or w < 2:
        return prev_map, prev_valid

    device = prev_map.device
    dtype = prev_map.dtype
    half = 0.5 * float(map_size)
    res_x = float(map_size) / float(h)
    res_y = float(map_size) / float(w)

    x_centers = (torch.arange(h, device=device, dtype=dtype) + 0.5) * res_x - half
    y_centers = (torch.arange(w, device=device, dtype=dtype) + 0.5) * res_y - half
    x_grid = x_centers[:, None].expand(h, w)
    y_grid = y_centers[None, :].expand(h, w)
    # In gravity-aligned mode this is exact for XY+Yaw transforms.
    # In full 6DoF mode this remains a 2.5D approximation.
    p_cur = torch.stack([x_grid, y_grid, torch.zeros_like(x_grid)], dim=-1)
    p_cur = p_cur.unsqueeze(0).expand(bsz, h, w, 3)

    pos_prev = prev_pose7[:, :3].to(dtype=dtype)
    pos_cur = cur_pose7[:, :3].to(dtype=dtype)
    if gravity_aligned:
        rot_prev = quat_to_yaw_rotmat_xyzw_torch(prev_pose7[:, 3:7].to(dtype=dtype))
        rot_cur = quat_to_yaw_rotmat_xyzw_torch(cur_pose7[:, 3:7].to(dtype=dtype))
    else:
        rot_prev = quat_to_rotmat_xyzw_torch(prev_pose7[:, 3:7].to(dtype=dtype))
        rot_cur = quat_to_rotmat_xyzw_torch(cur_pose7[:, 3:7].to(dtype=dtype))

    p_world = torch.einsum('bij,bhwj->bhwi', rot_cur, p_cur) + pos_cur[:, None, None, :]
    p_prev = torch.einsum('bij,bhwj->bhwi', rot_prev.transpose(1, 2), p_world - pos_prev[:, None, None, :])

    x_prev = p_prev[..., 0]
    y_prev = p_prev[..., 1]
    i_prev = (x_prev + half) / res_x - 0.5
    j_prev = (y_prev + half) / res_y - 0.5

    grid_x = 2.0 * (j_prev / max(w - 1, 1)) - 1.0
    grid_y = 2.0 * (i_prev / max(h - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    warped_h_prev_raw = F.grid_sample(
        prev_map,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    warped_valid = F.grid_sample(
        prev_valid,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    ).clamp_(0.0, 1.0)
    # Avoid numeric blow-up near FOV boundaries where warped_valid is tiny.
    valid_eps = 5e-2
    warped_h_prev = torch.where(
        warped_valid > valid_eps,
        warped_h_prev_raw / warped_valid.clamp_min(valid_eps),
        torch.zeros_like(warped_h_prev_raw),
    )

    p_prev_obj = torch.stack([x_prev, y_prev, warped_h_prev[:, 0]], dim=-1)
    p_world_obj = torch.einsum('bij,bhwj->bhwi', rot_prev, p_prev_obj) + pos_prev[:, None, None, :]
    p_cur_obj = torch.einsum(
        'bij,bhwj->bhwi',
        rot_cur.transpose(1, 2),
        p_world_obj - pos_cur[:, None, None, :],
    )
    warped_h_cur = p_cur_obj[..., 2].unsqueeze(1)
    warped_h_cur = torch.where(
        warped_valid > valid_eps,
        warped_h_cur,
        torch.zeros_like(warped_h_cur),
    )
    return warped_h_cur, warped_valid


def build_model_input(meas_h, meas_m, prev_h, prev_valid, in_channels):
    if in_channels == 4:
        return torch.cat([meas_h, meas_m, prev_h, prev_valid], dim=1)
    if in_channels == 3:
        return torch.cat([meas_h, meas_m, prev_h], dim=1)
    raise ValueError(f'Unsupported in_channels={in_channels}; expected 3 or 4.')


def unpack_model_outputs(model_out):
    if isinstance(model_out, (tuple, list)):
        h = model_out[0]
        edge_logits = model_out[1] if len(model_out) > 1 else None
        return h, edge_logits
    if isinstance(model_out, dict):
        return model_out.get('height', None), model_out.get('edge_logits', None)
    return model_out, None


def _binarize_valid(mask, threshold):
    if threshold <= 0.0:
        return mask.clamp(0.0, 1.0)
    if threshold >= 1.0:
        return (mask >= 1.0).float()
    return (mask > float(threshold)).float()


def _masked(arr, mask):
    out = arr.copy()
    out[mask <= 0.5] = np.nan
    return out


def _fill_missing_neighbor_mean(arr, mask, max_iters=120):
    filled = arr.astype(np.float32).copy()
    valid = (mask > 0.5) & np.isfinite(filled)
    if not np.any(valid):
        return np.zeros_like(filled, dtype=np.float32)

    mean_val = float(filled[valid].mean())
    filled[~valid] = mean_val
    known = valid.copy()

    for _ in range(max_iters):
        if np.all(known):
            break

        up_v = np.roll(known, 1, axis=0)
        up_v[0, :] = False
        down_v = np.roll(known, -1, axis=0)
        down_v[-1, :] = False
        left_v = np.roll(known, 1, axis=1)
        left_v[:, 0] = False
        right_v = np.roll(known, -1, axis=1)
        right_v[:, -1] = False

        up = np.roll(filled, 1, axis=0)
        down = np.roll(filled, -1, axis=0)
        left = np.roll(filled, 1, axis=1)
        right = np.roll(filled, -1, axis=1)

        denom = (
            up_v.astype(np.float32)
            + down_v.astype(np.float32)
            + left_v.astype(np.float32)
            + right_v.astype(np.float32)
        )
        numer = (
            up * up_v.astype(np.float32)
            + down * down_v.astype(np.float32)
            + left * left_v.astype(np.float32)
            + right * right_v.astype(np.float32)
        )

        update = (~known) & (denom > 0.0)
        if not np.any(update):
            break

        filled[update] = numer[update] / denom[update]
        known[update] = True

    filled[~known] = mean_val
    return filled


def _prepare_for_display(arr, mask, fill_missing=False, fill_iters=120):
    if fill_missing:
        return _fill_missing_neighbor_mean(arr, mask, max_iters=fill_iters)
    return _masked(arr, mask)


def _safe_percentile(values, q, default):
    if values.size == 0:
        return default
    return float(np.percentile(values, q))


def save_frame_png(
    out_path,
    meas_h,
    meas_m,
    pred_h,
    gt_h,
    gt_m,
    map_size,
    title_prefix,
    interp='bilinear',
    fill_missing=True,
    fill_iters=120,
    height_vmin=None,
    height_vmax=None,
    error_vmax=None,
):
    import matplotlib.pyplot as plt

    abs_err = np.abs(pred_h - gt_h)

    valid_h = np.concatenate(
        [
            meas_h[meas_m > 0.5],
            pred_h[gt_m > 0.5],
            gt_h[gt_m > 0.5],
        ]
    )
    if valid_h.size > 0:
        vmin = _safe_percentile(valid_h, 2, -0.2)
        vmax = _safe_percentile(valid_h, 98, 0.2)
        if vmax <= vmin:
            vmax = vmin + 1e-3
    else:
        vmin, vmax = -0.2, 0.2

    if height_vmin is not None:
        vmin = float(height_vmin)
    if height_vmax is not None:
        vmax = float(height_vmax)
    if vmax <= vmin:
        vmax = vmin + 1e-3

    valid_err = abs_err[gt_m > 0.5]
    err_vmax = _safe_percentile(valid_err, 98, 0.2)
    err_vmax = max(err_vmax, 1e-6)
    if error_vmax is not None:
        err_vmax = max(float(error_vmax), 1e-6)

    # Keep measurement visualization raw: do not display-time fill missing cells.
    meas_disp = _prepare_for_display(meas_h, meas_m, fill_missing=False, fill_iters=fill_iters)
    pred_disp = _prepare_for_display(pred_h, gt_m, fill_missing=fill_missing, fill_iters=fill_iters)
    gt_disp = _prepare_for_display(gt_h, gt_m, fill_missing=fill_missing, fill_iters=fill_iters)
    abs_err_disp = _prepare_for_display(abs_err, gt_m, fill_missing=fill_missing, fill_iters=fill_iters)

    extent = [-0.5 * map_size, 0.5 * map_size, -0.5 * map_size, 0.5 * map_size]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), constrained_layout=True)

    im0 = axes[0, 0].imshow(
        meas_disp,
        origin='lower',
        extent=extent,
        interpolation=interp,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 0].set_title('Measurement')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(
        pred_disp,
        origin='lower',
        extent=extent,
        interpolation=interp,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title('Prediction')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(
        gt_disp,
        origin='lower',
        extent=extent,
        interpolation=interp,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )
    axes[1, 0].set_title('Ground Truth')
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(
        abs_err_disp,
        origin='lower',
        extent=extent,
        interpolation=interp,
        cmap='magma',
        vmin=0.0,
        vmax=err_vmax,
    )
    axes[1, 1].set_title('Absolute Error (GT mask)')
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.reshape(-1):
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    fig.suptitle(title_prefix)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate_and_visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = HeightMapDataset(
        args.data_dir,
        split='val',
        min_seq_len=args.min_seq_len,
        sequence_length=args.seq_len,
        map_size=args.map_size,
        resolution=args.resolution,
        fill_value=args.fill_value,
        use_local_frame=not args.world_frame,
        gravity_aligned=args.gravity_aligned,
    )
    if len(ds) == 0:
        print('[vis] empty dataset')
        return

    num_sequences = int(args.num_sequences)
    if num_sequences <= 0:
        print('[vis] num_sequences must be positive')
        return

    start_seq = int(args.start_seq)
    if args.random_seq:
        max_start = max(0, len(ds) - num_sequences)
        if args.seed is not None and args.seed >= 0:
            rng = np.random.default_rng(args.seed)
        else:
            rng = np.random.default_rng()
        start_seq = int(rng.integers(0, max_start + 1))
        print(
            f"[vis] random start_seq={start_seq} "
            f"(num_sequences={num_sequences}, max_start={max_start}, seed={args.seed})"
        )

    if start_seq >= len(ds):
        print(f'[vis] start_seq={start_seq} out of range for dataset size={len(ds)}')
        return

    end_seq = min(start_seq + num_sequences, len(ds))
    target_indices = list(range(start_seq, end_seq))
    if not target_indices:
        print(f'[vis] empty target index range: start_seq={start_seq}, end_seq={end_seq}')
        return

    ds_view = Subset(ds, target_indices)
    loader = DataLoader(ds_view, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)

    model = HeightRecurrentUNet(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        out_channels=1,
        use_edge_head=args.use_edge_head,
        norm_type=args.norm_type,
        group_norm_groups=args.group_norm_groups,
    )
    try:
        load_result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint load failed for {args.ckpt}. "
            "Check --in_channels/--base_channels and checkpoint compatibility."
        ) from e

    missing = list(getattr(load_result, 'missing_keys', []))
    unexpected = list(getattr(load_result, 'unexpected_keys', []))
    if missing or unexpected:
        msg = (
            f"[vis] checkpoint key mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            msg += f" missing_example={missing[:3]}"
        if unexpected:
            msg += f" unexpected_example={unexpected[:3]}"
        if args.allow_partial_load:
            print(msg)
        else:
            raise RuntimeError(
                msg
                + ". Refusing partial load. Use matching checkpoint/model args, or pass "
                "--allow_partial_load for debugging only."
            )
    else:
        print('[vis] checkpoint loaded with full key match')

    if 'epoch' in ckpt:
        print(f"[vis] checkpoint epoch={ckpt['epoch']}")
    train_args = ckpt.get('train_args', None)
    if isinstance(train_args, dict):
        compare_keys = [
            'in_channels',
            'base_channels',
            'norm_type',
            'group_norm_groups',
            'use_edge_head',
            'map_size',
            'resolution',
            'fill_value',
            'gravity_aligned',
            'align_prev',
            'prev_valid_threshold',
        ]
        mismatches = []
        for k in compare_keys:
            if k in train_args and hasattr(args, k):
                cur_v = getattr(args, k)
                ckpt_v = train_args[k]
                if cur_v != ckpt_v:
                    mismatches.append((k, ckpt_v, cur_v))
        if mismatches:
            print('[vis] warning: args differ from checkpoint train_args:')
            for k, ckpt_v, cur_v in mismatches:
                print(f'  - {k}: ckpt={ckpt_v} current={cur_v}')
    model.to(device)
    if args.bn_use_batch_stats:
        set_batchnorm_use_batch_stats(model)
        print('[vis] BatchNorm set to batch-stat mode (running stats ignored)')
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    seq_saved = 0
    steps = 0

    with torch.no_grad():
        for local_idx, batch in enumerate(loader):
            seq_idx = target_indices[local_idx]
            if seq_saved >= num_sequences:
                break

            in_height = batch['in_height'].to(device)
            in_mask = batch['in_mask'].to(device)
            gt_height = batch['gt_height'].to(device)
            gt_mask = batch['gt_mask'].to(device)
            pose7 = batch['pose7'].to(device)

            seq_dir = os.path.join(args.out_dir, f'seq_{seq_idx:05d}')
            os.makedirs(seq_dir, exist_ok=True)

            prev_pred = None
            prev_pose = None
            prev_valid_state = None
            T = in_height.shape[1]
            seq_mae_sum = 0.0
            seq_gt_pix = 0.0
            seq_hole_mae_sum = 0.0
            seq_hole_pix = 0.0
            saved_pngs = []

            for t in range(T):
                meas_h = in_height[:, t]
                meas_m = in_mask[:, t]
                target_h = gt_height[:, t]
                target_m = gt_mask[:, t]
                cur_pose = pose7[:, t]

                if args.disable_prev or prev_pred is None:
                    prev_in = meas_h
                    prev_in_valid = meas_m
                else:
                    if args.align_prev:
                        prev_valid_for_warp = (
                            prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                        )
                        prev_in, prev_in_valid = warp_prev_to_current(
                            prev_pred,
                            prev_pose,
                            cur_pose,
                            args.map_size,
                            prev_valid=prev_valid_for_warp,
                            gravity_aligned=args.gravity_aligned,
                        )
                    else:
                        prev_in = prev_pred
                        prev_in_valid = (
                            prev_valid_state if prev_valid_state is not None else torch.ones_like(prev_pred)
                        )
                prev_in_valid = _binarize_valid(prev_in_valid, args.prev_valid_threshold)
                model_out = model(
                    build_model_input(
                        meas_h,
                        meas_m,
                        prev_in,
                        prev_in_valid,
                        args.in_channels,
                    )
                )
                pred_core, _pred_edge_logits = unpack_model_outputs(model_out)
                if args.residual_from_base:
                    prev_base = torch.where(
                        prev_in_valid > 0.5,
                        prev_in,
                        torch.full_like(prev_in, float(args.fill_value)),
                    )
                    base_h = torch.where(meas_m > 0.5, meas_h, prev_base)
                    residual = pred_core
                    if args.residual_tanh:
                        residual = args.residual_scale * torch.tanh(residual)
                    else:
                        residual = args.residual_scale * residual
                    pred_h = base_h + residual
                else:
                    pred_h = pred_core
                pred_h = torch.nan_to_num(pred_h, nan=float(args.fill_value), posinf=1e3, neginf=-1e3)
                hole = ((target_m > 0.5) & (meas_m <= 0.5)).float()
                if args.memory_meas_override:
                    prev_pred = torch.where(meas_m > 0.5, meas_h, pred_h)
                else:
                    prev_pred = pred_h
                prev_pose = cur_pose
                prev_valid_state = torch.maximum(meas_m, prev_in_valid)

                err = torch.abs(pred_h - target_h)
                seq_mae_sum += float((err * target_m).sum().item())
                seq_gt_pix += float(target_m.sum().item())

                seq_hole_mae_sum += float((err * hole).sum().item())
                seq_hole_pix += float(hole.sum().item())

                if t % args.frame_stride == 0:
                    meas_h_np = meas_h[0, 0].detach().cpu().numpy()
                    meas_m_np = meas_m[0, 0].detach().cpu().numpy()
                    pred_h_np = pred_h[0, 0].detach().cpu().numpy()
                    gt_h_np = target_h[0, 0].detach().cpu().numpy()
                    gt_m_np = target_m[0, 0].detach().cpu().numpy()

                    title = (
                        f'seq={seq_idx} frame={t} '
                        f'gt_mae={(np.abs(pred_h_np - gt_h_np)[gt_m_np > 0.5].mean() if (gt_m_np > 0.5).any() else 0.0):.4f}'
                    )
                    out_png = os.path.join(seq_dir, f'frame_{t:03d}.png')
                    save_frame_png(
                        out_png,
                        meas_h_np,
                        meas_m_np,
                        pred_h_np,
                        gt_h_np,
                        gt_m_np,
                        args.map_size,
                        title,
                        interp=args.interp,
                        fill_missing=args.vis_fill_missing,
                        fill_iters=args.fill_iters,
                        height_vmin=args.height_vmin,
                        height_vmax=args.height_vmax,
                        error_vmax=args.error_vmax,
                    )
                    saved_pngs.append(out_png)

                steps += 1
                if steps >= args.max_steps:
                    break

            seq_mae = seq_mae_sum / max(seq_gt_pix, 1.0)
            seq_hole_mae = seq_hole_mae_sum / max(seq_hole_pix, 1.0)
            with open(os.path.join(seq_dir, 'metrics.txt'), 'w') as f:
                f.write(f'seq_idx={seq_idx}\n')
                f.write(f'gt_mae={seq_mae:.6f}\n')
                f.write(f'hole_mae={seq_hole_mae:.6f}\n')
                f.write(f'gt_pixels={seq_gt_pix:.0f}\n')
                f.write(f'hole_pixels={seq_hole_pix:.0f}\n')

            gif_path = None
            if args.save_gif and saved_pngs:
                if imageio is None:
                    print('[vis] warning: imageio not available, skip gif export')
                else:
                    gif_path = os.path.join(seq_dir, f'seq_{seq_idx:05d}.gif')
                    # Pillow GIF backend uses millisecond frame duration.
                    frame_duration_ms = max(20, int(round(1000.0 / max(float(args.gif_fps), 1e-6))))
                    with imageio.get_writer(gif_path, mode='I', duration=frame_duration_ms, loop=0) as writer:
                        for png_path in saved_pngs:
                            writer.append_data(imageio.imread(png_path))

            if gif_path is None:
                print(f'[vis] saved {seq_dir} gt_mae={seq_mae:.4f} hole_mae={seq_hole_mae:.4f}')
            else:
                print(
                    f'[vis] saved {seq_dir} gt_mae={seq_mae:.4f} hole_mae={seq_hole_mae:.4f} '
                    f'gif={gif_path}'
                )
            seq_saved += 1

            if steps >= args.max_steps:
                break

    print(f'[vis] done: sequences={seq_saved}, steps={steps}, out_dir={args.out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='logs/nsr_height_vis')

    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--min_seq_len', type=int, default=12)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--use_edge_head', action='store_true', default=False, help='Enable auxiliary edge-prediction head')
    parser.add_argument('--norm_type', type=str, default='group', choices=['batch', 'group', 'instance'])
    parser.add_argument('--group_norm_groups', type=int, default=8)

    parser.add_argument('--map_size', type=float, default=3.2)
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--fill_value', type=float, default=0.0)
    parser.add_argument('--world_frame', action='store_true', default=False)
    parser.add_argument('--gravity_aligned', action='store_true', help='Use yaw-only local frame for mapping/warp')
    parser.add_argument('--full_6dof_frame', action='store_false', dest='gravity_aligned', help='Use full quaternion local frame (legacy)')
    parser.add_argument('--align_prev', action='store_true', help='Warp previous prediction to current frame')
    parser.add_argument('--no_align_prev', action='store_false', dest='align_prev')
    parser.add_argument('--disable_prev', action='store_true', default=False, help='Do not use previous prediction channel')
    parser.add_argument('--prev_valid_threshold', type=float, default=0.5, help='Binarization threshold for warped previous-valid mask before model input')
    parser.add_argument('--memory_meas_override', action='store_true', default=True, help='Update recurrent prev map with measured heights on observed cells')
    parser.add_argument('--no_memory_meas_override', action='store_false', dest='memory_meas_override')
    parser.add_argument('--residual_from_base', action='store_true', default=False, help='Predict residual on top of base map (measured cells + warped prev)')
    parser.add_argument('--residual_scale', type=float, default=0.2, help='Scale for residual branch output')
    parser.add_argument('--residual_tanh', action='store_true', default=True, help='Use tanh-bounded residual output')
    parser.add_argument('--no_residual_tanh', action='store_false', dest='residual_tanh')
    parser.add_argument('--interp', type=str, default='bilinear', choices=['nearest', 'bilinear', 'bicubic'])
    parser.add_argument('--vis_fill_missing', action='store_true', help='Fill empty cells for smoother display')
    parser.add_argument('--no_vis_fill_missing', action='store_false', dest='vis_fill_missing')
    parser.add_argument('--fill_iters', type=int, default=120, help='Iterations for display-only filling')
    parser.add_argument('--height_vmin', type=float, default=None, help='Fixed vmin for Measurement/Prediction/GT colorbars')
    parser.add_argument('--height_vmax', type=float, default=None, help='Fixed vmax for Measurement/Prediction/GT colorbars')
    parser.add_argument('--error_vmax', type=float, default=None, help='Fixed vmax for Absolute Error colorbar (vmin fixed at 0)')
    parser.add_argument('--save_gif', action='store_true', default=False, help='Also export each sequence as GIF')
    parser.add_argument('--gif_fps', type=float, default=2.0, help='GIF playback FPS')

    parser.add_argument('--num_sequences', type=int, default=3)
    parser.add_argument('--start_seq', type=int, default=0)
    parser.add_argument('--random_seq', action='store_true', default=False, help='Randomly choose start_seq each run')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed for --random_seq; -1 means random each run')
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--allow_partial_load', action='store_true', default=False, help='Allow checkpoint partial load (debug only)')
    parser.add_argument('--bn_use_batch_stats', action='store_true', default=True, help='Use BN batch stats during inference (recommended)')
    parser.add_argument('--bn_use_running_stats', action='store_false', dest='bn_use_batch_stats', help='Use BN running stats during inference')

    parser.set_defaults(vis_fill_missing=True, align_prev=True, gravity_aligned=True, bn_use_batch_stats=True)
    args = parser.parse_args()
    evaluate_and_visualize(args)


if __name__ == '__main__':
    main()
