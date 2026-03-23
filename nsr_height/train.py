import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from dataset import HeightMapDataset, collate_fn
from model import HeightRecurrentUNet


def zero_output_head(model):
    head = getattr(model, 'head', None)
    if head is None or not isinstance(head, torch.nn.Conv2d):
        return False
    with torch.no_grad():
        head.weight.zero_()
        if head.bias is not None:
            head.bias.zero_()
    return True


def quat_to_rotmat_xyzw_torch(q):
    # q: [B, 4] in xyzw
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
    # prev_map: [B, 1, H, W], pose7: [B, 7]
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
    p_cur = torch.stack([x_grid, y_grid, torch.zeros_like(x_grid)], dim=-1)  # [H, W, 3]
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
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

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

    # Convert sampled previous local heights into current local frame heights.
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


def masked_smooth_l1(pred, target, mask):
    loss = F.smooth_l1_loss(pred, target, reduction='none')
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def weighted_masked_smooth_l1(pred, target, weight):
    loss = F.smooth_l1_loss(pred, target, reduction='none')
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def masked_l1(pred, target, mask):
    loss = torch.abs(pred - target)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def smoothness_loss(pred):
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
    return dx + dy


def hole_tv_loss(pred, hole_mask):
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    mx = hole_mask[:, :, :, 1:] * hole_mask[:, :, :, :-1]
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    my = hole_mask[:, :, 1:, :] * hole_mask[:, :, :-1, :]
    numer = (dx * mx).sum() + (dy * my).sum()
    denom = mx.sum() + my.sum()
    return numer / denom.clamp_min(1.0)


def masked_grad_l1(pred, target, mask):
    pdx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tdx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mdx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    pdy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tdy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mdy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    num = (torch.abs(pdx - tdx) * mdx).sum() + (torch.abs(pdy - tdy) * mdy).sum()
    den = mdx.sum() + mdy.sum()
    return num / den.clamp_min(1.0)


def _laplace_4n(x):
    left = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode='constant', value=0.0)
    right = F.pad(x[:, :, :, 1:], (0, 1, 0, 0), mode='constant', value=0.0)
    up = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode='constant', value=0.0)
    down = F.pad(x[:, :, 1:, :], (0, 0, 0, 1), mode='constant', value=0.0)
    return 4.0 * x - left - right - up - down


def masked_laplace_l1(pred, target, mask):
    lp = _laplace_4n(pred)
    lt = _laplace_4n(target)

    m_left = F.pad(mask[:, :, :, :-1], (1, 0, 0, 0), mode='constant', value=0.0)
    m_right = F.pad(mask[:, :, :, 1:], (0, 1, 0, 0), mode='constant', value=0.0)
    m_up = F.pad(mask[:, :, :-1, :], (0, 0, 1, 0), mode='constant', value=0.0)
    m_down = F.pad(mask[:, :, 1:, :], (0, 0, 0, 1), mode='constant', value=0.0)
    m = mask * m_left * m_right * m_up * m_down

    denom = m.sum().clamp_min(1.0)
    return (torch.abs(lp - lt) * m).sum() / denom


def hole_hard_l1(pred, target, hole_mask, hard_ratio=0.2):
    # Mean L1 over top-k hardest hole pixels per sample.
    bsz = pred.shape[0]
    ratio = float(max(1e-4, min(1.0, hard_ratio)))
    err = torch.abs(pred - target).reshape(bsz, -1)
    m = (hole_mask > 0.5).reshape(bsz, -1)
    vals = []
    for b in range(bsz):
        e = err[b][m[b]]
        if e.numel() == 0:
            continue
        k = max(1, int(e.numel() * ratio))
        v = torch.topk(e, k=k, largest=True, sorted=False).values.mean()
        vals.append(v)
    if not vals:
        return pred.new_zeros(())
    return torch.stack(vals).mean()


def build_edge_target(target_h, target_m, edge_thresh):
    gx = torch.abs(target_h[:, :, :, 1:] - target_h[:, :, :, :-1])
    gy = torch.abs(target_h[:, :, 1:, :] - target_h[:, :, :-1, :])
    mx = target_m[:, :, :, 1:] * target_m[:, :, :, :-1]
    my = target_m[:, :, 1:, :] * target_m[:, :, :-1, :]

    edge = torch.zeros_like(target_h)
    gx = gx * mx
    gy = gy * my
    edge[:, :, :, 1:] = torch.maximum(edge[:, :, :, 1:], gx)
    edge[:, :, :, :-1] = torch.maximum(edge[:, :, :, :-1], gx)
    edge[:, :, 1:, :] = torch.maximum(edge[:, :, 1:, :], gy)
    edge[:, :, :-1, :] = torch.maximum(edge[:, :, :-1, :], gy)
    edge = edge * target_m
    return (edge > float(edge_thresh)).float()


def masked_edge_bce_with_logits(logits, target_edge, valid_mask, base_pos_weight=1.0, auto_pos_weight=True):
    if logits is None:
        return target_edge.new_zeros(())

    mask = (valid_mask > 0.5).float()
    pos = (target_edge * mask).sum()
    neg = ((1.0 - target_edge) * mask).sum()

    if auto_pos_weight:
        pos_w = (neg / pos.clamp_min(1.0)).clamp_min(1.0).clamp_max(30.0)
    else:
        pos_w = torch.as_tensor(float(base_pos_weight), dtype=logits.dtype, device=logits.device)

    weight = mask * (1.0 + (pos_w - 1.0) * target_edge)
    bce = F.binary_cross_entropy_with_logits(logits, target_edge, reduction='none')
    return (bce * weight).sum() / weight.sum().clamp_min(1.0)


def masked_edge_f1(logits, target_edge, valid_mask, threshold=0.5):
    if logits is None:
        return target_edge.new_zeros(())
    pred = (torch.sigmoid(logits) > float(threshold)).float()
    mask = (valid_mask > 0.5).float()
    tp = (pred * target_edge * mask).sum()
    fp = (pred * (1.0 - target_edge) * mask).sum()
    fn = ((1.0 - pred) * target_edge * mask).sum()
    return (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)


def edge_weight_map(target_h, target_m, gain):
    if gain <= 0.0:
        return torch.ones_like(target_h)

    gx = torch.abs(target_h[:, :, :, 1:] - target_h[:, :, :, :-1])
    gy = torch.abs(target_h[:, :, 1:, :] - target_h[:, :, :-1, :])
    mx = target_m[:, :, :, 1:] * target_m[:, :, :, :-1]
    my = target_m[:, :, 1:, :] * target_m[:, :, :-1, :]

    edge = torch.zeros_like(target_h)
    gx = gx * mx
    gy = gy * my
    edge[:, :, :, 1:] += gx
    edge[:, :, :, :-1] += gx
    edge[:, :, 1:, :] += gy
    edge[:, :, :-1, :] += gy
    edge = edge * target_m

    edge_mean = edge.sum() / target_m.sum().clamp_min(1.0)
    edge_norm = edge / edge_mean.clamp_min(1e-6)
    edge_factor = 1.0 + gain * edge_norm
    return edge_factor.clamp_max(1.0 + 4.0 * gain)


def _translate_single(h, m, dx, dy, fill_value):
    _, _, H, W = h.shape
    out_h = torch.full_like(h, float(fill_value))
    out_m = torch.zeros_like(m)

    src_x0 = max(0, -dx)
    src_x1 = min(H, H - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(H, H + dx)

    src_y0 = max(0, -dy)
    src_y1 = min(W, W - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(W, W + dy)

    if src_x1 > src_x0 and src_y1 > src_y0:
        out_h[:, :, dst_x0:dst_x1, dst_y0:dst_y1] = h[:, :, src_x0:src_x1, src_y0:src_y1]
        out_m[:, :, dst_x0:dst_x1, dst_y0:dst_y1] = m[:, :, src_x0:src_x1, src_y0:src_y1]
    return out_h, out_m


def augment_measurement_visibility(meas_h, meas_m, fill_value, aug_prob, max_shift, morph_prob, morph_kernel):
    if aug_prob <= 0.0 and morph_prob <= 0.0:
        return meas_h, meas_m

    B = meas_h.shape[0]
    out_h = meas_h
    out_m = meas_m

    if aug_prob > 0.0 and max_shift > 0:
        translated_h = []
        translated_m = []
        for b in range(B):
            hb = out_h[b:b + 1]
            mb = out_m[b:b + 1]
            if float(torch.rand((), device=hb.device)) < aug_prob:
                dx = int(torch.randint(-max_shift, max_shift + 1, (1,), device=hb.device).item())
                dy = int(torch.randint(-max_shift, max_shift + 1, (1,), device=hb.device).item())
                hb, mb = _translate_single(hb, mb, dx, dy, fill_value)
            translated_h.append(hb)
            translated_m.append(mb)
        out_h = torch.cat(translated_h, dim=0)
        out_m = torch.cat(translated_m, dim=0)

    if morph_prob > 0.0 and morph_kernel > 1:
        k = int(max(1, morph_kernel))
        if k % 2 == 0:
            k += 1
        if float(torch.rand((), device=out_h.device)) < morph_prob:
            if float(torch.rand((), device=out_h.device)) < 0.5:
                # Dilation
                out_m = F.max_pool2d(out_m, kernel_size=k, stride=1, padding=k // 2)
            else:
                # Erosion
                out_m = 1.0 - F.max_pool2d(1.0 - out_m, kernel_size=k, stride=1, padding=k // 2)
            out_m = (out_m > 0.5).float()
            out_h = torch.where(out_m > 0.5, out_h, torch.full_like(out_h, float(fill_value)))

    return out_h, out_m


def augment_model_input_channels(
    meas_h,
    meas_m,
    prev_h,
    prev_valid,
    input_mask_dropout_prob=0.0,
    input_prev_valid_dropout_prob=0.0,
    input_invalid_jitter=0.0,
):
    meas_h_in = meas_h
    meas_m_in = meas_m
    prev_h_in = prev_h
    prev_valid_in = prev_valid

    if input_invalid_jitter > 0.0:
        amp = float(input_invalid_jitter)
        noise_meas = (torch.rand_like(meas_h_in) * 2.0 - 1.0) * amp
        noise_prev = (torch.rand_like(prev_h_in) * 2.0 - 1.0) * amp
        meas_h_in = torch.where(meas_m_in > 0.5, meas_h_in, meas_h_in + noise_meas)
        prev_h_in = torch.where(prev_valid_in > 0.5, prev_h_in, prev_h_in + noise_prev)

    if input_mask_dropout_prob > 0.0:
        keep = (torch.rand_like(meas_m_in) > float(input_mask_dropout_prob)).float()
        meas_m_in = meas_m_in * keep

    if input_prev_valid_dropout_prob > 0.0:
        keep = (torch.rand_like(prev_valid_in) > float(input_prev_valid_dropout_prob)).float()
        prev_valid_in = prev_valid_in * keep

    return meas_h_in, meas_m_in, prev_h_in, prev_valid_in


def _binarize_valid(mask, threshold):
    if threshold <= 0.0:
        return mask.clamp(0.0, 1.0)
    if threshold >= 1.0:
        return (mask >= 1.0).float()
    return (mask > float(threshold)).float()


def evaluate_val_epoch(model, val_loader, args, device):
    if val_loader is None:
        return None

    model.eval()
    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_cons = 0.0
    epoch_smooth = 0.0
    epoch_hole_smooth = 0.0
    epoch_grad = 0.0
    epoch_lap = 0.0
    epoch_hard = 0.0
    epoch_edge = 0.0
    epoch_edge_f1 = 0.0
    epoch_hole_mae = 0.0
    epoch_seen_mae = 0.0
    epoch_prev_cov_hole = 0.0
    epoch_iters = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if args.max_val_iters_per_epoch > 0 and i >= args.max_val_iters_per_epoch:
                break

            in_height = batch['in_height'].to(device)  # [B, T, 1, H, W]
            in_mask = batch['in_mask'].to(device)      # [B, T, 1, H, W]
            gt_height = batch['gt_height'].to(device)  # [B, T, 1, H, W]
            gt_mask = batch['gt_mask'].to(device)      # [B, T, 1, H, W]
            pose7 = batch['pose7'].to(device)          # [B, T, 7]

            total_loss = torch.zeros((), device=device)
            total_recon = 0.0
            total_cons = 0.0
            total_smooth = 0.0
            total_hole_smooth = 0.0
            total_grad = 0.0
            total_lap = 0.0
            total_hard = 0.0
            total_edge = 0.0
            total_edge_f1 = 0.0
            total_hole_mae = 0.0
            total_seen_mae = 0.0
            total_prev_cov_hole = 0.0
            total_prev_cov_hole_count = 0.0

            prev_pred = None
            prev_pose = None
            prev_valid_state = None
            T = in_height.shape[1]
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

                model_input = build_model_input(
                    meas_h,
                    meas_m,
                    prev_in,
                    prev_in_valid,
                    args.in_channels,
                )
                prev_base = torch.where(
                    prev_in_valid > 0.5,
                    prev_in,
                    torch.full_like(prev_in, float(args.fill_value)),
                )
                base_h = torch.where(meas_m > 0.5, meas_h, prev_base)

                model_out = model(model_input)
                pred_core, pred_edge_logits = unpack_model_outputs(model_out)
                if args.residual_from_base:
                    residual = pred_core
                    if args.residual_tanh:
                        residual = args.residual_scale * torch.tanh(residual)
                    else:
                        residual = args.residual_scale * residual
                    pred_h = base_h + residual
                else:
                    pred_h = pred_core
                pred_h = torch.nan_to_num(pred_h, nan=float(args.fill_value), posinf=1e3, neginf=-1e3)
                hole_m = ((target_m > 0.5) & (meas_m <= 0.5)).float()
                seen_m = ((target_m > 0.5) & (meas_m > 0.5)).float()

                recon_w = args.seen_weight * seen_m + args.hole_weight * hole_m
                recon_w = recon_w * edge_weight_map(target_h, target_m, args.edge_weight_gain)
                if args.hole_structure_gain > 0.0:
                    structure_mag = torch.abs(target_h - base_h) * hole_m
                    structure_mean = structure_mag.sum() / hole_m.sum().clamp_min(1.0)
                    structure_norm = structure_mag / structure_mean.clamp_min(1e-6)
                    structure_factor = 1.0 + args.hole_structure_gain * structure_norm
                    recon_w = recon_w * torch.where(hole_m > 0.5, structure_factor, torch.ones_like(structure_factor))
                loss_recon = weighted_masked_smooth_l1(pred_h, target_h, recon_w)
                if args.meas_consistency_weight > 0.0:
                    loss_cons = masked_l1(pred_h, meas_h, meas_m)
                else:
                    loss_cons = pred_h.new_zeros(())
                loss_smooth = smoothness_loss(pred_h)
                if args.hole_smooth_weight > 0.0:
                    loss_hole_smooth = hole_tv_loss(pred_h, hole_m)
                else:
                    loss_hole_smooth = pred_h.new_zeros(())
                if args.grad_match_weight > 0.0:
                    loss_grad = masked_grad_l1(pred_h, target_h, target_m)
                else:
                    loss_grad = pred_h.new_zeros(())
                if args.laplace_weight > 0.0:
                    loss_lap = masked_laplace_l1(pred_h, target_h, hole_m)
                else:
                    loss_lap = pred_h.new_zeros(())
                if args.hole_hard_weight > 0.0:
                    loss_hard = hole_hard_l1(pred_h, target_h, hole_m, hard_ratio=args.hole_hard_ratio)
                else:
                    loss_hard = pred_h.new_zeros(())
                if args.edge_head_weight > 0.0 and pred_edge_logits is not None:
                    target_edge = build_edge_target(target_h, target_m, edge_thresh=args.edge_target_thresh)
                    loss_edge = masked_edge_bce_with_logits(
                        pred_edge_logits,
                        target_edge,
                        target_m,
                        base_pos_weight=args.edge_pos_weight,
                        auto_pos_weight=args.edge_pos_weight_auto,
                    )
                    edge_f1 = masked_edge_f1(
                        pred_edge_logits,
                        target_edge,
                        target_m,
                        threshold=args.edge_eval_thresh,
                    )
                else:
                    loss_edge = pred_h.new_zeros(())
                    edge_f1 = pred_h.new_zeros(())
                hole_mae = masked_l1(pred_h, target_h, hole_m)
                seen_mae = masked_l1(pred_h, target_h, seen_m)

                if (not args.disable_prev) and (t > 0):
                    hole_denom = hole_m.sum()
                    if float(hole_denom.item()) > 0.0:
                        prev_cov_hole = (prev_in_valid * hole_m).sum() / hole_denom
                        total_prev_cov_hole += float(prev_cov_hole.item())
                        total_prev_cov_hole_count += 1.0

                loss = (
                    loss_recon
                    + args.meas_consistency_weight * loss_cons
                    + args.smooth_weight * loss_smooth
                    + args.hole_smooth_weight * loss_hole_smooth
                    + args.grad_match_weight * loss_grad
                    + args.laplace_weight * loss_lap
                    + args.hole_hard_weight * loss_hard
                    + args.edge_head_weight * loss_edge
                )

                total_loss = total_loss + loss
                total_recon += float(loss_recon.item())
                total_cons += float(loss_cons.item())
                total_smooth += float(loss_smooth.item())
                total_hole_smooth += float(loss_hole_smooth.item())
                total_grad += float(loss_grad.item())
                total_lap += float(loss_lap.item())
                total_hard += float(loss_hard.item())
                total_edge += float(loss_edge.item())
                total_edge_f1 += float(edge_f1.item())
                total_hole_mae += float(hole_mae.item())
                total_seen_mae += float(seen_mae.item())
                if args.memory_meas_override:
                    prev_pred = torch.where(meas_m > 0.5, meas_h, pred_h)
                else:
                    prev_pred = pred_h
                prev_pose = cur_pose
                prev_valid_state = torch.maximum(meas_m, prev_in_valid)

            total_loss = total_loss / max(1, T)
            if not torch.isfinite(total_loss):
                continue

            epoch_loss += float(total_loss.item())
            epoch_recon += total_recon / max(1, T)
            epoch_cons += total_cons / max(1, T)
            epoch_smooth += total_smooth / max(1, T)
            epoch_hole_smooth += total_hole_smooth / max(1, T)
            epoch_grad += total_grad / max(1, T)
            epoch_lap += total_lap / max(1, T)
            epoch_hard += total_hard / max(1, T)
            epoch_edge += total_edge / max(1, T)
            epoch_edge_f1 += total_edge_f1 / max(1, T)
            epoch_hole_mae += total_hole_mae / max(1, T)
            epoch_seen_mae += total_seen_mae / max(1, T)
            batch_prev_cov_hole = total_prev_cov_hole / max(1.0, total_prev_cov_hole_count)
            epoch_prev_cov_hole += batch_prev_cov_hole
            epoch_iters += 1

    if epoch_iters <= 0:
        return None

    denom = float(epoch_iters)
    return {
        'loss': epoch_loss / denom,
        'recon': epoch_recon / denom,
        'cons': epoch_cons / denom,
        'smooth': epoch_smooth / denom,
        'hole_smooth': epoch_hole_smooth / denom,
        'grad': epoch_grad / denom,
        'lap': epoch_lap / denom,
        'hard': epoch_hard / denom,
        'edge': epoch_edge / denom,
        'edge_f1': epoch_edge_f1 / denom,
        'hole_mae': epoch_hole_mae / denom,
        'seen_mae': epoch_seen_mae / denom,
        'prev_cov_hole': epoch_prev_cov_hole / denom,
        'iters': int(epoch_iters),
    }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir) if SummaryWriter is not None else None

    print(f"Loading data from {args.data_dir}...")
    train_set = HeightMapDataset(
        args.data_dir,
        split='train',
        min_seq_len=args.min_seq_len,
        sequence_length=args.seq_len,
        map_size=args.map_size,
        resolution=args.resolution,
        fill_value=args.fill_value,
        use_local_frame=not args.world_frame,
        gravity_aligned=args.gravity_aligned,
        augment_dropout=args.augment_dropout,
        augment_jitter_xy=args.augment_jitter_xy,
        augment_jitter_z=args.augment_jitter_z,
        augment_tilt_deg=args.augment_tilt_deg,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Train samples: {len(train_set)}")
    val_loader = None
    if args.run_val:
        val_set = HeightMapDataset(
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
        print(f"Val samples: {len(val_set)}")
        if len(val_set) > 0:
            val_batch_size = int(args.val_batch_size) if args.val_batch_size > 0 else int(args.batch_size)
            val_loader = DataLoader(
                val_set,
                batch_size=val_batch_size,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=args.val_num_workers,
                pin_memory=True,
            )
        else:
            print('[val] dataset is empty; skip validation.')

    resume_obj = None
    resume_state_dict = None
    if args.resume_ckpt:
        resume_obj = torch.load(args.resume_ckpt, map_location=device)
        resume_state_dict = resume_obj.get('model_state_dict', resume_obj)

    model = HeightRecurrentUNet(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        out_channels=1,
        use_edge_head=args.use_edge_head,
        norm_type=args.norm_type,
        group_norm_groups=args.group_norm_groups,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    start_epoch = 0
    best_val_hole_mae = float('inf')
    best_epoch = -1
    if args.resume_ckpt:
        load_result = model.load_state_dict(resume_state_dict, strict=False)
        missing = list(getattr(load_result, 'missing_keys', []))
        unexpected = list(getattr(load_result, 'unexpected_keys', []))
        if missing or unexpected:
            print(
                f"[resume] key mismatch: missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_example={missing[:3] if missing else []} "
                f"unexpected_example={unexpected[:3] if unexpected else []}"
            )
        if (not args.no_resume_optimizer) and ('optimizer_state_dict' in resume_obj):
            try:
                optimizer.load_state_dict(resume_obj['optimizer_state_dict'])
            except Exception as e:
                print(f"[resume] failed to load optimizer state: {e}")
        start_epoch = int(resume_obj.get('epoch', -1)) + 1
        best_val_hole_mae = float(resume_obj.get('best_val_hole_mae', float('inf')))
        best_epoch = int(resume_obj.get('best_epoch', -1))
        print(f"[resume] loaded {args.resume_ckpt}, start_epoch={start_epoch}")
        if best_epoch >= 0 and np.isfinite(best_val_hole_mae):
            print(f"[resume] best_val_hole_mae={best_val_hole_mae:.6f} at epoch={best_epoch}")
        if args.zero_head_on_resume:
            if zero_output_head(model):
                print('[resume] output head reinitialized to zeros')
            else:
                print('[resume] output head reinit skipped (no compatible head)')

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_cons = 0.0
        epoch_smooth = 0.0
        epoch_hole_smooth = 0.0
        epoch_grad = 0.0
        epoch_lap = 0.0
        epoch_hard = 0.0
        epoch_edge = 0.0
        epoch_edge_f1 = 0.0
        epoch_hole_mae = 0.0
        epoch_seen_mae = 0.0
        epoch_prev_cov_hole = 0.0
        epoch_iters = 0

        for i, batch in enumerate(train_loader):
            if args.max_iters_per_epoch > 0 and i >= args.max_iters_per_epoch:
                break
            optimizer.zero_grad()

            in_height = batch['in_height'].to(device)  # [B, T, 1, H, W]
            in_mask = batch['in_mask'].to(device)      # [B, T, 1, H, W]
            gt_height = batch['gt_height'].to(device)  # [B, T, 1, H, W]
            gt_mask = batch['gt_mask'].to(device)      # [B, T, 1, H, W]
            pose7 = batch['pose7'].to(device)          # [B, T, 7]

            total_loss = torch.zeros((), device=device)
            total_recon = 0.0
            total_cons = 0.0
            total_smooth = 0.0
            total_hole_smooth = 0.0
            total_grad = 0.0
            total_lap = 0.0
            total_hard = 0.0
            total_edge = 0.0
            total_edge_f1 = 0.0
            total_hole_mae = 0.0
            total_seen_mae = 0.0
            total_prev_cov_hole = 0.0
            total_prev_cov_hole_count = 0.0

            prev_pred = None
            prev_pose = None
            prev_valid_state = None
            T = in_height.shape[1]
            for t in range(T):
                meas_h_raw = in_height[:, t]
                meas_m_raw = in_mask[:, t]
                target_h = gt_height[:, t]
                target_m = gt_mask[:, t]
                cur_pose = pose7[:, t]

                # Input-only visibility augmentation. Keep loss masks on raw measurements.
                meas_h_aug, meas_m_aug = augment_measurement_visibility(
                    meas_h_raw,
                    meas_m_raw,
                    args.fill_value,
                    args.mask_aug_prob,
                    args.mask_aug_max_shift,
                    args.mask_morph_prob,
                    args.mask_morph_kernel,
                )

                if args.disable_prev or prev_pred is None:
                    prev_in = meas_h_aug
                    prev_in_valid = meas_m_aug
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

                meas_h_in, meas_m_in, prev_in_in, prev_in_valid_in = augment_model_input_channels(
                    meas_h_aug,
                    meas_m_aug,
                    prev_in,
                    prev_in_valid,
                    input_mask_dropout_prob=args.input_mask_dropout_prob,
                    input_prev_valid_dropout_prob=args.input_prev_valid_dropout_prob,
                    input_invalid_jitter=args.input_invalid_jitter,
                )
                model_input = build_model_input(
                    meas_h_in,
                    meas_m_in,
                    prev_in_in,
                    prev_in_valid_in,
                    args.in_channels,
                )
                prev_base = torch.where(
                    prev_in_valid > 0.5,
                    prev_in,
                    torch.full_like(prev_in, float(args.fill_value)),
                )
                base_h = torch.where(meas_m_raw > 0.5, meas_h_raw, prev_base)

                model_out = model(model_input)
                pred_core, pred_edge_logits = unpack_model_outputs(model_out)
                if args.residual_from_base:
                    residual = pred_core
                    if args.residual_tanh:
                        residual = args.residual_scale * torch.tanh(residual)
                    else:
                        residual = args.residual_scale * residual
                    pred_h = base_h + residual
                else:
                    pred_h = pred_core
                pred_h = torch.nan_to_num(pred_h, nan=float(args.fill_value), posinf=1e3, neginf=-1e3)
                hole_m = ((target_m > 0.5) & (meas_m_raw <= 0.5)).float()
                seen_m = ((target_m > 0.5) & (meas_m_raw > 0.5)).float()
                recon_w = args.seen_weight * seen_m + args.hole_weight * hole_m
                recon_w = recon_w * edge_weight_map(target_h, target_m, args.edge_weight_gain)
                if args.hole_structure_gain > 0.0:
                    structure_mag = torch.abs(target_h - base_h) * hole_m
                    structure_mean = structure_mag.sum() / hole_m.sum().clamp_min(1.0)
                    structure_norm = structure_mag / structure_mean.clamp_min(1e-6)
                    structure_factor = 1.0 + args.hole_structure_gain * structure_norm
                    recon_w = recon_w * torch.where(hole_m > 0.5, structure_factor, torch.ones_like(structure_factor))
                loss_recon = weighted_masked_smooth_l1(pred_h, target_h, recon_w)
                if args.meas_consistency_weight > 0.0:
                    loss_cons = masked_l1(pred_h, meas_h_raw, meas_m_raw)
                else:
                    loss_cons = pred_h.new_zeros(())
                loss_smooth = smoothness_loss(pred_h)
                if args.hole_smooth_weight > 0.0:
                    loss_hole_smooth = hole_tv_loss(pred_h, hole_m)
                else:
                    loss_hole_smooth = pred_h.new_zeros(())
                if args.grad_match_weight > 0.0:
                    loss_grad = masked_grad_l1(pred_h, target_h, target_m)
                else:
                    loss_grad = pred_h.new_zeros(())
                if args.laplace_weight > 0.0:
                    loss_lap = masked_laplace_l1(pred_h, target_h, hole_m)
                else:
                    loss_lap = pred_h.new_zeros(())
                if args.hole_hard_weight > 0.0:
                    loss_hard = hole_hard_l1(pred_h, target_h, hole_m, hard_ratio=args.hole_hard_ratio)
                else:
                    loss_hard = pred_h.new_zeros(())
                if args.edge_head_weight > 0.0 and pred_edge_logits is not None:
                    target_edge = build_edge_target(target_h, target_m, edge_thresh=args.edge_target_thresh)
                    loss_edge = masked_edge_bce_with_logits(
                        pred_edge_logits,
                        target_edge,
                        target_m,
                        base_pos_weight=args.edge_pos_weight,
                        auto_pos_weight=args.edge_pos_weight_auto,
                    )
                    edge_f1 = masked_edge_f1(
                        pred_edge_logits,
                        target_edge,
                        target_m,
                        threshold=args.edge_eval_thresh,
                    )
                else:
                    loss_edge = pred_h.new_zeros(())
                    edge_f1 = pred_h.new_zeros(())
                hole_mae = masked_l1(pred_h, target_h, hole_m)
                seen_mae = masked_l1(pred_h, target_h, seen_m)

                if (not args.disable_prev) and (t > 0):
                    hole_denom = hole_m.sum()
                    if float(hole_denom.item()) > 0.0:
                        prev_cov_hole = (prev_in_valid * hole_m).sum() / hole_denom
                        total_prev_cov_hole += float(prev_cov_hole.item())
                        total_prev_cov_hole_count += 1.0

                loss = (
                    loss_recon
                    + args.meas_consistency_weight * loss_cons
                    + args.smooth_weight * loss_smooth
                    + args.hole_smooth_weight * loss_hole_smooth
                    + args.grad_match_weight * loss_grad
                    + args.laplace_weight * loss_lap
                    + args.hole_hard_weight * loss_hard
                    + args.edge_head_weight * loss_edge
                )

                total_loss = total_loss + loss
                total_recon += float(loss_recon.item())
                total_cons += float(loss_cons.item())
                total_smooth += float(loss_smooth.item())
                total_hole_smooth += float(loss_hole_smooth.item())
                total_grad += float(loss_grad.item())
                total_lap += float(loss_lap.item())
                total_hard += float(loss_hard.item())
                total_edge += float(loss_edge.item())
                total_edge_f1 += float(edge_f1.item())
                total_hole_mae += float(hole_mae.item())
                total_seen_mae += float(seen_mae.item())
                if args.memory_meas_override:
                    # Keep measured cells sharp in recurrent state to avoid recursive flattening.
                    prev_pred = torch.where(meas_m_raw > 0.5, meas_h_raw, pred_h).detach()
                else:
                    prev_pred = pred_h.detach()
                prev_pose = cur_pose
                prev_valid_state = torch.maximum(meas_m_raw, prev_in_valid).detach()

            total_loss = total_loss / max(1, T)
            if not torch.isfinite(total_loss):
                print(
                    f"[warn] non-finite loss at epoch={epoch} iter={i}; "
                    "skip this batch to protect optimizer state."
                )
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            epoch_loss += float(total_loss.item())
            epoch_recon += total_recon / max(1, T)
            epoch_cons += total_cons / max(1, T)
            epoch_smooth += total_smooth / max(1, T)
            epoch_hole_smooth += total_hole_smooth / max(1, T)
            epoch_grad += total_grad / max(1, T)
            epoch_lap += total_lap / max(1, T)
            epoch_hard += total_hard / max(1, T)
            epoch_edge += total_edge / max(1, T)
            epoch_edge_f1 += total_edge_f1 / max(1, T)
            epoch_hole_mae += total_hole_mae / max(1, T)
            epoch_seen_mae += total_seen_mae / max(1, T)
            batch_prev_cov_hole = total_prev_cov_hole / max(1.0, total_prev_cov_hole_count)
            epoch_prev_cov_hole += batch_prev_cov_hole
            epoch_iters += 1
            global_step += 1

            if i % args.log_interval == 0:
                msg = (
                    f"Epoch [{epoch}/{args.epochs}] Iter [{i}/{len(train_loader)}] "
                    f"Loss={total_loss.item():.4f} "
                    f"(Recon={total_recon/max(1, T):.4f}, Cons={total_cons/max(1, T):.4f}, "
                    f"Smooth={total_smooth/max(1, T):.4f}, "
                    f"HoleSmooth={total_hole_smooth/max(1, T):.4f}, "
                    f"Grad={total_grad/max(1, T):.4f}, "
                    f"Lap={total_lap/max(1, T):.4f}, "
                    f"Hard={total_hard/max(1, T):.4f}, "
                    f"Edge={total_edge/max(1, T):.4f}, EdgeF1={total_edge_f1/max(1, T):.4f}, "
                    f"HoleMAE={total_hole_mae/max(1, T):.4f}, SeenMAE={total_seen_mae/max(1, T):.4f}, "
                    f"PrevCovHole={batch_prev_cov_hole:.4f})"
                )
                print(msg)
                if writer is not None:
                    writer.add_scalar('Loss/train', total_loss.item(), global_step)
                    writer.add_scalar('Loss/recon', total_recon / max(1, T), global_step)
                    writer.add_scalar('Loss/consistency', total_cons / max(1, T), global_step)
                    writer.add_scalar('Loss/smooth', total_smooth / max(1, T), global_step)
                    writer.add_scalar('Loss/hole_smooth', total_hole_smooth / max(1, T), global_step)
                    writer.add_scalar('Loss/grad_match', total_grad / max(1, T), global_step)
                    writer.add_scalar('Loss/laplace', total_lap / max(1, T), global_step)
                    writer.add_scalar('Loss/hole_hard', total_hard / max(1, T), global_step)
                    writer.add_scalar('Loss/edge_bce', total_edge / max(1, T), global_step)
                    writer.add_scalar('Edge/f1', total_edge_f1 / max(1, T), global_step)
                    writer.add_scalar('Error/hole_mae', total_hole_mae / max(1, T), global_step)
                    writer.add_scalar('Error/seen_mae', total_seen_mae / max(1, T), global_step)
                    writer.add_scalar('Coverage/prev_valid_on_hole', batch_prev_cov_hole, global_step)

        scheduler.step()
        denom = max(1, epoch_iters)
        print(
            f"Epoch {epoch} Done. "
            f"AvgLoss={epoch_loss/denom:.4f} "
            f"AvgRecon={epoch_recon/denom:.4f} "
            f"AvgCons={epoch_cons/denom:.4f} "
            f"AvgSmooth={epoch_smooth/denom:.4f} "
            f"AvgHoleSmooth={epoch_hole_smooth/denom:.4f} "
            f"AvgGrad={epoch_grad/denom:.4f} "
            f"AvgLap={epoch_lap/denom:.4f} "
            f"AvgHard={epoch_hard/denom:.4f} "
            f"AvgEdge={epoch_edge/denom:.4f} "
            f"AvgEdgeF1={epoch_edge_f1/denom:.4f} "
            f"AvgHoleMAE={epoch_hole_mae/denom:.4f} "
            f"AvgSeenMAE={epoch_seen_mae/denom:.4f} "
            f"AvgPrevCovHole={epoch_prev_cov_hole/denom:.4f}"
        )
        if writer is not None:
            writer.add_scalar('Epoch/train_loss', epoch_loss / denom, epoch)
            writer.add_scalar('Epoch/train_hole_mae', epoch_hole_mae / denom, epoch)
            writer.add_scalar('Epoch/train_seen_mae', epoch_seen_mae / denom, epoch)

        val_metrics = None
        if val_loader is not None and args.val_interval > 0 and ((epoch + 1) % args.val_interval == 0):
            val_metrics = evaluate_val_epoch(model, val_loader, args, device)
            if val_metrics is None:
                print(f"[val] epoch={epoch} skipped (no valid iterations)")
            else:
                print(
                    f"[val] epoch={epoch} "
                    f"Loss={val_metrics['loss']:.4f} "
                    f"Recon={val_metrics['recon']:.4f} "
                    f"HoleMAE={val_metrics['hole_mae']:.4f} "
                    f"SeenMAE={val_metrics['seen_mae']:.4f} "
                    f"PrevCovHole={val_metrics['prev_cov_hole']:.4f} "
                    f"(iters={val_metrics['iters']})"
                )
                if writer is not None:
                    writer.add_scalar('Val/loss', val_metrics['loss'], epoch)
                    writer.add_scalar('Val/recon', val_metrics['recon'], epoch)
                    writer.add_scalar('Val/hole_mae', val_metrics['hole_mae'], epoch)
                    writer.add_scalar('Val/seen_mae', val_metrics['seen_mae'], epoch)
                    writer.add_scalar('Val/prev_cov_hole', val_metrics['prev_cov_hole'], epoch)

                if val_metrics['hole_mae'] < best_val_hole_mae:
                    best_val_hole_mae = float(val_metrics['hole_mae'])
                    best_epoch = int(epoch)
                    best_ckpt_path = os.path.join(args.checkpoint_dir, args.best_ckpt_name)
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_args': vars(args),
                            'best_val_hole_mae': best_val_hole_mae,
                            'best_epoch': best_epoch,
                            'val_metrics': val_metrics,
                        },
                        best_ckpt_path,
                    )
                    print(
                        f"[best] updated {best_ckpt_path} "
                        f"(epoch={best_epoch}, hole_mae={best_val_hole_mae:.6f})"
                    )

        ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_args': vars(args),
                'best_val_hole_mae': best_val_hole_mae,
                'best_epoch': best_epoch,
                'val_metrics': val_metrics,
            },
            ckpt_path,
        )

    if writer is not None:
        writer.close()
    print("Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/final_200k')
    parser.add_argument('--log_dir', type=str, default='logs/height_experiment_1')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_ckpt', type=str, default='', help='Optional checkpoint path to resume training from')
    parser.add_argument('--no_resume_optimizer', action='store_true', default=False, help='Do not load optimizer state when resuming')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--min_seq_len', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=4)  # meas_height, meas_mask, prev_pred, prev_valid_mask
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
    parser.add_argument('--augment_dropout', type=float, default=0.1, help='Random point dropout ratio for measurement augmentation')
    parser.add_argument('--augment_jitter_xy', type=float, default=0.03, help='XY jitter magnitude (meters) for measurement augmentation')
    parser.add_argument('--augment_jitter_z', type=float, default=0.02, help='Z jitter magnitude (meters) for measurement augmentation')
    parser.add_argument('--augment_tilt_deg', type=float, default=2.0, help='Random roll/pitch augmentation for input points')

    parser.add_argument('--meas_consistency_weight', type=float, default=0.0)
    parser.add_argument('--smooth_weight', type=float, default=0.01)
    parser.add_argument('--hole_smooth_weight', type=float, default=0.0, help='Additional TV smoothness on hole regions only')
    parser.add_argument('--grad_match_weight', type=float, default=0.0, help='Gradient alignment loss weight for edge/detail preservation')
    parser.add_argument('--laplace_weight', type=float, default=0.0, help='Second-order (Laplacian) shape loss weight on hole regions')
    parser.add_argument('--hole_hard_weight', type=float, default=0.0, help='Top-k hard-pixel L1 loss weight on hole regions')
    parser.add_argument('--hole_hard_ratio', type=float, default=0.2, help='Top-k ratio for hole hard loss (0,1]')
    parser.add_argument('--hole_structure_gain', type=float, default=0.0, help='Extra hole weighting by |GT-base| magnitude')
    parser.add_argument('--edge_head_weight', type=float, default=0.0, help='Auxiliary edge BCE loss weight')
    parser.add_argument('--edge_target_thresh', type=float, default=0.01, help='Height-gradient threshold (m) for GT edge labels')
    parser.add_argument('--edge_pos_weight', type=float, default=1.0, help='Positive-class weight for edge BCE (if auto disabled)')
    parser.add_argument('--edge_pos_weight_auto', action='store_true', default=True, help='Auto-balance edge BCE positive class')
    parser.add_argument('--no_edge_pos_weight_auto', action='store_false', dest='edge_pos_weight_auto')
    parser.add_argument('--edge_eval_thresh', type=float, default=0.5, help='Sigmoid threshold for edge F1 metric')
    parser.add_argument('--hole_weight', type=float, default=3.0)
    parser.add_argument('--seen_weight', type=float, default=1.0)
    parser.add_argument('--edge_weight_gain', type=float, default=0.0, help='Increase recon weight on GT height edges')
    parser.add_argument('--mask_aug_prob', type=float, default=0.5, help='Probability of random translating measurement visibility mask')
    parser.add_argument('--mask_aug_max_shift', type=int, default=2, help='Max pixel shift for measurement visibility translation')
    parser.add_argument('--mask_morph_prob', type=float, default=0.2, help='Probability of random mask dilation/erosion')
    parser.add_argument('--mask_morph_kernel', type=int, default=3, help='Kernel size for mask dilation/erosion')
    parser.add_argument('--input_mask_dropout_prob', type=float, default=0.0, help='Dropout probability for measurement mask input channel')
    parser.add_argument('--input_prev_valid_dropout_prob', type=float, default=0.0, help='Dropout probability for warped-prev validity channel')
    parser.add_argument('--input_invalid_jitter', type=float, default=0.0, help='Uniform jitter amplitude added to invalid pixels of input height channels')
    parser.add_argument('--prev_valid_threshold', type=float, default=0.5, help='Binarization threshold for warped previous-valid mask before model input')
    parser.add_argument('--memory_meas_override', action='store_true', default=True, help='Update recurrent prev map with measured heights on observed cells')
    parser.add_argument('--no_memory_meas_override', action='store_false', dest='memory_meas_override')
    parser.add_argument('--residual_from_base', action='store_true', default=False, help='Predict residual on top of base map (measured cells + warped prev)')
    parser.add_argument('--residual_scale', type=float, default=0.2, help='Scale for residual branch output')
    parser.add_argument('--residual_tanh', action='store_true', default=True, help='Use tanh-bounded residual output')
    parser.add_argument('--no_residual_tanh', action='store_false', dest='residual_tanh')
    parser.add_argument('--zero_head_on_resume', action='store_true', default=False, help='Reinitialize output head to zero after loading resume checkpoint')
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--max_iters_per_epoch', type=int, default=0, help='If >0, cap iterations per epoch for fast sanity runs')
    parser.add_argument('--run_val', action='store_true', default=True, help='Run validation at epoch end (default on)')
    parser.add_argument('--no_run_val', action='store_false', dest='run_val')
    parser.add_argument('--val_interval', type=int, default=1, help='Run validation every N epochs')
    parser.add_argument('--val_batch_size', type=int, default=0, help='Validation batch size (<=0 uses train batch_size)')
    parser.add_argument('--val_num_workers', type=int, default=2)
    parser.add_argument('--max_val_iters_per_epoch', type=int, default=0, help='If >0, cap val iterations per epoch')
    parser.add_argument('--best_ckpt_name', type=str, default='checkpoint_best_hole_mae.pth')
    parser.add_argument('--lr_step', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.5)

    parser.set_defaults(align_prev=True, gravity_aligned=True)
    args = parser.parse_args()
    train(args)
