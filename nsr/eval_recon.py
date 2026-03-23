import argparse
import torch
import numpy as np
import MinkowskiEngine as ME

from dataset import PointCloudDataset, collate_fn
from model import RecurrentUNet
from torch.utils.data import DataLoader


def add_time_to_coords(coords, t):
    time_col = coords.new_full((coords.shape[0], 1), int(t))
    return torch.cat([coords, time_col], dim=1)


def gather_features_at_coords(out, query_coords, default=0.0, coords_dim=None, return_mask=False):
    out_coords = out.C
    if out_coords.dtype != torch.int64:
        out_coords = out_coords.long()
    if query_coords.dtype != torch.int64:
        query_coords = query_coords.long()

    if coords_dim is not None:
        out_coords = out_coords[:, :coords_dim]
        query_coords = query_coords[:, :coords_dim]

    combined = torch.cat([out_coords, query_coords], dim=0)
    _, inverse = torch.unique(combined, dim=0, return_inverse=True)
    out_inv = inverse[: out_coords.shape[0]]
    q_inv = inverse[out_coords.shape[0] :]

    feats_for_unique = out.F.new_full((inverse.max().item() + 1, out.F.shape[1]), float(default))
    feats_for_unique[out_inv] = out.F
    gathered = feats_for_unique[q_inv]

    if not return_mask:
        return gathered

    present = out.F.new_zeros((inverse.max().item() + 1,), dtype=torch.bool)
    present[out_inv] = True
    mask = present[q_inv]
    return gathered, mask


def build_union_targets(out_coords, gt_coords):
    if out_coords.dtype != torch.int64:
        out_coords = out_coords.long()
    if gt_coords.dtype != torch.int64:
        gt_coords = gt_coords.long()

    combined = torch.cat([out_coords, gt_coords], dim=0)
    union_coords, inverse = torch.unique(combined, dim=0, return_inverse=True)
    gt_inv = inverse[out_coords.shape[0] :]

    targets = torch.zeros((union_coords.shape[0], 1), device=out_coords.device, dtype=torch.float32)
    targets[gt_inv] = 1.0
    return union_coords, targets


def compute_metrics(pred_coords, gt_coords):
    if pred_coords.numel() == 0 and gt_coords.numel() == 0:
        return 1.0, 1.0, 1.0
    if pred_coords.numel() == 0:
        return 0.0, 0.0, 0.0
    if gt_coords.numel() == 0:
        return 0.0, 0.0, 0.0

    combined = torch.cat([pred_coords, gt_coords], dim=0)
    _, inverse = torch.unique(combined, dim=0, return_inverse=True)
    pred_inv = inverse[: pred_coords.shape[0]]
    gt_inv = inverse[pred_coords.shape[0] :]

    pred_set = torch.zeros((inverse.max().item() + 1,), dtype=torch.bool, device=pred_coords.device)
    gt_set = torch.zeros((inverse.max().item() + 1,), dtype=torch.bool, device=pred_coords.device)
    pred_set[pred_inv] = True
    gt_set[gt_inv] = True

    inter = (pred_set & gt_set).sum().item()
    pred_n = pred_set.sum().item()
    gt_n = gt_set.sum().item()
    union = (pred_set | gt_set).sum().item()

    precision = inter / pred_n if pred_n > 0 else 0.0
    recall = inter / gt_n if gt_n > 0 else 0.0
    iou = inter / union if union > 0 else 0.0
    return precision, recall, iou


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = PointCloudDataset(
        args.data_dir,
        split='val',
        min_seq_len=args.min_seq_len,
        sequence_length=args.seq_len,
    )
    if len(ds) == 0:
        print('[eval] empty dataset')
        return

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = RecurrentUNet(in_channels=args.in_channels, out_channels=1, D=4)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    total_p = total_r = total_iou = 0.0
    total_reg = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            seq = batch['seq']
            prev_occ = None
            prev_off = None

            for t, frame in enumerate(seq):
                coords = frame['coords'].to(device)
                feats = frame['feats'].to(device)
                gt_coords = frame['gt_coords'].to(device)
                gt_offsets = frame['gt_offsets'].to(device)

                coords_curr = add_time_to_coords(coords, 0)
                gt_coords_t = add_time_to_coords(gt_coords, 0)

                if prev_occ is None:
                    in_coords = coords_curr
                    in_feats = feats
                else:
                    prev_time_mask = prev_occ.C[:, -1] == 0
                    if torch.any(prev_time_mask):
                        prev_logits = prev_occ.F[prev_time_mask].squeeze()
                        keep = prev_logits > args.prev_threshold
                        if not torch.any(keep):
                            keep[torch.argmax(prev_logits)] = True
                        prev_coords = prev_occ.C[prev_time_mask][keep].clone()
                        prev_coords[:, -1] = 1
                        prev_feats = prev_off.F[prev_time_mask][keep]
                        in_coords = torch.cat([coords_curr, prev_coords], dim=0)
                        in_feats = torch.cat([feats, prev_feats], dim=0)
                    else:
                        in_coords = coords_curr
                        in_feats = feats

                in_input = ME.SparseTensor(features=in_feats, coordinates=in_coords)
                out_occ, out_off = model(in_input)

                out_time_mask = out_occ.C[:, -1] == 0
                out_occ_cur = ME.SparseTensor(out_occ.F[out_time_mask], coordinates=out_occ.C[out_time_mask])
                out_off_cur = ME.SparseTensor(out_off.F[out_time_mask], coordinates=out_off.C[out_time_mask])

                # classification metrics
                pred_mask = out_occ_cur.F.squeeze() > args.occ_threshold
                pred_coords = out_occ_cur.C[pred_mask]
                precision, recall, iou = compute_metrics(pred_coords, gt_coords_t)

                # regression metric on GT coords
                pred_off_at_gt, mask_reg = gather_features_at_coords(
                    out_off_cur, gt_coords_t, default=0.0, coords_dim=5, return_mask=True
                )
                if torch.any(mask_reg):
                    reg_l1 = torch.mean(torch.abs(pred_off_at_gt[mask_reg] - gt_offsets[mask_reg])).item()
                else:
                    reg_l1 = 0.0

                total_p += precision
                total_r += recall
                total_iou += iou
                total_reg += reg_l1
                steps += 1

                prev_occ, prev_off = out_occ_cur, out_off_cur

            if steps >= args.max_steps:
                break

    if steps == 0:
        print('[eval] no steps')
        return

    print(f"[eval] steps={steps} precision={total_p/steps:.4f} recall={total_r/steps:.4f} iou={total_iou/steps:.4f} reg_l1={total_reg/steps:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--min_seq_len', type=int, default=12)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--prev_threshold', type=float, default=0.0)
    parser.add_argument('--occ_threshold', type=float, default=0.0)
    parser.add_argument('--max_steps', type=int, default=200)
    args = parser.parse_args()
    evaluate(args)
