import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from dataset import PointCloudDataset, collate_fn
from model import RecurrentUNet


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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading data from {args.data_dir}...")
    train_set = PointCloudDataset(
        args.data_dir,
        split='train',
        min_seq_len=args.min_seq_len,
        sequence_length=args.seq_len,
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

    model = RecurrentUNet(in_channels=args.in_channels, out_channels=1, D=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.SmoothL1Loss()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            seq = batch['seq']
            prev_occ = None
            prev_off = None
            total_loss = 0.0
            total_cls = 0.0
            total_reg = 0.0

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

                union_coords, targets = build_union_targets(out_occ_cur.C, gt_coords_t)
                preds_union = gather_features_at_coords(
                    out_occ_cur, union_coords, default=-10.0, coords_dim=5, return_mask=False
                )
                loss_cls = criterion_cls(preds_union, targets)

                pred_off_at_gt, mask_reg = gather_features_at_coords(
                    out_off_cur, gt_coords_t, default=0.0, coords_dim=5, return_mask=True
                )
                if torch.any(mask_reg):
                    loss_reg = criterion_reg(pred_off_at_gt[mask_reg], gt_offsets[mask_reg])
                else:
                    loss_reg = torch.zeros((), device=feats.device)

                if out_occ_cur.F.numel() > 0:
                    loss_sparsity = torch.mean(torch.sigmoid(out_occ_cur.F))
                else:
                    loss_sparsity = torch.zeros((), device=feats.device)
                loss = loss_cls + args.reg_weight * loss_reg + args.sparsity_weight * loss_sparsity

                total_loss += loss
                total_cls += loss_cls.item()
                total_reg += loss_reg.item()

                prev_occ, prev_off = out_occ_cur.detach(), out_off_cur.detach()

            total_loss = total_loss / max(1, len(seq))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            epoch_loss += total_loss.item()
            global_step += 1

            if i % args.log_interval == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Iter [{i}/{len(train_loader)}] "
                    f"Loss: {total_loss.item():.4f} (Cls: {total_cls/len(seq):.4f}, Reg: {total_reg/len(seq):.4f})"
                )
                if writer is not None:
                    writer.add_scalar('Loss/train', total_loss.item(), global_step)
                    writer.add_scalar('Loss/cls', total_cls / len(seq), global_step)
                    writer.add_scalar('Loss/reg', total_reg / len(seq), global_step)

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            ckpt_path,
        )

    if writer is not None:
        writer.close()
    print("Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='dataset/final_200k')
    parser.add_argument('--log_dir', type=str, default='logs/experiment_1')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=12)
    parser.add_argument('--min_seq_len', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--in_channels', type=int, default=3)

    parser.add_argument('--sparsity_weight', type=float, default=0.1)
    parser.add_argument('--reg_weight', type=float, default=1.0)
    parser.add_argument('--prev_threshold', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--lr_step', type=int, default=5)
    parser.add_argument('--lr_gamma', type=float, default=0.5)

    args = parser.parse_args()
    train(args)
