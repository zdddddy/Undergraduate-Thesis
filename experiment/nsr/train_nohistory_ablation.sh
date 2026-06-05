#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-}"

if [[ -z "$DATA_DIR" ]]; then
  echo "[ERROR] DATA_DIR is required, for example:"
  echo "  export DATA_DIR=/path/to/nsr_go2_v3/split_env_90_10"
  exit 1
fi

cd "$ROOT_DIR"

"$PYTHON_BIN" -u -m nsr_height.train \
  --data_dir "$DATA_DIR" \
  --log_dir results/nsr/logs/nsr_go2_v3_nohistory_main_e10_seq24 \
  --checkpoint_dir results/nsr/checkpoints/nsr_go2_v3_nohistory_main_e10_seq24 \
  --batch_size 8 \
  --seq_len 24 \
  --min_seq_len 24 \
  --epochs 3 \
  --lr 0.001 \
  --num_workers 8 \
  --in_channels 2 \
  --base_channels 32 \
  --norm_type group \
  --group_norm_groups 8 \
  --map_size 3.2 \
  --resolution 0.05 \
  --gravity_aligned \
  --disable_prev \
  --no_align_prev \
  --augment_dropout 0.1 \
  --augment_jitter_xy 0.03 \
  --augment_jitter_z 0.02 \
  --augment_tilt_deg 2.0 \
  --smooth_weight 0.01 \
  --hole_weight 3.0 \
  --seen_weight 1.0 \
  --mask_aug_prob 0.5 \
  --mask_aug_max_shift 2 \
  --mask_morph_prob 0.2 \
  --mask_morph_kernel 3 \
  --memory_meas_override \
  --grad_clip 5.0 \
  --lr_step 5 \
  --lr_gamma 0.5

"$PYTHON_BIN" -u -m nsr_height.train \
  --data_dir "$DATA_DIR" \
  --log_dir results/nsr/logs/nsr_go2_v3_nohistory_edgeft_v1 \
  --checkpoint_dir results/nsr/checkpoints/nsr_go2_v3_nohistory_edgeft_v1 \
  --resume_ckpt results/nsr/checkpoints/nsr_go2_v3_nohistory_main_e10_seq24/checkpoint_epoch_2.pth \
  --no_resume_optimizer \
  --batch_size 8 \
  --seq_len 24 \
  --min_seq_len 24 \
  --epochs 7 \
  --lr 0.0003 \
  --num_workers 8 \
  --in_channels 2 \
  --base_channels 32 \
  --norm_type group \
  --group_norm_groups 8 \
  --map_size 3.2 \
  --resolution 0.05 \
  --gravity_aligned \
  --disable_prev \
  --no_align_prev \
  --augment_dropout 0.1 \
  --augment_jitter_xy 0.03 \
  --augment_jitter_z 0.02 \
  --augment_tilt_deg 2.0 \
  --smooth_weight 0.005 \
  --grad_match_weight 0.05 \
  --laplace_weight 0.02 \
  --hole_hard_weight 0.05 \
  --hole_hard_ratio 0.2 \
  --hole_structure_gain 0.3 \
  --hole_weight 3.0 \
  --seen_weight 1.0 \
  --edge_weight_gain 0.5 \
  --mask_aug_prob 0.5 \
  --mask_aug_max_shift 2 \
  --mask_morph_prob 0.2 \
  --mask_morph_kernel 3 \
  --memory_meas_override \
  --grad_clip 5.0 \
  --lr_step 5 \
  --lr_gamma 0.5
