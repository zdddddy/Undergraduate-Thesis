# 实验1：历史表征对齐与融合消融

## 实验目的

验证上一时刻局部高度图预测结果及其位姿对齐机制，是否能够提升盲区补全能力和连续运动过程中的时序稳定性。

## 对比方法

| 方法 | 当前观测 | t-1 预测高度图 | 历史对齐 | 输入通道 | 说明 |
|---|---|---|---|---:|---|
| w/o History | 使用 | 不使用 | 不使用 | 2 | 仅输入当前帧局部观测高度图和当前观测 mask |
| Full model | 使用 | 使用 | 使用 | 4 | 输入当前帧观测、上一时刻预测高度图及其有效 mask，并按机器人位姿对齐到当前帧 |

该实验只改变“是否使用历史表征及其对齐”这一项。训练集、验证集、序列长度、训练轮数、网络宽度、主要损失函数和数据增强策略应保持一致。

## 变量控制

| 项目 | 设置 |
|---|---|
| 数据集 | `$DATA_DIR`，例如 `/path/to/nsr_go2_v3/split_env_90_10` |
| 验证集 | `val` split，全量验证序列 |
| 序列长度 | 24 帧 |
| 地图范围 | `3.2 m x 3.2 m` |
| 分辨率 | `0.05 m` |
| 网络主体 | `HeightRecurrentUNet`, `base_channels=32`, `norm_type=group` |
| 主要损失 | weighted reconstruction + smoothness + gradient matching + Laplacian + hard-hole loss |
| 历史相关差异 | 仅 Full model 使用 `prev_pred + prev_valid` 和 `align_prev` |

## 训练设置

完整模型训练流程为主训练后再进行 `edgeft_v1` 结构细节微调。因此 w/o History 也采用同样两段训练流程，只把输入与历史开关改掉。当前仓库只保留论文表格所需的最终结果，中间主训练 checkpoint 可由 `train_nohistory_ablation.sh` 重新生成。

参考主模型 checkpoint 元信息：

| 阶段 | 主模型 checkpoint | `train_args["epochs"]` | 实际含义 |
|---|---|---:|---|
| main | 运行脚本时生成的中间 checkpoint | 10 | 主训练目标轮数为 10，best 在 epoch 2 |
| edgeft | `results/nsr/checkpoints/nsr_go2_v3_edgeft_v1/checkpoint_best_hole_mae.pth` | 7 | 从 main best 的 epoch 2 继续，实际微调 epoch 3-6 |

因此 no-history 消融按主模型实际采用的阶段节点训练：main 阶段跑完 epoch 2，edgeft 阶段从 epoch 2 继续跑完 epoch 6。`train.py` 里的 `--epochs` 是结束 epoch 索引上界；从零训练到 epoch 2 需要设为 `--epochs 3`，从 epoch 2 resume 后跑到 epoch 6 需要设为 `--epochs 7`。

Full model 使用当前完整模型结果：

```bash
results/nsr/checkpoints/nsr_go2_v3_edgeft_v1/checkpoint_best_hole_mae.pth
```

w/o History 第一段：主训练到 epoch 2，关闭历史输入：

```bash
export DATA_DIR=/path/to/nsr_go2_v3/split_env_90_10
python -m nsr_height.train \
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
```

w/o History 第二段：结构细节微调到 epoch 6，损失权重与 Full model 的 `edgeft_v1` 保持一致：

```bash
export DATA_DIR=/path/to/nsr_go2_v3/split_env_90_10
python -m nsr_height.train \
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
  --prev_valid_threshold 0.5 \
  --memory_meas_override \
  --grad_clip 5.0 \
  --lr_step 5 \
  --lr_gamma 0.5
```

说明：`--in_channels 2` 表示只输入 `[current height, current mask]`；`--disable_prev --no_align_prev` 保证训练和推理阶段都不使用上一时刻预测。

## 离线验证

w/o History：

```bash
python experiment/nsr/offline_validate.py \
  --ckpt results/nsr/checkpoints/nsr_go2_v3_nohistory_edgeft_v1/checkpoint_best_hole_mae.pth \
  --out_dir results/experiments/nsr/ablation_history/no_history \
  --save_predictions \
  --save_frame_stride 1 \
  --edge_thresh 0.01 \
  --in_channels 2 \
  --disable_prev \
  --no_align_prev
```

Full model：

```bash
python experiment/nsr/offline_validate.py \
  --ckpt results/nsr/checkpoints/nsr_go2_v3_edgeft_v1/checkpoint_best_hole_mae.pth \
  --out_dir results/experiments/nsr/ablation_history/full_model \
  --save_predictions \
  --save_frame_stride 1 \
  --edge_thresh 0.01 \
  --in_channels 4 \
  --gravity_aligned
```

## 指标

| 指标 | 统计区域 | 作用 |
|---|---|---|
| Overall MAE | GT 有效区域 | 评价整体重建误差 |
| Seen MAE | 当前可见且 GT 有效区域 | 评价可见区域保持能力 |
| Hole MAE | 当前不可见但 GT 有效区域 | 评价盲区补全能力 |
| Edge MAE | GT 高度梯度边界区域 | 评价台阶、障碍物边界等结构区域误差 |

Edge MAE 使用 `edge_thresh=0.01 m` 在 `offline_validate.py` 内构建 GT 高度梯度边界 mask。若需要强调强结构地形，可再用 `terrain_type_analysis.py` 单独统计上台阶、下台阶和离散障碍物子集。

## 论文表格

| 方法 | Overall MAE | Seen MAE | Hole MAE | Edge MAE |
|---|---:|---:|---:|---:|
| w/o History |  |  |  |  |
| Full model |  |  |  |  |

可额外给出相对提升：

```text
Relative improvement = (MAE_no_history - MAE_full) / MAE_no_history * 100%
```

## 预期分析重点

1. 如果 Full model 的 Hole MAE 明显低于 w/o History，说明上一时刻预测结果能够为当前盲区提供有效先验。
2. 如果 Edge MAE 下降，说明历史对齐不仅改善平滑区域补全，也能保留台阶边缘、离散障碍物边界等强结构信息。
3. Seen MAE 不应明显变差。若 Full model 在 Seen MAE 接近或优于 w/o History，说明历史融合没有破坏当前可见区域观测。
4. 结论表述应强调：历史表征不是简单增加输入通道，而是通过位姿对齐把过去可见的地形信息投影到当前局部坐标系，从而提升连续运动过程中的盲区补全。
