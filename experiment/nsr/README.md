# NSR 论文实验

本目录保留论文第 4.2 节的神经地形表征实验脚本。

## 表 4-4：整体重建误差

```bash
python experiment/nsr/offline_validate.py \
  --out_dir results/experiments/nsr/full_val_edgeft_v1
```

主要输出：

- `metrics_summary.json`：Overall、Seen、Hole 区域的 MAE/RMSE/Mean Error。
- `per_frame_metrics.csv`：逐帧误差。
- `per_sequence_metrics.csv`：逐序列误差。
- `predictions/seq_*.npz`：后续地形类型分析和盲区可视化使用的预测结果。

## 表 4-5 和图 4-1：地形类型分析

```bash
python experiment/nsr/terrain_type_analysis.py
```

默认读取 `results/experiments/nsr/full_val_edgeft_v1/predictions`，输出到：

```text
results/experiments/nsr/terrain_type_analysis_edgeft_v1/
```

保留内容包括 `terrain_metrics.csv`、`terrain_report.md`、各地形可视化图和误差分布图。

## 图 4-2：盲区补全可视化

```bash
python experiment/nsr/temporal_blind_completion.py
```

默认输出到：

```text
results/experiments/nsr/temporal_blind_completion_long_no_box/
```

该脚本只保留论文中使用的连续帧示例。

## 表 4-6：历史信息消融

```bash
export DATA_DIR=/path/to/nsr_go2_v3/split_env_90_10
bash experiment/nsr/train_nohistory_ablation.sh
```

当前已保留论文表格所需结果：

```text
results/experiments/nsr/ablation_history/
```

具体训练配置和对比口径见 `history_ablation_design.md`。
