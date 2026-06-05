# 论文实验目录

本目录只保留论文第 4 章中除 sim2sim 外的实验脚本。

## NSR 地形表征实验

| 论文内容 | 脚本 | 输出目录 |
| --- | --- | --- |
| 表 4-4 整体重建误差 | `nsr/offline_validate.py` | `results/experiments/nsr/full_val_edgeft_v1/` |
| 表 4-5 不同地形类型误差、图 4-1 | `nsr/terrain_type_analysis.py` | `results/experiments/nsr/terrain_type_analysis_edgeft_v1/` |
| 图 4-2 盲区补全可视化 | `nsr/temporal_blind_completion.py` | `results/experiments/nsr/temporal_blind_completion_long_no_box/` |
| 表 4-6 历史信息消融 | `nsr/train_nohistory_ablation.sh`、`nsr/history_ablation_design.md` | `results/experiments/nsr/ablation_history/` |

## Go2 越障控制实验

| 论文内容 | 脚本 | 输出目录 |
| --- | --- | --- |
| 图 4-3、图 4-4、表 4-7 训练过程 | `go2/training_process_analysis.py` | `results/experiments/go2/training_process_analysis/` |
| 表 4-8、表 4-9 多地形与 Blind-History 对比 | `go2/five_terrain_distinctive_comparison.py` | `results/experiments/go2/five_terrain_distinctive_comparison/` |
| 表 4-10 综合扰动鲁棒性 | `go2/nsr_five_terrain_robustness.py` | `results/experiments/go2/nsr_five_terrain_robustness/` |

`go2/paper_common.py` 只存放论文实验共用的地形标签、评估随机化开关和结果描述函数。
