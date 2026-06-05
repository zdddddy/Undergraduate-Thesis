# 基于强化学习和神经地形表征的四足机器人越障控制

本仓库是论文《基于强化学习和神经地形表征的四足机器人越障控制》的代码整理版，保留论文中除 sim2sim 迁移外的主要内容：神经地形表征（NSR）训练与验证、Go2 越障策略训练、Blind-History 对比基线、多地形评估和结果分析脚本。

## 代码范围

保留内容：

- `nsr_height/`：神经地形表征网络、数据集、训练和可视化。
- `legged_gym/envs/go2/go2_stage2/`：真值/退化真值地形输入策略训练。
- `legged_gym/envs/go2/go2_stage3/`：NSR 在线地形输入策略适配。
- `legged_gym/envs/go2/go2_blind/`：仅依赖本体历史观测的 Blind-History 基线。
- `experiment/`：论文实验复现、离线分析和绘图脚本。
- `results/`：训练日志、检查点、评估 JSON、实验表格和图像。

为支撑上述任务，`go2_ts`、`go2_dreamwaq` 及对应 PPO/runner 模块仍被保留：`stage2/stage3` 依赖 teacher-student 框架，`go2_blind` 依赖 DreamWaQ 风格历史编码基线。

## 阶段映射

论文只描述两个越障策略训练阶段，本代码中阶段名略有偏移：

| 论文阶段 | 代码任务 | 作用 |
| --- | --- | --- |
| 阶段一 | `go2_stage2a` / `go2_stage2b` / `go2_stage2c` | 使用真值地形和退化真值地形训练策略 |
| 阶段二 | `go2_stage3` / `go2_stage3a` / `go2_stage3b` | 接入 NSR 预测地形并继续适配 |
| 不计入论文阶段 | `go2_stage1*` | 早期/预训练实验，保留用于追溯，不作为论文阶段一 |

新实验建议使用 `go2_stage3` 作为统一 NSR 适配任务；`go2_stage3a/b` 保留用于复现旧拆分实验。

## 目录结构

```text
LeggedGym-Ex/
├── legged_gym/              # 仿真环境、Go2 任务、训练/评估入口
│   ├── envs/go2/            # Go2 基础、stage2、stage3、blind baseline
│   └── scripts/             # train/play/evaluate 以及链式训练脚本
├── rsl_rl/                  # PPO、TS、DreamWaQ 风格 runner 和网络
├── nsr_height/              # 神经地形表征模型、数据集、训练、可视化
├── experiment/              # 论文实验统计、离线验证、绘图
├── resources/               # Go2 URDF/XML/mesh 和基础地形资源
├── results/                 # 统一结果目录
└── tests/                   # 任务注册和轻量测试
```

## 环境

论文策略训练主要使用 Isaac Gym。推荐使用 Python 3.8 环境：

```bash
conda create -n leggedGym python=3.8
conda activate leggedGym
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[isaacgym]"
export SIMULATOR=isaacgym
```

如果是已有环境，运行测试或训练时遇到 `Ninja is required to load C++ extensions`，执行 `pip install ninja` 后再重试。

Genesis/IsaacLab 相关适配代码仍保留，但本整理版的论文结果以 Isaac Gym 为主。

## NSR 训练

NSR 数据集需提供 `train/` 和 `val/` 划分，内部为 `traj_*` 或 `env_*/traj_*` 序列，每帧 `.npz` 至少包含相机点云、机器人位姿和 GT 地形/高度图信息。

```bash
export DATA_DIR=/path/to/nsr_go2_v3/split_env_90_10

python -m nsr_height.train \
  --data_dir "$DATA_DIR" \
  --log_dir results/nsr/logs/nsr_go2_v3_main \
  --checkpoint_dir results/nsr/checkpoints/nsr_go2_v3_main \
  --batch_size 8 \
  --seq_len 24 \
  --min_seq_len 24 \
  --epochs 10 \
  --in_channels 4 \
  --base_channels 32 \
  --gravity_aligned \
  --align_prev
```

Stage3 默认读取：

```text
results/nsr/checkpoints/nsr_go2_v3_edgeft_v1/checkpoint_best_hole_mae.pth
```

如使用其他 checkpoint，请修改 `legged_gym/envs/go2/go2_stage3/go2_stage3_config.py` 中的 `cfg.nsr.ckpt`。

## 策略训练

单独训练：

```bash
export SIMULATOR=isaacgym

python -m legged_gym.scripts.train --task go2_stage2a --headless
python -m legged_gym.scripts.train --task go2_stage2b --resume --load_run results/training_logs/go2_stage2/<stage2a_run> --ckpt -1 --headless
python -m legged_gym.scripts.train --task go2_stage2c --resume --load_run results/training_logs/go2_stage2/<stage2b_run> --ckpt -1 --headless
python -m legged_gym.scripts.train --task go2_stage3  --resume --load_run results/training_logs/go2_stage2/<stage2c_run> --ckpt -1 --headless
```

链式脚本：

```bash
# 论文阶段一前半：stage2a -> stage2b
bash legged_gym/scripts/train_go2_stage2_chain.sh

# 论文阶段一后半和阶段二：stage2c -> stage3
export STAGE2B_RUN=results/training_logs/go2_stage2/<stage2b_run>
bash legged_gym/scripts/train_go2_stage3_chain.sh
```

历史 stage1 链式脚本仍在：

```bash
bash legged_gym/scripts/train_go2_stage1_chain.sh
```

该脚本只用于复现早期预训练，不对应论文阶段一。

## 评估和分析

论文实验脚本：

```bash
python experiment/nsr/offline_validate.py \
  --out_dir results/experiments/nsr/full_val_edgeft_v1
python experiment/nsr/terrain_type_analysis.py
python experiment/nsr/temporal_blind_completion.py

python experiment/go2/training_process_analysis.py
python experiment/go2/five_terrain_distinctive_comparison.py --aggregate_only
python experiment/go2/nsr_five_terrain_robustness.py --aggregate_only
```

论文实验入口见 `experiment/README.md`。保留的脚本对应论文第 4 章中除 sim2sim 外的表格和图像，输出应写入 `results/experiments/`。

## 结果目录

论文相关运行结果统一放在 `results/`，不再使用仓库根目录下的 `logs/` 或分散输出：

| 路径 | 内容 |
| --- | --- |
| `results/training_logs/` | 策略训练日志、TensorBoard 事件、模型检查点 |
| `results/nsr/checkpoints/` | NSR 模型检查点 |
| `results/experiments/` | 论文实验表格、图像、统计结果 |
| `results/chain_logs/` | 链式训练脚本运行后生成的日志 |

## 常用检查

```bash
python tests/test_all_tasks.py --list
python tests/test_all_tasks.py --tasks go2_stage2a go2_stage3 go2_blind --iterations 1
python -m compileall nsr_height legged_gym/envs/go2 legged_gym/scripts/evaluate_stage3.py
```

`test_all_tasks.py` 会真正创建仿真环境，需 Isaac Gym/显卡环境可用；`compileall` 只做语法级检查。
