# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import *

from .base.legged_robot import LeggedRobot
from .base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO
from legged_gym.envs.go2.go2_stage1.go2_stage1 import GO2Stage1
from legged_gym.envs.go2.go2_stage1.go2_stage1_config import (
    GO2Stage1ACfg,
    GO2Stage1ACfgPPO,
    GO2Stage1BCfg,
    GO2Stage1BCfgPPO,
)
from legged_gym.envs.go2.go2_stage2.go2_stage2 import Go2Stage2
from legged_gym.envs.go2.go2_stage2.go2_stage2_config import (
    Go2Stage2ACfg,
    Go2Stage2ACfgPPO,
    Go2Stage2BCfg,
    Go2Stage2BCfgPPO,
    Go2Stage2CCfg,
    Go2Stage2CCfgPPO,
)
from legged_gym.envs.go2.go2_blind.go2_blind import GO2Blind
from legged_gym.envs.go2.go2_blind.go2_blind_config import GO2BlindCfg, GO2BlindCfgPPO
from legged_gym.envs.go2.go2_stage3.go2_stage3 import Go2Stage3
from legged_gym.envs.go2.go2_stage3.go2_stage3_config import (
    Go2Stage3ACfg,
    Go2Stage3ACfgPPO,
    Go2Stage3BCfg,
    Go2Stage3BCfgPPO,
    Go2Stage3Cfg,
    Go2Stage3CfgPPO,
)
from legged_gym.utils.task_registry import task_registry


task_registry.register("go2", GO2, GO2Cfg(), GO2CfgPPO())

# Stage1 is retained for historical/pretraining experiments. In the thesis
# narrative, the obstacle-crossing stages start from code stage2.
task_registry.register("go2_stage1", GO2Stage1, GO2Stage1ACfg(), GO2Stage1ACfgPPO())
task_registry.register("go2_stage1a", GO2Stage1, GO2Stage1ACfg(), GO2Stage1ACfgPPO())
task_registry.register("go2_stage1b", GO2Stage1, GO2Stage1BCfg(), GO2Stage1BCfgPPO())
task_registry.register("go2_stage1_1a", GO2Stage1, GO2Stage1ACfg(), GO2Stage1ACfgPPO())
task_registry.register("go2_stage1_1b", GO2Stage1, GO2Stage1BCfg(), GO2Stage1BCfgPPO())

task_registry.register("go2_stage2", Go2Stage2, Go2Stage2ACfg(), Go2Stage2ACfgPPO())
task_registry.register("go2_stage2a", Go2Stage2, Go2Stage2ACfg(), Go2Stage2ACfgPPO())
task_registry.register("go2_stage2b", Go2Stage2, Go2Stage2BCfg(), Go2Stage2BCfgPPO())
task_registry.register("go2_stage2c", Go2Stage2, Go2Stage2CCfg(), Go2Stage2CCfgPPO())

task_registry.register("go2_blind", GO2Blind, GO2BlindCfg(), GO2BlindCfgPPO())

task_registry.register("go2_stage3", Go2Stage3, Go2Stage3Cfg(), Go2Stage3CfgPPO())
task_registry.register("go2_stage3a", Go2Stage3, Go2Stage3ACfg(), Go2Stage3ACfgPPO())
task_registry.register("go2_stage3b", Go2Stage3, Go2Stage3BCfg(), Go2Stage3BCfgPPO())
