# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base.legged_robot import LeggedRobot
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .anymal_c.anymal import Anymal
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.mixed_terrains.anymal_c_taskbook_config import (
    AnymalCTaskbookCfg,
    AnymalCTaskbookCfgPPO,
    AnymalCTaskbookNSRCfg,
    AnymalCTaskbookNSRCfgPPO,
)
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from .go2.go2_stage1_config import (
    GO2Stage1BlindFlatCfg,
    GO2Stage1BlindFlatCfgPPO,
    GO2Stage1BlindHardenedCfg,
    GO2Stage1BlindHardenedCfgPPO,
    GO2Stage1BlindCfg,
    GO2Stage1BlindCfgPPO,
    GO2Stage1GTCleanCfg,
    GO2Stage1GTCleanCfgPPO,
)
from .go2.go2_taskbook_config import (
    GO2CollectCfg,
    GO2CollectCfgPPO,
    GO2TaskbookCfg,
    GO2TaskbookCfgPPO,
    GO2TaskbookNSRCfg,
    GO2TaskbookNSRCfgPPO,
)

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "rough_anymal_c", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_taskbook", Anymal, AnymalCTaskbookCfg(), AnymalCTaskbookCfgPPO() )
task_registry.register( "rough_anymal_c_taskbook", Anymal, AnymalCTaskbookCfg(), AnymalCTaskbookCfgPPO() )
task_registry.register( "anymal_c_taskbook_nsr", Anymal, AnymalCTaskbookNSRCfg(), AnymalCTaskbookNSRCfgPPO() )
task_registry.register( "rough_anymal_c_taskbook_nsr", Anymal, AnymalCTaskbookNSRCfg(), AnymalCTaskbookNSRCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO() )
task_registry.register( "rough_go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO() )
task_registry.register( "go2_stage1_blind_flat", LeggedRobot, GO2Stage1BlindFlatCfg(), GO2Stage1BlindFlatCfgPPO() )
task_registry.register( "rough_go2_stage1_blind_flat", LeggedRobot, GO2Stage1BlindFlatCfg(), GO2Stage1BlindFlatCfgPPO() )
task_registry.register( "go2_stage1_blind_hardened", LeggedRobot, GO2Stage1BlindHardenedCfg(), GO2Stage1BlindHardenedCfgPPO() )
task_registry.register( "rough_go2_stage1_blind_hardened", LeggedRobot, GO2Stage1BlindHardenedCfg(), GO2Stage1BlindHardenedCfgPPO() )
task_registry.register( "go2_stage1_blind", LeggedRobot, GO2Stage1BlindCfg(), GO2Stage1BlindCfgPPO() )
task_registry.register( "rough_go2_stage1_blind", LeggedRobot, GO2Stage1BlindCfg(), GO2Stage1BlindCfgPPO() )
task_registry.register( "go2_stage1_gt_clean", LeggedRobot, GO2Stage1GTCleanCfg(), GO2Stage1GTCleanCfgPPO() )
task_registry.register( "rough_go2_stage1_gt_clean", LeggedRobot, GO2Stage1GTCleanCfg(), GO2Stage1GTCleanCfgPPO() )
task_registry.register( "go2_collect", LeggedRobot, GO2CollectCfg(), GO2CollectCfgPPO() )
task_registry.register( "rough_go2_collect", LeggedRobot, GO2CollectCfg(), GO2CollectCfgPPO() )
task_registry.register( "go2_taskbook", LeggedRobot, GO2TaskbookCfg(), GO2TaskbookCfgPPO() )
task_registry.register( "rough_go2_taskbook", LeggedRobot, GO2TaskbookCfg(), GO2TaskbookCfgPPO() )
task_registry.register( "go2_taskbook_nsr", LeggedRobot, GO2TaskbookNSRCfg(), GO2TaskbookNSRCfgPPO() )
task_registry.register( "rough_go2_taskbook_nsr", LeggedRobot, GO2TaskbookNSRCfg(), GO2TaskbookNSRCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
