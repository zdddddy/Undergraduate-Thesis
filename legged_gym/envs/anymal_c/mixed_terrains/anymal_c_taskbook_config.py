# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.anymal_c.mixed_terrains.anymal_c_rough_config import (
    AnymalCRoughCfg,
    AnymalCRoughCfgPPO,
)


class AnymalCTaskbookCfg(AnymalCRoughCfg):
    """Taskbook-oriented pure RL setup:
    - ground-truth terrain heights as observation
    - explicit height-noise range to mimic NSR uncertainty
    - stronger safety/energy/stability shaping
    - curriculum-enabled mixed complex terrains
    """

    class terrain(AnymalCRoughCfg.terrain):
        # Match NSR dataset collection terrain setup (collect_large.py --task rough_anymal_c):
        # [smooth slope, rough slope, stairs up, stairs down, obstacles]
        curriculum = True
        max_init_terrain_level = 5
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.1, 0.1, 0.2, 0.2, 0.4]

    class commands(AnymalCRoughCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        resampling_time = 8.0

        class ranges(AnymalCRoughCfg.commands.ranges):
            lin_vel_x = [0.2, 0.8]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.7, 0.7]
            heading = [-3.14, 3.14]

    class rewards(AnymalCRoughCfg.rewards):
        base_height_target = 0.5
        max_contact_force = 450.0
        only_positive_rewards = True
        tracking_sigma = 0.35

        class scales(AnymalCRoughCfg.rewards.scales):
            # task objective
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.8
            foot_clearance = 0.8
            feet_air_time = 0.0

            # safety
            collision = -2.5
            feet_stumble = -1.5
            termination = -2.0
            feet_contact_forces = -0.001

            # stability + low energy
            orientation = -1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.15
            base_height = -0.4
            torques = -2.0e-5
            dof_vel = -1.0e-4
            dof_acc = -8.0e-7
            action_rate = -0.02
            stand_still = -0.0

    class noise(AnymalCRoughCfg.noise):
        add_noise = True
        noise_level = 1.0
        # Explicit terrain-height noise in meters: U([-a,+a]), a~U(0.02, 0.05)
        height_measurements_noise_m_min = 0.02
        height_measurements_noise_m_max = 0.05

        class noise_scales(AnymalCRoughCfg.noise.noise_scales):
            # Disable legacy fixed-amplitude height noise to avoid double counting.
            height_measurements = 0.0

    class domain_rand(AnymalCRoughCfg.domain_rand):
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0


class AnymalCTaskbookCfgPPO(AnymalCRoughCfgPPO):
    class algorithm(AnymalCRoughCfgPPO.algorithm):
        learning_rate = 5.0e-4
        entropy_coef = 0.005

    class runner(AnymalCRoughCfgPPO.runner):
        experiment_name = "rough_anymal_c_taskbook"
        run_name = ""
        max_iterations = 5000
        save_interval = 100


class AnymalCTaskbookNSRCfg(AnymalCTaskbookCfg):
    """Taskbook policy with online NSR terrain completion as height observation source."""

    class env(AnymalCTaskbookCfg.env):
        use_warp = True

    class sensor(AnymalCTaskbookCfg.sensor):
        add_depth = True
        num_sensors = 4

        class depth_camera_config(AnymalCTaskbookCfg.sensor.depth_camera_config):
            num_history = 1
            near_clip = 0.1
            far_clip = 50.0
            near_plane = 0.1
            far_plane = 50.0
            resolution = (80, 60)
            horizontal_fov_deg = 75
            decimation = 1
            calculate_depth = False
            return_pointcloud = True
            pointcloud_in_world_frame = True
            euler = [
                (0.0, 2.09, 0.0),
                (0.0, 2.09, 3.1416),
                (0.0, 2.09, 1.5708),
                (0.0, 2.09, -1.5708),
            ]
            pos = [
                (0.3, 0.0, 0.1),
                (-0.3, 0.0, 0.1),
                (0.0, 0.3, 0.1),
                (0.0, -0.3, 0.1),
            ]

    class noise(AnymalCTaskbookCfg.noise):
        add_noise = False
        height_measurements_noise_m_min = 0.0
        height_measurements_noise_m_max = 0.0

        class noise_scales(AnymalCTaskbookCfg.noise.noise_scales):
            height_measurements = 0.0

    class nsr(AnymalCTaskbookCfg.nsr):
        enable = True
        ckpt = os.path.join(LEGGED_GYM_ROOT_DIR, "nsr_height", "checkpoints", "taskbook_candidate_v1", "model.pth")
        max_valid_depth = 50.0
        gravity_aligned = True
        align_prev = True
        disable_prev = False
        prev_valid_threshold = 0.5
        memory_meas_override = True


class AnymalCTaskbookNSRCfgPPO(AnymalCTaskbookCfgPPO):
    class runner(AnymalCTaskbookCfgPPO.runner):
        experiment_name = "rough_anymal_c_taskbook_nsr"
        run_name = ""
        max_iterations = 2000
