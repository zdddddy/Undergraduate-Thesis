import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.camera_profiles import GO2_D435I_4CAM_PROFILE
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2Stage1GTBaseCfg(GO2RoughCfg):
    """Shared stage-1 GT terrain-observation setup."""

    class env(GO2RoughCfg.env):
        # base(48) + local height samples(17*11=187)
        num_observations = 235
        use_warp = False
        num_envs = 1024

    class terrain(GO2RoughCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        min_init_terrain_level = 0
        max_init_terrain_level = 0
        curriculum_move_up_dist = 1.2
        curriculum_move_down_dist = 0.2
        curriculum_recycle_min_level = 0
        num_rows = 10
        num_cols = 20
        # [smooth slope, rough slope, stairs up, stairs down, obstacles]
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]

    class control(GO2RoughCfg.control):
        action_scale = 0.25

    class commands(GO2RoughCfg.commands):
        curriculum = False
        num_commands = 4
        heading_command = True
        resampling_time = 10.0

        class ranges(GO2RoughCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class rewards(GO2RoughCfg.rewards):
        base_height_target = 0.25
        max_contact_force = 100.0
        tracking_sigma = 0.25
        # Penalize "commanded to move but barely moving".
        cmd_nonzero_thresh = 0.2
        slow_speed_thresh = 0.2
        feet_air_time_target = 0.5
        foot_clearance_cap = 0.25

        class scales(GO2RoughCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            foot_clearance = 0.0
            feet_air_time = 1.0
            pronk = 0.0
            non_alternating = 0.0

            collision = -1.0
            feet_stumble = 0.0
            termination = 0.0
            feet_contact_forces = 0.0
            orientation = 0.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            base_height = 0.0

            torques = -2.0e-4
            dof_vel = 0.0
            dof_acc = -2.5e-7
            dof_pos_limits = -10.0
            action_rate = -0.01
            stand_still = 0.0
            cmd_nonzero_but_slow = 0.0

    class sensor(GO2RoughCfg.sensor):
        add_depth = False

    class normalization(GO2RoughCfg.normalization):
        clip_actions = 1.0

    class domain_rand(GO2RoughCfg.domain_rand):
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.25

    class nsr(GO2RoughCfg.nsr):
        enable = False
        ckpt = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "nsr_height",
            "checkpoints",
            "nsr_go2_v3_edgeft_v1",
            "checkpoint_best_hole_mae.pth",
        )
        max_valid_depth = 50.0
        gravity_aligned = True
        align_prev = True
        disable_prev = False
        prev_valid_threshold = 0.5
        memory_meas_override = True


class GO2Stage1GTBaseCfgPPO(GO2RoughCfgPPO):
    class policy(GO2RoughCfgPPO.policy):
        init_noise_std = 1.0

    class algorithm(GO2RoughCfgPPO.algorithm):
        schedule = "fixed"
        learning_rate = 1.0e-3
        entropy_coef = 0.01
        num_learning_epochs = 5
        desired_kl = 0.02
        num_mini_batches = 4

    class runner(GO2RoughCfgPPO.runner):
        max_iterations = 2500
        save_interval = 100
        num_steps_per_env = 24
        resume_partial = False
        resume_load_optimizer = False
        resume_reset_iter = True
        resume_override_action_std = None


class GO2Stage1GTCleanCfg(GO2Stage1GTBaseCfg):
    """Stage-1 GT baseline without observation noise."""

    class rewards(GO2Stage1GTBaseCfg.rewards):
        # Align with official reward clipping behavior.
        only_positive_rewards = True

        class scales(GO2Stage1GTBaseCfg.rewards.scales):
            # Keep stage-1 shaping but avoid "head-down crouch" local minima.
            feet_air_time = 0.30
            orientation = -1.00
            base_height = -1.00
            termination = -2.00
            cmd_nonzero_but_slow = -0.50

    class noise(GO2Stage1GTBaseCfg.noise):
        add_noise = False
        height_measurements_noise_m_min = 0.0
        height_measurements_noise_m_max = 0.0

        class noise_scales(GO2Stage1GTBaseCfg.noise.noise_scales):
            height_measurements = 0.0


class GO2Stage1GTCleanCfgPPO(GO2Stage1GTBaseCfgPPO):
    class runner(GO2Stage1GTBaseCfgPPO.runner):
        experiment_name = "rough_go2_stage1_gt_clean"
        run_name = "stage1_gt_clean_move_v1"
        max_iterations = 2000


class GO2Stage2GTNoiseCfg(GO2Stage1GTBaseCfg):
    """Stage-2 GT+noise baseline for NSR transition."""

    class terrain(GO2Stage1GTBaseCfg.terrain):
        mesh_type = "trimesh"
        max_init_terrain_level = 1
        curriculum_recycle_min_level = 1
        terrain_proportions = [0.72, 0.20, 0.0, 0.0, 0.08]

    class noise(GO2Stage1GTBaseCfg.noise):
        add_noise = True
        noise_level = 1.0
        height_measurements_noise_m_min = 0.0
        height_measurements_noise_m_max = 0.0

        class noise_scales(GO2Stage1GTBaseCfg.noise.noise_scales):
            height_measurements = 0.0

    class rewards(GO2Stage1GTBaseCfg.rewards):
        # Stage-2: keep locomotion shape from stage-1, but reduce anti-stall pressure
        # and allow slightly more terrain-aware swing behavior.
        class scales(GO2Stage1GTBaseCfg.rewards.scales):
            foot_clearance = 0.30
            feet_air_time = 0.12
            pronk = -1.60
            non_alternating = -0.60
            action_rate = -0.016
            cmd_nonzero_but_slow = -0.50


class GO2Stage2GTNoiseCfgPPO(GO2Stage1GTBaseCfgPPO):
    class runner(GO2Stage1GTBaseCfgPPO.runner):
        experiment_name = "rough_go2_stage2_gt_noise"
        run_name = "stage2_gt_noise_move_v1"
        max_iterations = 2500


class GO2Stage3NSRAdaptCfg(GO2Stage2GTNoiseCfg):
    """Stage-3 NSR adaptation on top of stage-2 GT+noise behavior."""

    class env(GO2Stage2GTNoiseCfg.env):
        use_warp = True
        num_envs = 512

    class sensor(GO2Stage2GTNoiseCfg.sensor):
        add_depth = True
        num_sensors = GO2_D435I_4CAM_PROFILE["num_sensors"]

        class depth_camera_config(GO2Stage2GTNoiseCfg.sensor.depth_camera_config):
            num_history = 1
            near_clip = GO2_D435I_4CAM_PROFILE["near_clip"]
            far_clip = GO2_D435I_4CAM_PROFILE["far_clip"]
            near_plane = GO2_D435I_4CAM_PROFILE["near_plane"]
            far_plane = GO2_D435I_4CAM_PROFILE["far_plane"]
            resolution = GO2_D435I_4CAM_PROFILE["resolution"]
            horizontal_fov_deg = GO2_D435I_4CAM_PROFILE["horizontal_fov_deg"]
            decimation = GO2_D435I_4CAM_PROFILE["decimation"]
            calculate_depth = GO2_D435I_4CAM_PROFILE["calculate_depth"]
            return_pointcloud = GO2_D435I_4CAM_PROFILE["return_pointcloud"]
            pointcloud_in_world_frame = GO2_D435I_4CAM_PROFILE["pointcloud_in_world_frame"]
            euler = GO2_D435I_4CAM_PROFILE["euler"]
            pos = GO2_D435I_4CAM_PROFILE["pos"]

    class nsr(GO2Stage2GTNoiseCfg.nsr):
        enable = True

    class rewards(GO2Stage2GTNoiseCfg.rewards):
        # Stage-3: keep motion alive while avoiding over-constraining NSR adaptation.
        class scales(GO2Stage2GTNoiseCfg.rewards.scales):
            foot_clearance = 0.35
            feet_air_time = 0.10
            pronk = -1.40
            non_alternating = -0.45
            action_rate = -0.015
            cmd_nonzero_but_slow = -0.45


class GO2Stage3NSRAdaptCfgPPO(GO2Stage2GTNoiseCfgPPO):
    class policy(GO2Stage2GTNoiseCfgPPO.policy):
        init_noise_std = 0.35

    class algorithm(GO2Stage2GTNoiseCfgPPO.algorithm):
        schedule = "fixed"
        learning_rate = 7.0e-5
        entropy_coef = 0.004
        num_learning_epochs = 4
        desired_kl = 0.025
        num_mini_batches = 8

    class runner(GO2Stage2GTNoiseCfgPPO.runner):
        experiment_name = "rough_go2_stage3_nsr_adapt"
        run_name = "stage3_nsr_adapt_v1"
        max_iterations = 1500
