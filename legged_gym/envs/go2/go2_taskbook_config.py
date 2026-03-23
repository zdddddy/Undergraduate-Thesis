import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.camera_profiles import GO2_D435I_4CAM_PROFILE
from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2TaskbookCfg(GO2RoughCfg):
    class terrain(GO2RoughCfg.terrain):
        curriculum = True
        min_init_terrain_level = 1
        max_init_terrain_level = 4
        curriculum_move_up_dist = 3.2
        curriculum_move_down_dist = 1.2
        curriculum_recycle_min_level = 5
        num_rows = 10
        num_cols = 20
        # [smooth slope, rough slope, stairs up, stairs down, obstacles]
        terrain_proportions = [0.1, 0.1, 0.2, 0.2, 0.4]
        # GO2-specific terrain caps: keep top stair height in a deployable range.
        stairs_step_width_min = 0.24
        stairs_step_width_max = 0.45
        stairs_step_height_min = 0.04
        stairs_step_height_max = 0.17
        stairs_curriculum_scale = True
        obstacle_height_min = 0.06
        obstacle_height_max = 0.18

    class control(GO2RoughCfg.control):
        # Keep explicit for auditability during tuning.
        action_scale = 0.25

    class commands(GO2RoughCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        resampling_time = 8.0

        class ranges(GO2RoughCfg.commands.ranges):
            lin_vel_x = [0.2, 0.8]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.7, 0.7]
            heading = [-3.14, 3.14]

    class rewards(GO2RoughCfg.rewards):
        base_height_target = 0.32
        max_contact_force = 300.0
        only_positive_rewards = True
        tracking_sigma = 0.35

        class scales(GO2RoughCfg.rewards.scales):
            # task objective
            tracking_lin_vel = 1.8
            tracking_ang_vel = 0.8
            foot_clearance = 1.2
            feet_air_time = 0.0

            # safety
            collision = -3.0
            feet_stumble = -1.5
            termination = -4.0
            feet_contact_forces = -0.0015

            # stability + low energy
            orientation = -0.6
            lin_vel_z = -2.2
            ang_vel_xy = -0.2
            base_height = -0.45
            torques = -3.0e-5
            dof_vel = -1.5e-4
            dof_acc = -1.2e-6
            dof_pos_limits = -1.0
            action_rate = -0.03
            stand_still = -0.0

    class noise(GO2RoughCfg.noise):
        add_noise = True
        noise_level = 1.0
        height_measurements_noise_m_min = 0.02
        height_measurements_noise_m_max = 0.05

        class noise_scales(GO2RoughCfg.noise.noise_scales):
            height_measurements = 0.0

    class domain_rand(GO2RoughCfg.domain_rand):
        push_robots = True
        push_interval_s = 12
        max_push_vel_xy = 0.6


class GO2TaskbookCfgPPO(GO2RoughCfgPPO):
    class algorithm(GO2RoughCfgPPO.algorithm):
        learning_rate = 5.0e-4
        entropy_coef = 0.005

    class runner(GO2RoughCfgPPO.runner):
        experiment_name = "rough_go2_taskbook"
        run_name = ""
        max_iterations = 5000
        save_interval = 100


class GO2CollectCfg(GO2TaskbookCfg):
    """Collector policy: same terrain mix as taskbook, but easier observations."""

    class terrain(GO2TaskbookCfg.terrain):
        # Force starts near hard rows to break the "safe-but-stuck" local optimum.
        min_init_terrain_level = 6
        max_init_terrain_level = 8
        # Hard-row curriculum tuning: easier to level up, harder to be demoted.
        curriculum_move_up_dist = 3.2
        curriculum_move_down_dist = 1.0
        curriculum_recycle_min_level = 6

    class control(GO2TaskbookCfg.control):
        # Increase command-to-joint range to allow larger step height on taller stairs.
        action_scale = 0.35

    class commands(GO2TaskbookCfg.commands):
        # Keep command range fixed; command curriculum can re-introduce easy/slow commands.
        curriculum = False
        class ranges(GO2TaskbookCfg.commands.ranges):
            # Raise commanded forward speed so episodes are more likely to cross terrain blocks.
            lin_vel_x = [0.6, 1.2]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.3, 0.3]

    class rewards(GO2TaskbookCfg.rewards):
        class scales(GO2TaskbookCfg.rewards.scales):
            # Collector tuning: prioritize robust traversal + viewpoint diversity.
            tracking_lin_vel = 1.2
            foot_clearance = 1.7
            collision = -0.8
            feet_stumble = -0.4
            orientation = 0.0
            dof_pos_limits = -0.5
            dof_acc = -5.0e-7
            action_rate = -0.012

    class env(GO2TaskbookCfg.env):
        # Give slightly more time for hard-row traversal before timeout reset.
        episode_length_s = 24

    class noise(GO2TaskbookCfg.noise):
        add_noise = False
        height_measurements_noise_m_min = 0.0
        height_measurements_noise_m_max = 0.0

        class noise_scales(GO2TaskbookCfg.noise.noise_scales):
            height_measurements = 0.0

    class domain_rand(GO2TaskbookCfg.domain_rand):
        # Mild perturbations improve viewpoint diversity without dominating failures.
        push_robots = True
        push_interval_s = 12
        max_push_vel_xy = 0.35


class GO2CollectCfgPPO(GO2TaskbookCfgPPO):
    class algorithm(GO2TaskbookCfgPPO.algorithm):
        # Keep exploration alive while policy adapts to hard-start curriculum.
        entropy_coef = 0.015

    class runner(GO2TaskbookCfgPPO.runner):
        experiment_name = "rough_go2_collect"
        run_name = ""
        max_iterations = 3000
        save_interval = 100


class GO2TaskbookNSRCfg(GO2TaskbookCfg):
    class env(GO2TaskbookCfg.env):
        use_warp = True
        # NSR runtime + 4-camera pointcloud buffers are memory-heavy on 8GB GPUs.
        # Keep a conservative default and scale up via --num_envs if headroom allows.
        num_envs = 1024

    class sensor(GO2TaskbookCfg.sensor):
        add_depth = True
        num_sensors = GO2_D435I_4CAM_PROFILE["num_sensors"]

        class depth_camera_config(GO2TaskbookCfg.sensor.depth_camera_config):
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

    class noise(GO2TaskbookCfg.noise):
        add_noise = False
        height_measurements_noise_m_min = 0.0
        height_measurements_noise_m_max = 0.0

        class noise_scales(GO2TaskbookCfg.noise.noise_scales):
            height_measurements = 0.0

    class nsr(GO2TaskbookCfg.nsr):
        enable = True
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


class GO2TaskbookNSRCfgPPO(GO2TaskbookCfgPPO):
    class runner(GO2TaskbookCfgPPO.runner):
        experiment_name = "rough_go2_taskbook_nsr"
        run_name = ""
        max_iterations = 2000
