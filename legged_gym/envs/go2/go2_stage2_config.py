from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2Stage2GTTeacherCfg(GO2RoughCfg):
    """Stage-2(a): terrain-aware teacher with GT local height observations."""

    class env(GO2RoughCfg.env):
        # base(48) + local height samples(17*11=187)
        num_observations = 235
        episode_length_s = 24

    class terrain(GO2RoughCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        height_measurements_center = 0.30
        curriculum = True
        min_init_terrain_level = 0
        max_init_terrain_level = 3
        curriculum_move_up_dist = 2.8
        curriculum_move_down_dist = 1.0
        curriculum_recycle_min_level = 5
        num_rows = 10
        num_cols = 20
        # [smooth slope, rough slope, stairs up, stairs down, obstacles]
        terrain_proportions = [0.10, 0.10, 0.25, 0.25, 0.30]
        # Align stair/obstacle geometry caps with GO2 collect profile.
        stairs_step_width_min = 0.24
        stairs_step_width_max = 0.45
        stairs_step_height_min = 0.04
        stairs_step_height_max = 0.17
        stairs_curriculum_scale = True
        obstacle_height_min = 0.06
        obstacle_height_max = 0.17

    class commands(GO2RoughCfg.commands):
        curriculum = True
        max_curriculum = 1.8
        resampling_time = 8.0

        class ranges(GO2RoughCfg.commands.ranges):
            lin_vel_x = [0.2, 1.0]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.9, 0.9]
            heading = [-3.14, 3.14]

    class rewards(GO2RoughCfg.rewards):
        base_height_target = 0.30
        base_height_use_terrain = True
        max_contact_force = 320.0
        only_positive_rewards = True
        tracking_sigma = 0.30
        feet_air_time_target = 0.40
        foot_clearance_cap = 0.30

        class scales(GO2RoughCfg.rewards.scales):
            # task objective
            tracking_lin_vel = 1.8
            tracking_ang_vel = 0.9
            feet_air_time = 0.0
            foot_clearance = 1.1

            # safety + traversal stability
            collision = -3.0
            feet_stumble = -1.5
            feet_contact_forces = -1.2e-3
            termination = -4.0
            orientation = -1.0
            lin_vel_z = -2.2
            ang_vel_xy = -0.25
            base_height = -1.2

            # regularization
            torques = -5.0e-5
            dof_vel = -1.2e-4
            dof_acc = -1.0e-6
            dof_pos_limits = -1.0
            action_rate = -0.04
            cmd_nonzero_but_slow = -0.8
            pronk = -0.2
            non_alternating = -0.5
            stand_still = -0.0

    class noise(GO2RoughCfg.noise):
        add_noise = True
        noise_level = 1.0
        # Stage-2(a): clean GT teacher.
        height_measurements_noise_m_min = 0.0
        height_measurements_noise_m_max = 0.0

        class noise_scales(GO2RoughCfg.noise.noise_scales):
            # Keep generic i.i.d. noise off for height channel in clean teacher.
            height_measurements = 0.0

    class domain_rand(GO2RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.4]
        randomize_base_mass = True
        added_mass_range = [-0.8, 1.2]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.7


class GO2Stage2GTTeacherCfgPPO(GO2RoughCfgPPO):
    class algorithm(GO2RoughCfgPPO.algorithm):
        learning_rate = 5.0e-4
        entropy_coef = 0.006

    class runner(GO2RoughCfgPPO.runner):
        experiment_name = "go2_stage2_gt_teacher"
        run_name = ""
        max_iterations = 3000
        save_interval = 100
        # Stage1(48-dim obs) -> Stage2(235-dim obs) warm-start:
        # keep overlap weights, reset optimizer/iteration as a fresh stage2 run.
        resume_partial = True
        resume_load_optimizer = False
        resume_reset_iter = True


class GO2Stage2GTTeacherHardenedCfg(GO2Stage2GTTeacherCfg):
    """Stage-2(b): GT teacher with mild map degradation + stronger randomization."""

    class terrain(GO2Stage2GTTeacherCfg.terrain):
        max_init_terrain_level = 5
        curriculum_recycle_min_level = 6
        # [smooth slope, rough slope, stairs up, stairs down, obstacles]
        terrain_proportions = [0.05, 0.10, 0.30, 0.25, 0.30]

    class rewards(GO2Stage2GTTeacherCfg.rewards):
        class scales(GO2Stage2GTTeacherCfg.rewards.scales):
            foot_clearance = 1.3
            collision = -3.5
            feet_stumble = -1.8
            feet_contact_forces = -1.5e-3
            orientation = -1.1
            base_height = -1.4
            action_rate = -0.045
            cmd_nonzero_but_slow = -1.0
            pronk = -0.25
            non_alternating = -0.6

    class noise(GO2Stage2GTTeacherCfg.noise):
        add_noise = True
        noise_level = 1.0
        # Approximate deploy-time map corruption with bounded metric perturbation.
        height_measurements_noise_m_min = 0.01
        height_measurements_noise_m_max = 0.05

        class noise_scales(GO2Stage2GTTeacherCfg.noise.noise_scales):
            # Avoid stacking additional white-noise on top of explicit meter noise above.
            height_measurements = 0.0

    class domain_rand(GO2Stage2GTTeacherCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.5]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.9


class GO2Stage2GTTeacherHardenedCfgPPO(GO2Stage2GTTeacherCfgPPO):
    class algorithm(GO2Stage2GTTeacherCfgPPO.algorithm):
        learning_rate = 3.0e-4
        entropy_coef = 0.004

    class runner(GO2Stage2GTTeacherCfgPPO.runner):
        experiment_name = "go2_stage2_gt_teacher_hardened"
        run_name = ""
        max_iterations = 5000
        save_interval = 100
        # Stage2(a)->Stage2(b) is same observation space; default to strict resume behavior.
        resume_partial = False
        resume_load_optimizer = True
        resume_reset_iter = False


# Canonical stage-2 alias.
class GO2Stage2GTCfg(GO2Stage2GTTeacherHardenedCfg):
    pass


class GO2Stage2GTCfgPPO(GO2Stage2GTTeacherHardenedCfgPPO):
    class runner(GO2Stage2GTTeacherHardenedCfgPPO.runner):
        experiment_name = "go2_stage2_gt"
