from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import Go2FlatCommonCfg, get_simulator_suffix


class GO2Stage1BaseCfg(Go2FlatCommonCfg):
    class env(Go2FlatCommonCfg.env):
        num_envs = 4096
        num_observations = 45  # deployment-style proprioception
        num_privileged_obs = None
        num_actions = 12

    class terrain(Go2FlatCommonCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(Go2FlatCommonCfg.commands):
        curriculum = True
        max_curriculum = 1.2
        num_commands = 4
        resampling_time = 8.0
        heading_command = True
        zero_cmd_prob = 0.25

        class ranges(Go2FlatCommonCfg.commands.ranges):
            lin_vel_x = [-0.6, 0.6]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class domain_rand(Go2FlatCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.4]
        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]
        push_robots = True
        push_interval_s = 20
        max_push_vel_xy = 0.6
        randomize_com_displacement = True
        com_pos_x_range = [-0.015, 0.015]
        com_pos_y_range = [-0.015, 0.015]
        com_pos_z_range = [-0.015, 0.015]
        randomize_pd_gain = False

    class rewards(Go2FlatCommonCfg.rewards):
        soft_dof_pos_limit = 0.9
        tracking_sigma = 0.25

        # Targets/trackers used by stage1 priors.
        base_height_target = 0.32
        base_height_tracking_sigma = 0.01
        foot_clearance_target = 0.06
        foot_height_offset = 0.022
        foot_clearance_tracking_sigma = 0.01
        euler_tracking_sigma = 0.1
        about_landing_threshold = 0.03
        only_positive_rewards = True

        class periodic_reward_framework:
            b_swing = 0.5
            # Start with trot only in 1a (single gait entry).
            theta_fl_list = [0.0]
            theta_fr_list = [0.5]
            theta_rl_list = [0.5]
            theta_rr_list = [0.0]

        class behavior_params_range:
            resampling_time = 6.0
            gait_period_range = [0.48, 0.56]
            foot_clearance_target_range = [0.05, 0.07]
            base_height_target_range = [0.30, 0.34]
            pitch_target_range = [-0.05, 0.05]

        class scales(Go2FlatCommonCfg.rewards.scales):
            # limitation
            dof_pos_limits = -1.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.6
            # smoothness / regularization
            lin_vel_z = -0.6
            base_height = -1.0
            ang_vel_xy = -0.1
            orientation = -1.0
            dof_vel = -5.0e-4
            dof_acc = -2.0e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.0e-4
            feet_slip = -0.05
            dof_vel_stand_still = -0.02
            dof_pos_stand_still = -0.10
            # gait / morphology priors
            feet_air_time = 1.0
            foot_clearance = 0.4
            hip_pos = -0.3
            thigh_pos = -0.2
            # WTW-inspired priors (enabled in 1b)
            tracking_base_height = 0.0
            tracking_orientation = 0.0
            tracking_foot_clearance = 0.0
            quad_periodic_gait = 0.0
            foot_landing_vel = 0.0


class GO2Stage1ACfg(GO2Stage1BaseCfg):
    pass


class GO2Stage1BCfg(GO2Stage1BaseCfg):
    class commands(GO2Stage1BaseCfg.commands):
        class ranges(GO2Stage1BaseCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1.2, 1.2]
            heading = [-3.14, 3.14]

    class domain_rand(GO2Stage1BaseCfg.domain_rand):
        friction_range = [0.4, 1.6]
        added_mass_range = [-0.8, 0.8]
        push_interval_s = 12
        max_push_vel_xy = 0.9
        com_pos_x_range = [-0.025, 0.025]
        com_pos_y_range = [-0.025, 0.025]
        com_pos_z_range = [-0.02, 0.02]
        randomize_pd_gain = True
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]

    class rewards(GO2Stage1BaseCfg.rewards):
        class periodic_reward_framework(GO2Stage1BaseCfg.rewards.periodic_reward_framework):
            # trot, pronk, pace, bound
            theta_fl_list = [0.0, 0.0, 0.5, 0.0]
            theta_fr_list = [0.5, 0.0, 0.0, 0.0]
            theta_rl_list = [0.5, 0.0, 0.5, 0.5]
            theta_rr_list = [0.0, 0.0, 0.0, 0.5]

        class behavior_params_range(GO2Stage1BaseCfg.rewards.behavior_params_range):
            resampling_time = 4.0
            gait_period_range = [0.35, 0.65]
            foot_clearance_target_range = [0.05, 0.12]
            base_height_target_range = [0.26, 0.34]
            pitch_target_range = [-0.20, 0.20]

        class scales(GO2Stage1BaseCfg.rewards.scales):
            # keep velocity tracking while increasing prior constraints
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5

            lin_vel_z = -0.5
            base_height = -0.3
            ang_vel_xy = -0.08
            orientation = -0.3
            feet_slip = -0.10

            feet_air_time = 0.8
            foot_clearance = 0.2
            hip_pos = -0.8
            thigh_pos = -0.4

            # WTW-inspired constraints
            tracking_base_height = 0.6
            tracking_orientation = 0.7
            tracking_foot_clearance = 0.9
            quad_periodic_gait = 1.5
            foot_landing_vel = -0.10


class GO2Stage1ACfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = "stage1_1a" + get_simulator_suffix()
        experiment_name = "go2_stage1"
        save_interval = 200
        max_iterations = 2000


class GO2Stage1BCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 5.0e-4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "stage1_1b" + get_simulator_suffix()
        experiment_name = "go2_stage1"
        save_interval = 200
        max_iterations = 2000
