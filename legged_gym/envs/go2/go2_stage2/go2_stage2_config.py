from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotTSCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg, get_simulator_suffix


class Go2Stage2BaseCfg(Go2RoughCommonCfg):
    """代码 stage2：论文越障阶段一的共享配置。"""

    class env(Go2RoughCommonCfg.env):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 81  # 地形编码器输入的局部高度
        frame_stack = 20
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_critic_obs = c_frame_stack * num_single_critic_obs
        num_actions = 12
        env_spacing = 0.5

    class terrain(Go2RoughCommonCfg.terrain):
        curriculum = True
        measure_heights = True
        # stage2a 使用默认粗糙地形比例。
        num_rows = 10
        num_cols = 10
        terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]

    class asset(Go2RoughCommonCfg.asset):
        obtain_link_contact_states = True

    class commands(Go2RoughCommonCfg.commands):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges(Go2RoughCommonCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class domain_rand(Go2RoughCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

    class terrain_degrade:
        # 默认关闭；stage2c 再启用。
        enable = False
        curriculum_total_iters = 2000
        phase_ratios = [0.2, 0.6, 0.2]
        # [轻, 中, 重] 三档退化锚点，高度单位为 m。
        noise_std_m = [0.0015, 0.0045, 0.0080]
        point_dropout_prob = [0.02, 0.10, 0.18]
        patch_dropout_prob = [0.00, 0.06, 0.12]
        blur_prob = [0.05, 0.20, 0.35]
        bias_prob = [0.05, 0.18, 0.30]
        bias_std_m = [0.0015, 0.0040, 0.0080]
        delay_prob = [0.02, 0.10, 0.20]
        delay_mix = [0.05, 0.20, 0.35]
        # 9x9 局部网格中的块状缺失半径。
        patch_half_span_min = 1
        patch_half_span_max = 2


class Go2Stage2ACfg(Go2Stage2BaseCfg):
    """stage2a：中等难度真值地形。"""
    pass


class Go2Stage2BCfg(Go2Stage2BaseCfg):
    """stage2b：更强地形和随机化分布。"""

    class terrain(Go2Stage2BaseCfg.terrain):
        # 扩大课程覆盖并提高末期障碍难度。
        num_rows = 16
        num_cols = 12
        terrain_proportions = [0.12, 0.08, 0.30, 0.30, 0.20]
        terrain_curriculum_difficulty = {
            "slope": "difficulty * 0.55",
            "step_height": "0.06 + 0.22 * difficulty",
            "discrete_height": "0.06 + 0.22 * difficulty",
            "stepping_stones_size": "1.5 * (1.02 - difficulty)",
            "stone_distance": "0.08 if difficulty < 0.5 else 0.12",
            "gap_size": "1.2 * difficulty",
            "pit_depth": "0.4 * difficulty",
        }

    class commands(Go2Stage2BaseCfg.commands):
        max_curriculum = 1.2

        class ranges(Go2Stage2BaseCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            lin_vel_y = [-1.2, 1.2]
            ang_vel_yaw = [-1.4, 1.4]
            heading = [-3.14, 3.14]

    class domain_rand(Go2Stage2BaseCfg.domain_rand):
        friction_range = [0.1, 2.5]
        added_mass_range = [-2.0, 2.0]
        push_interval_s = 6
        max_push_vel_xy = 1.6
        com_pos_x_range = [-0.05, 0.05]
        com_pos_y_range = [-0.05, 0.05]
        com_pos_z_range = [-0.04, 0.04]
        kp_range = [0.7, 1.3]
        kd_range = [0.7, 1.3]


class Go2Stage2CCfg(Go2Stage2BCfg):
    """stage2c：用退化真值地形衔接 NSR 输入。"""

    class terrain_degrade(Go2Stage2BCfg.terrain_degrade):
        enable = True
        # 放慢退化课程，降低策略能力遗忘。
        curriculum_total_iters = 3000
        phase_ratios = [0.25, 0.55, 0.20]
        # 来自 NSR-vs-GT 标定批次的退化锚点。
        noise_std_m = [0.0025, 0.0056, 0.0095]
        point_dropout_prob = [0.084, 0.240, 0.415]
        patch_dropout_prob = [0.000, 0.084, 0.151]
        blur_prob = [0.079, 0.176, 0.320]
        bias_prob = [0.067, 0.149, 0.240]
        bias_std_m = [0.0025, 0.0056, 0.0098]
        delay_prob = [0.069, 0.153, 0.245]
        delay_mix = [0.080, 0.240, 0.400]
        patch_half_span_min = 1
        patch_half_span_max = 2


class Go2Stage2BaseCfgPPO(LeggedRobotTSCfgPPO):
    class policy(LeggedRobotTSCfgPPO.policy):
        critic_hidden_dims = [1024, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP"
        history_encoder_hidden_dims = [256, 128]
        history_encoder_channel_dims = [1, 1, 1, 1]
        history_encoder_dilation = [1, 1, 2, 1]
        history_encoder_stride = [1, 2, 1, 2]
        history_encoder_final_layer_dim = 128
        kernel_size = 5

    class algorithm(LeggedRobotTSCfgPPO.algorithm):
        # 编码器辅助损失仅作正则，PPO 仍是主目标。
        encoder_lr = 2.0e-4
        num_encoder_epochs = 1


class Go2Stage2ACfgPPO(Go2Stage2BaseCfgPPO):
    class runner(Go2Stage2BaseCfgPPO.runner):
        run_name = "stage2a" + get_simulator_suffix()
        experiment_name = "go2_stage2"
        save_interval = 100
        max_iterations = 5000


class Go2Stage2BCfgPPO(Go2Stage2BaseCfgPPO):
    class runner(Go2Stage2BaseCfgPPO.runner):
        run_name = "stage2b" + get_simulator_suffix()
        experiment_name = "go2_stage2"
        save_interval = 100
        max_iterations = 3000


class Go2Stage2CCfgPPO(Go2Stage2BaseCfgPPO):
    class algorithm(Go2Stage2BaseCfgPPO.algorithm):
        # 退化地形阶段降低 PPO 更新强度。
        learning_rate = 5.0e-4
        schedule = "fixed"
        entropy_coef = 0.005

    class runner(Go2Stage2BaseCfgPPO.runner):
        run_name = "stage2c" + get_simulator_suffix()
        experiment_name = "go2_stage2"
        save_interval = 100
        max_iterations = 3000
