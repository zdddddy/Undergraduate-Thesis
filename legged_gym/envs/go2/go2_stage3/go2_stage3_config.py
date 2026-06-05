import os

from legged_gym import LEGGED_GYM_RESULTS_DIR
from legged_gym.camera_profiles import GO2_D435I_4CAM_PROFILE
from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_stage2.go2_stage2_config import (
    Go2Stage2BCfg,
    Go2Stage2CCfg,
    Go2Stage2BaseCfgPPO,
)


class Go2Stage3BaseCfg(Go2Stage2BCfg):
    """代码 stage3：论文越障阶段二，actor 地形输入切到 NSR。"""

    class env(Go2Stage2BCfg.env):
        # 四相机点云占显存，默认减小并行数。
        num_envs = 1024
        num_camera_envs = num_envs

    class terrain_degrade(Go2Stage2BCfg.terrain_degrade):
        # stage3 直接使用 NSR 高度，不做合成退化。
        enable = False

    class sensor(Go2Stage2BCfg.sensor):
        add_depth = True
        use_warp = True

        class depth_camera_config(Go2Stage2BCfg.sensor.depth_camera_config):
            num_sensors = GO2_D435I_4CAM_PROFILE["num_sensors"]
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

    class nsr(Go2Stage2BCfg.nsr):
        enable = True
        ckpt = os.path.join(
            LEGGED_GYM_RESULTS_DIR,
            "nsr",
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

    class stage3:
        # 子类可覆盖 warmup 设置。
        warmup_iters = 0
        warmup_max_terrain_level = 9
        warmup_friction_range = [0.2, 1.7]
        warmup_added_mass_range = [-1.0, 1.0]
        warmup_push_interval_s = 10
        warmup_max_push_vel_xy = 1.0
        warmup_com_pos_x_range = [-0.03, 0.03]
        warmup_com_pos_y_range = [-0.03, 0.03]
        warmup_com_pos_z_range = [-0.03, 0.03]
        warmup_kp_range = [0.8, 1.2]
        warmup_kd_range = [0.8, 1.2]
        # 禁用指令课程时使用的 warmup 上限。
        warmup_max_command_x = 0.8
        warmup_max_command_y = 1.0
        warmup_max_ang_vel_yaw = 1.0
        # NSR 运行指标
        track_runtime_metrics = True


class Go2Stage3BaseCfgPPO(Go2Stage2BaseCfgPPO):
    class algorithm(Go2Stage2BaseCfgPPO.algorithm):
        # stage3 只做 PPO 适配。
        schedule = "fixed"
        num_mini_batches = 2
        entropy_coef = 0.001
        lr_decay_total_iters = 0
        privilege_encoder_freeze_iters = 0
        privilege_encoder_grad_scale = 1.0

        # 关闭蒸馏。
        distill_action_coef = 0.0
        distill_action_coef_final = 0.0
        distill_latent_coef = 0.0
        distill_latent_coef_final = 0.0
        distill_height_coef = 0.0
        distill_total_iters = 0
        distill_terrain_dim = 81

    class runner(Go2Stage2BaseCfgPPO.runner):
        experiment_name = "go2_stage3"
        teacher_model_path = ""
        reset_optimizer_on_resume = True
        reset_iteration_on_resume = True
        save_interval = 100


class Go2Stage3ACfg(Go2Stage3BaseCfg):
    """stage3a：从 stage2c 保守适配到 NSR。"""

    # 3a 复用 stage2c 分布。
    class terrain(Go2Stage2CCfg.terrain):
        pass

    class rewards(Go2Stage2CCfg.rewards):
        foot_clearance_target = 0.10

        class scales(Go2Stage2CCfg.rewards.scales):
            feet_air_time = 1.1
            foot_clearance = 0.25

    class commands(Go2Stage2CCfg.commands):
        pass

    class domain_rand(Go2Stage2CCfg.domain_rand):
        pass

    class terrain_degrade(Go2Stage2CCfg.terrain_degrade):
        # 使用真实 NSR 高度。
        enable = False

    class stage3(Go2Stage3BaseCfg.stage3):
        warmup_iters = 0


class Go2Stage3ACfgPPO(Go2Stage3BaseCfgPPO):
    class algorithm(Go2Stage3BaseCfgPPO.algorithm):
        # 从 2c 到 NSR 保守适配。
        learning_rate = 1.5e-4
        schedule = "fixed"
        lr_decay_total_iters = 0
        num_mini_batches = 2
        entropy_coef = 0.003
        privilege_encoder_freeze_iters = 150
        privilege_encoder_grad_scale = 0.3

        # 关闭蒸馏。
        distill_action_coef = 0.0
        distill_action_coef_final = 0.0
        distill_latent_coef = 0.0
        distill_latent_coef_final = 0.0
        distill_height_coef = 0.0
        distill_total_iters = 0
        distill_terrain_dim = 81

    class runner(Go2Stage3BaseCfgPPO.runner):
        run_name = "stage3a" + get_simulator_suffix()
        experiment_name = "go2_stage3"
        teacher_model_path = ""
        # 继承 stage2c 优化器状态。
        reset_optimizer_on_resume = False
        reset_iteration_on_resume = True
        save_interval = 100
        max_iterations = 3000


class Go2Stage3BCfg(Go2Stage3BaseCfg):
    """stage3b：在 hard 分布下继续强化 NSR 策略。"""

    # 3b 对齐 stage2b hard 分布。
    class terrain(Go2Stage2BCfg.terrain):
        pass

    class rewards(Go2Stage2BCfg.rewards):
        foot_clearance_target = 0.10

        class scales(Go2Stage2BCfg.rewards.scales):
            feet_air_time = 1.1
            foot_clearance = 0.25

    class commands(Go2Stage2BCfg.commands):
        pass

    class domain_rand(Go2Stage2BCfg.domain_rand):
        pass

    class terrain_degrade(Go2Stage2BCfg.terrain_degrade):
        enable = False

    class stage3(Go2Stage3BaseCfg.stage3):
        warmup_iters = 0


class Go2Stage3BCfgPPO(Go2Stage3BaseCfgPPO):
    class algorithm(Go2Stage3BaseCfgPPO.algorithm):
        # hard 分布下继续强化。
        learning_rate = 2.5e-4
        schedule = "fixed"
        lr_decay_total_iters = 0
        num_mini_batches = 2
        entropy_coef = 0.002
        privilege_encoder_freeze_iters = 0
        privilege_encoder_grad_scale = 0.3

        distill_action_coef = 0.0
        distill_action_coef_final = 0.0
        distill_latent_coef = 0.0
        distill_latent_coef_final = 0.0
        distill_height_coef = 0.0
        distill_total_iters = 0
        distill_terrain_dim = 81

    class runner(Go2Stage3BaseCfgPPO.runner):
        run_name = "stage3b" + get_simulator_suffix()
        experiment_name = "go2_stage3"
        teacher_model_path = ""
        reset_optimizer_on_resume = False
        reset_iteration_on_resume = False
        save_interval = 100
        max_iterations = 1000


class Go2Stage3Cfg(Go2Stage3ACfg):
    """统一 stage3：新实验建议使用的单阶段 NSR 适配。"""

    pass


class Go2Stage3CfgPPO(Go2Stage3BaseCfgPPO):
    class algorithm(Go2Stage3BaseCfgPPO.algorithm):
        # 从 stage2c checkpoint 做单阶段 NSR 适配。
        learning_rate = 2.0e-4
        schedule = "fixed"
        lr_decay_total_iters = 0
        num_mini_batches = 2
        entropy_coef = 0.0025
        privilege_encoder_freeze_iters = 150
        privilege_encoder_grad_scale = 0.3

        distill_action_coef = 0.0
        distill_action_coef_final = 0.0
        distill_latent_coef = 0.0
        distill_latent_coef_final = 0.0
        distill_height_coef = 0.0
        distill_total_iters = 0
        distill_terrain_dim = 81

    class runner(Go2Stage3BaseCfgPPO.runner):
        run_name = "stage3" + get_simulator_suffix()
        experiment_name = "go2_stage3"
        teacher_model_path = ""
        reset_optimizer_on_resume = False
        reset_iteration_on_resume = True
        save_interval = 100
        max_iterations = 5000
