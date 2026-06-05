"""论文 Go2 实验共用配置。"""


TERRAIN_LABELS = {
    "smooth_slope": "平滑坡面",
    "rough_slope": "粗糙坡面",
    "stairs_up": "上台阶",
    "stairs_down": "下台阶",
    "discrete_obstacles": "离散障碍物",
}


def performance_note(success_rate, fall_rate, episode_length):
    if success_rate >= 0.95:
        prefix = "稳定通过"
    elif success_rate >= 0.80:
        prefix = "基本可通过"
    elif success_rate >= 0.60:
        prefix = "可通过但稳定性不足"
    else:
        prefix = "通过能力不足"

    details = []
    if fall_rate >= 0.20:
        details.append("跌倒较多")
    elif fall_rate > 0.0:
        details.append("存在少量跌倒")
    else:
        details.append("未见跌倒")

    if episode_length >= 950:
        details.append("episode 接近满长")
    elif episode_length >= 800:
        details.append("episode 长度较高")
    else:
        details.append("episode 偏短")
    return prefix + "，" + "，".join(details)


def disable_eval_randomness(env_cfg):
    """关闭评估中的观测噪声、扰动和动力学随机化。"""
    if hasattr(env_cfg, "noise") and hasattr(env_cfg.noise, "add_noise"):
        env_cfg.noise.add_noise = False
    if not hasattr(env_cfg, "domain_rand"):
        return

    for attr in (
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_com_displacement",
        "randomize_ctrl_delay",
        "randomize_pd_gain",
        "randomize_joint_armature",
        "randomize_joint_friction",
        "randomize_joint_damping",
        "push_robots",
        "push_links",
    ):
        if hasattr(env_cfg.domain_rand, attr):
            setattr(env_cfg.domain_rand, attr, False)

    scalar_ranges = {
        "friction_range": [1.0, 1.0],
        "restitution_range": [0.0, 0.0],
        "added_mass_range": [0.0, 0.0],
        "com_pos_x_range": [0.0, 0.0],
        "com_pos_y_range": [0.0, 0.0],
        "com_pos_z_range": [0.0, 0.0],
        "ctrl_delay_step_range": [0, 0],
        "kp_range": [1.0, 1.0],
        "kd_range": [1.0, 1.0],
        "joint_armature_range": [0.0, 0.0],
        "joint_friction_range": [0.0, 0.0],
        "joint_damping_range": [0.0, 0.0],
    }
    for attr, value in scalar_ranges.items():
        if hasattr(env_cfg.domain_rand, attr):
            setattr(env_cfg.domain_rand, attr, list(value))
    if hasattr(env_cfg.domain_rand, "max_push_vel_xy"):
        env_cfg.domain_rand.max_push_vel_xy = 0.0
    if hasattr(env_cfg.domain_rand, "max_push_force"):
        env_cfg.domain_rand.max_push_force = 0.0
