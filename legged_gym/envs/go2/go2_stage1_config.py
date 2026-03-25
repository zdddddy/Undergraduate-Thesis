from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO


class GO2Stage1BlindFlatCfg(GO2RoughCfg):
    """Stage-1(a): blind/proprioceptive warm-up on flat ground."""

    class env(GO2RoughCfg.env):
        num_observations = 48
        episode_length_s = 20

    class terrain(GO2RoughCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False


class GO2Stage1BlindFlatCfgPPO(GO2RoughCfgPPO):
    class runner(GO2RoughCfgPPO.runner):
        experiment_name = "go2_stage1_blind_flat"
        run_name = ""
        max_iterations = 1500
        save_interval = 50


class GO2Stage1BlindHardenedCfg(GO2Stage1BlindFlatCfg):
    """Stage-1(b): still blind, but with mild terrain/domain perturbations."""

    class env(GO2Stage1BlindFlatCfg.env):
        episode_length_s = 24

    class terrain(GO2Stage1BlindFlatCfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        min_init_terrain_level = 0
        max_init_terrain_level = 2
        curriculum_move_up_dist = 2.5
        curriculum_move_down_dist = 1.0
        curriculum_recycle_min_level = 3
        num_rows = 8
        num_cols = 10
        # Restrict to mild slopes and rough-slope undulations only.
        # [smooth slope, rough slope, stairs up, stairs down, obstacles]
        terrain_proportions = [0.55, 0.45, 0.0, 0.0, 0.0]

    class commands(GO2Stage1BlindFlatCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        resampling_time = 8.0

        class ranges(GO2Stage1BlindFlatCfg.commands.ranges):
            lin_vel_x = [0.2, 1.0]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.9, 0.9]
            heading = [-3.14, 3.14]

    class rewards(GO2Stage1BlindFlatCfg.rewards):
        class scales(GO2Stage1BlindFlatCfg.rewards.scales):
            # task objective
            tracking_lin_vel = 1.8
            tracking_ang_vel = 0.9
            feet_air_time = 0.0
            foot_clearance = 0.9

            # safety + stability
            collision = -2.5
            feet_stumble = -1.2
            feet_contact_forces = -1.0e-3
            termination = -4.0
            orientation = -1.0
            lin_vel_z = -2.2
            ang_vel_xy = -0.25
            base_height = -1.8

            # regularization
            torques = -5.0e-5
            dof_vel = -1.2e-4
            dof_acc = -1.0e-6
            dof_pos_limits = -1.0
            action_rate = -0.04
            cmd_nonzero_but_slow = -0.8
            # stronger anti-synchrony on mildly perturbed terrains
            pronk = -0.25
            non_alternating = -0.6
            stand_still = -0.0

    class domain_rand(GO2Stage1BlindFlatCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.5]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.5]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.8


class GO2Stage1BlindHardenedCfgPPO(GO2Stage1BlindFlatCfgPPO):
    class algorithm(GO2Stage1BlindFlatCfgPPO.algorithm):
        learning_rate = 5.0e-4
        entropy_coef = 0.008

    class runner(GO2Stage1BlindFlatCfgPPO.runner):
        experiment_name = "go2_stage1_blind_hardened"
        run_name = ""
        max_iterations = 4000
        save_interval = 100


# Canonical stage-1 alias.
class GO2Stage1BlindCfg(GO2Stage1BlindFlatCfg):
    pass


class GO2Stage1BlindCfgPPO(GO2Stage1BlindFlatCfgPPO):
    class runner(GO2Stage1BlindFlatCfgPPO.runner):
        experiment_name = "go2_stage1_blind"


# Backward-compatible aliases used by existing scripts/task names.
class GO2Stage1GTCleanCfg(GO2Stage1BlindFlatCfg):
    pass


class GO2Stage1GTCleanCfgPPO(GO2Stage1BlindFlatCfgPPO):
    class runner(GO2Stage1BlindFlatCfgPPO.runner):
        experiment_name = "rough_go2_stage1_gt_clean"
