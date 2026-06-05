from legged_gym.envs.base.common_cfgs import get_simulator_suffix
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import (
    Go2DreamwaqCfg,
    Go2DreamwaqCfgPPO,
)
from legged_gym.envs.go2.go2_stage2.go2_stage2_config import Go2Stage2BCfg


class GO2BlindCfg(Go2DreamwaqCfg):
    """History-proprioception blind baseline on the paper terrain split.

    Actor input:
        - current 45D proprioceptive observation
        - DreamWaQ VAE latent inferred from stacked proprioceptive history

    Training-only privileged signals:
        - terrain heights in the asymmetric critic
        - contact/foot/base labels for the VAE auxiliary losses
    """

    class env(Go2DreamwaqCfg.env):
        num_envs = 4096
        num_actions = 12
        num_observations = 45
        frame_stack = 20
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 24
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        env_spacing = 0.5
        debug_draw_depth_images = False

    class terrain(Go2Stage2BCfg.terrain):
        measure_heights = True
        curriculum = True
        num_rows = 16
        num_cols = 12
        terrain_proportions = [0.12, 0.08, 0.30, 0.30, 0.20]
        terrain_curriculum_difficulty = {
            "slope": "difficulty * 0.55",
            "step_height": "0.04 + 0.13 * difficulty",
            "discrete_height": "0.06 + 0.12 * difficulty",
            "stepping_stones_size": "1.5 * (1.02 - difficulty)",
            "stone_distance": "0.08 if difficulty < 0.5 else 0.12",
            "gap_size": "1.2 * difficulty",
            "pit_depth": "0.4 * difficulty",
        }

    class asset(Go2Stage2BCfg.asset):
        pass

    class commands(Go2Stage2BCfg.commands):
        pass

    class domain_rand(Go2Stage2BCfg.domain_rand):
        pass


class GO2BlindCfgPPO(Go2DreamwaqCfgPPO):
    class policy(Go2DreamwaqCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [512, 256, 128]
        decoder_hidden_dims = [256, 128]

    class algorithm(Go2DreamwaqCfgPPO.algorithm):
        learning_rate = 5.0e-4
        schedule = "fixed"
        entropy_coef = 0.005
        encoder_lr = 2.0e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0

    class runner(Go2DreamwaqCfgPPO.runner):
        run_name = "blind" + get_simulator_suffix()
        experiment_name = "go2_blind"
        save_interval = 200
        max_iterations = 8000
