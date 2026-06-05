# This file keeps the algorithm templates used by the Go2 obstacle-crossing
# project. Unused LeggedGym-Ex method templates were removed during cleanup.

from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class LeggedRobotTSCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 48
        num_privileged_obs = None
        frame_stack = 20
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        num_single_critic_obs = num_observations
        num_critic_obs = c_frame_stack * num_single_critic_obs


class LeggedRobotTSCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "TSRunner"

    class policy(LeggedRobotCfgPPO.policy):
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP"
        history_encoder_hidden_dims = [256, 128]
        history_encoder_channel_dims = [1, 1, 1, 1]
        history_encoder_dilation = [1, 1, 2, 1]
        history_encoder_stride = [1, 2, 1, 2]
        history_encoder_final_layer_dim = 128
        kernel_size = 5

    class algorithm(LeggedRobotCfgPPO.algorithm):
        encoder_lr = 1.0e-3
        num_encoder_epochs = 1

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticTS"
        algorithm_class_name = "PPO_TS"


class LeggedRobotDreamwaqCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 45
        frame_stack = 20
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 24
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * num_single_critic_obs


class LeggedRobotDreamwaqCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "DreamWaQRunner"

    class policy(LeggedRobotCfgPPO.policy):
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        encoder_lr = 2.0e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
