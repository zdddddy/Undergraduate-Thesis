import time

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if LEGGED_GYM_ROOT not in sys.path:
    sys.path.insert(0, LEGGED_GYM_ROOT)

from legged_gym import *

from legged_gym.envs import *
from legged_gym.utils import *

import numpy as np
import torch
    
def override_configs(env_cfg, args, task_type):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
        task_type: type of the task
    """
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = min(env_cfg.env.num_camera_envs, env_cfg.env.num_envs)
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))

    # Disable domain randomization in play to keep evaluation behavior deterministic.
    if hasattr(env_cfg, "domain_rand"):
        # Turn off all supported randomization flags.
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
            "randomize_camera_pos",
            "randomize_camera_euler",
        ):
            if hasattr(env_cfg.domain_rand, attr):
                setattr(env_cfg.domain_rand, attr, False)

        # Disable external pushes.
        if hasattr(env_cfg.domain_rand, "push_robots"):
            env_cfg.domain_rand.push_robots = False
        if hasattr(env_cfg.domain_rand, "max_push_vel_xy"):
            env_cfg.domain_rand.max_push_vel_xy = 0.0
        if hasattr(env_cfg.domain_rand, "push_links"):
            env_cfg.domain_rand.push_links = False
        if hasattr(env_cfg.domain_rand, "max_push_force"):
            env_cfg.domain_rand.max_push_force = 0.0

        # Collapse randomization ranges to deterministic values for safety.
        if hasattr(env_cfg.domain_rand, "friction_range"):
            env_cfg.domain_rand.friction_range = [1.0, 1.0]
        if hasattr(env_cfg.domain_rand, "restitution_range"):
            env_cfg.domain_rand.restitution_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "added_mass_range"):
            env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "com_pos_x_range"):
            env_cfg.domain_rand.com_pos_x_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "com_pos_y_range"):
            env_cfg.domain_rand.com_pos_y_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "com_pos_z_range"):
            env_cfg.domain_rand.com_pos_z_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "ctrl_delay_step_range"):
            env_cfg.domain_rand.ctrl_delay_step_range = [0, 0]
        if hasattr(env_cfg.domain_rand, "kp_range"):
            env_cfg.domain_rand.kp_range = [1.0, 1.0]
        if hasattr(env_cfg.domain_rand, "kd_range"):
            env_cfg.domain_rand.kd_range = [1.0, 1.0]
        if hasattr(env_cfg.domain_rand, "joint_armature_range"):
            env_cfg.domain_rand.joint_armature_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "joint_friction_range"):
            env_cfg.domain_rand.joint_friction_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "joint_damping_range"):
            env_cfg.domain_rand.joint_damping_range = [0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "camera_com_displacement_range"):
            env_cfg.domain_rand.camera_com_displacement_range = [0.0, 0.0, 0.0]
        if hasattr(env_cfg.domain_rand, "camera_euler_range"):
            env_cfg.domain_rand.camera_euler_range = [0.0, 0.0, 0.0]

    # Disable observation noise for play.
    if hasattr(env_cfg, "noise") and hasattr(env_cfg.noise, "add_noise"):
        env_cfg.noise.add_noise = False

    # adjust parameters according to terrain type
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        # Use mixed curriculum terrain in play so all rough-terrain categories are covered.
        env_cfg.terrain.num_rows = 4
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.border_size = 5.0
        env_cfg.terrain.curriculum = True
        if hasattr(env_cfg.terrain, "selected"):
            env_cfg.terrain.selected = False
        env_cfg.env.debug_draw_terrain_height_points = False
        # Ensure five-class rough-terrain mix in play:
        # [slope, random_uniform, stairs_up/down, discrete_obstacles].
        if hasattr(env_cfg.terrain, "terrain_proportions") and len(env_cfg.terrain.terrain_proportions) == 5:
            env_cfg.terrain.terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
        # selected-terrain mode uses terrain_kwargs; clear it for mixed curriculum mode.
        if hasattr(env_cfg.terrain, "terrain_kwargs"):
            env_cfg.terrain.terrain_kwargs = None

    env_cfg.env.debug = True
    env_cfg.commands.zero_cmd_prob = 0.0 # for testing, use non-zero commands all the time
    env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    
    if args.use_joystick:
        env_cfg.commands.heading_command = False

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"knee pitch: {env.simulator.dof_pos[robot_index, [13,19]].cpu().numpy()}")
    # print(f"feet distance: {torch.norm(env.simulator.feet_pos[robot_index, 0, [0, 1]] - env.simulator.feet_pos[robot_index, 1, [0, 1]]).item()}")
    # print(f"actions: {env.simulator.dof_pos[robot_index].cpu().numpy()}")
    # print(f"command: {env.commands[robot_index].cpu().numpy()}")
    # print(f"dr_ctrl_delay: {env.simulator.dr_ctrl_delay[robot_index].item()}")
    pass

def interaction_loop(env, policy, args, task_type):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
    """
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
        
    ts_like_task = task_type in {
        "stage2",
        "stage2a",
        "stage2b",
        "stage2c",
        "stage3",
        "stage3a",
        "stage3b",
    }

    # Get initial observations according to task type
    dreamwaq_like_task = task_type == "blind"

    if ts_like_task:
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif dreamwaq_like_task:
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    else:
        obs_buf = env.get_observations()
    
    # Setup joystick if needed
    if args.use_joystick:
        from legged_gym.scripts.joystick import Joystick
        joystick = Joystick(joystick_type=args.joystick_type)
    
    frame_dt = 1 / 60.0 # 30Hz
    ts_policy_mode = getattr(args, "ts_policy_mode", "aux")

    # interaction loop
    for i in range(10*int(env.max_episode_length)):
        
        t_start = time.perf_counter()
        # update commands from joystick
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx
        
        # set the viewer camera to follow the first environment by default
        if args.follow_robot:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
            env.set_viewer_camera(pos, lookat)
            
        # Step the environment according to task type
        if ts_like_task:
            if ts_policy_mode == "deploy":
                actions = policy(obs_buf, privileged_obs_buf)
            else:
                actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif dreamwaq_like_task:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        else:
            actions = policy(obs_buf.detach())
            obs_buf, _, rews, dones, infos = env.step(actions.detach())
        
        # print debug info
        print_debug_info(env, robot_index)
        
        # Update logger info
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
                    # 'contact_forces_z': env.feet_max_force_z[robot_index, 
                    #                                             env.simulator.feet_contact_indices].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        # sleep for the remainder of the frame budget to match real-time playback
        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg, task_type, ts_policy_mode: str):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    ts_like_task = task_type in {
        "stage2",
        "stage2a",
        "stage2b",
        "stage2c",
        "stage3",
        "stage3a",
        "stage3b",
    }

    dreamwaq_like_task = task_type == "blind"

    if ts_like_task:
        if task_type in {"stage3", "stage3a", "stage3b"}:
            if ts_policy_mode == "deploy":
                exporter = PolicyExporterTSDeploy(alg_runner.alg.actor_critic)
            else:
                exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        else:
            if ts_policy_mode == "deploy":
                exporter = PolicyExporterTSTeacher(alg_runner.alg.actor_critic)
            else:
                exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif dreamwaq_like_task:
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    splitted = args.task.split("_")
    # Task names are "go2" or "go2_<variant>"; the variant becomes task_type.
    task_type = "_".join(splitted[1:])
    print("Task type: ", task_type)
    ts_like_task = task_type in {
        "stage2",
        "stage2a",
        "stage2b",
        "stage2c",
        "stage3",
        "stage3a",
        "stage3b",
    }
    stage3_like_task = task_type in {"stage3", "stage3a", "stage3b"}
    if task_type in {"stage2", "stage2a", "stage2b", "stage2c"} and not args.use_teacher:
        print("Warning: stage2 play defaults to student policy (no privileged heights). "
              "Use --use_teacher to evaluate GT-teacher behavior.")

    if stage3_like_task:
        if args.use_aux_policy:
            ts_policy_mode = "aux"
            print("Stage3 inference mode set to AUX history branch (--use_aux_policy).")
        else:
            ts_policy_mode = "deploy"
            if args.use_teacher:
                print("Info: --use_teacher is a legacy alias. Stage3 default already uses deploy branch.")
            print("Stage3 inference mode default: DEPLOY branch (runtime terrain input).")
    else:
        ts_policy_mode = "deploy" if args.use_teacher else "aux"

    args.ts_policy_mode = ts_policy_mode
    override_configs(env_cfg, args, task_type)

    if stage3_like_task:
        # Inference does not require distillation teacher attachment.
        # Avoid forcing --teacher_model_path for stage3 play.
        for key in [
            "distill_action_coef",
            "distill_action_coef_final",
            "distill_latent_coef",
            "distill_latent_coef_final",
            "distill_height_coef",
            "distill_total_iters",
        ]:
            if hasattr(train_cfg.algorithm, key):
                setattr(train_cfg.algorithm, key, 0.0 if key != "distill_total_iters" else 0)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if ts_like_task:
        if ts_policy_mode == "deploy":
            print("TS inference mode: deploy (uses terrain/privileged observations).")
            policy = ppo_runner.get_deploy_inference_policy(device=env.device)
        else:
            print("TS inference mode: aux history branch.")
            policy = ppo_runner.get_aux_inference_policy(device=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++ or python)
    export_run_name = os.path.basename(os.path.normpath(train_cfg.runner.load_run))
    path = os.path.join(LEGGED_GYM_RESULTS_DIR, 'training_logs', train_cfg.runner.experiment_name,
                            export_run_name, 'exported')
    export_policy(ppo_runner, path, args, env_cfg, train_cfg, task_type, ts_policy_mode)

    interaction_loop(env, policy, args, task_type)
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
