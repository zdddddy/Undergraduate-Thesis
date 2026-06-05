import os
import sys
import inspect

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if LEGGED_GYM_ROOT not in sys.path:
    sys.path.insert(0, LEGGED_GYM_ROOT)

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import shutil

def train(args):
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning')
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Copy env.py and env_config.py to log_dir for backup
    log_dir = ppo_runner.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    robot_file_path = inspect.getsourcefile(env.__class__)
    robot_config_path = inspect.getsourcefile(env_cfg.__class__)
    if robot_file_path is not None and os.path.exists(robot_file_path):
        shutil.copy(robot_file_path, log_dir)
    if robot_config_path is not None and os.path.exists(robot_config_path):
        shutil.copy(robot_config_path, log_dir)
    
    # Start training session
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.num_envs = 1
    train(args)
