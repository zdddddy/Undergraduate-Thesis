#!/usr/bin/env python3
"""
Internal helper script - runs a single task for smoke testing.
This is called by test_all_tasks.py for each task in a separate subprocess.
Supports IsaacGym, Genesis, and IsaacLab simulators.
"""

import sys
import argparse
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import legged_gym which automatically detects and imports the correct simulator
# based on Python version and SIMULATOR environment variable
from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import task_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    
    try:
        # Build command line args for get_args()
        from legged_gym.utils import get_args
        import sys
        
        original_argv = sys.argv
        sys.argv = ['train.py', '--task', args.task]
        if args.cpu:
            sys.argv.append('--cpu')
        if args.headless:
            sys.argv.append('--headless')
        
        train_args = get_args()
        sys.argv = original_argv
        
        # Initialize Genesis if using Genesis simulator
        if SIMULATOR == "genesis":
            gs.init(
                backend=gs.cpu if args.cpu else gs.gpu,
                logging_level='warning')
        
        # Create environment
        env, env_cfg = task_registry.make_env(name=args.task, args=train_args)
        
        # Get config and override iterations
        _, train_cfg = task_registry.get_cfgs(args.task)
        train_cfg.runner.max_iterations = args.iterations
        
        # Create runner
        ppo_runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=train_args,
            train_cfg=train_cfg
        )
        
        # Run training
        ppo_runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=True
        )
        
        # Cleanup
        if hasattr(env, 'destroy'):
            env.destroy()
        
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
