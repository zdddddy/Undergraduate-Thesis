#!/usr/bin/env python3
"""
Smoke test script for LeggedGym-Ex training tasks.

This script runs a quick smoke test on all registered training tasks,
executing only 5 iterations to verify that training can start normally.

Each task runs in a separate subprocess to avoid PhysX Foundation conflicts.

Usage:
    python scripts/test_all_tasks.py
    python scripts/test_all_tasks.py --tasks go2 go2_stage2 go2_stage3  # Test specific tasks
    python scripts/test_all_tasks.py --iterations 3     # Run 3 iterations instead of 5
    python scripts/test_all_tasks.py --cpu             # Use CPU instead of GPU

Returns:
    Exit code 0 if all tests pass, 1 if any task fails
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# We only import task_registry here to list tasks, not to run them
def get_registered_tasks() -> List[str]:
    """Get list of all registered task names."""
    # Import here to avoid loading isaacgym in main process
    from legged_gym.envs import task_registry
    return list(task_registry.task_classes.keys())


def run_task_subprocess(task_name: str, num_iterations: int, use_cpu: bool, headless: bool) -> Dict:
    """
    Run a single task in a subprocess to avoid PhysX Foundation conflicts.
    
    Returns:
        Dictionary with 'success', 'message', and 'duration' keys
    """
    script_path = os.path.join(os.path.dirname(__file__), '_test_single_task.py')
    
    cmd = [
        sys.executable,
        script_path,
        '--task', task_name,
        '--iterations', str(num_iterations),
    ]
    
    if use_cpu:
        cmd.append('--cpu')
    if headless:
        cmd.append('--headless')
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per task
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return {
                'success': True,
                'message': f"Completed {num_iterations} iterations",
                'duration': duration
            }
        else:
            # Parse error from stderr
            error_lines = result.stderr.strip().split('\n') if result.stderr else ['Unknown error']
            # Get the last meaningful error line
            error_msg = error_lines[-1] if error_lines else 'Unknown error'
            return {
                'success': False,
                'message': error_msg,
                'duration': duration
            }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return {
            'success': False,
            'message': 'Timeout (300s)',
            'duration': duration
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'message': f"{type(e).__name__}: {str(e)}",
            'duration': duration
        }


def run_smoke_tests(
    tasks: List[str] = None,
    num_iterations: int = 5,
    use_cpu: bool = False,
    headless: bool = True,
    continue_on_error: bool = True
) -> Dict:
    """
    Run smoke tests on specified tasks.
    """
    if tasks is None:
        tasks = get_registered_tasks()
    
    print("=" * 80)
    print(f"LeggedGym-Ex Smoke Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing {len(tasks)} task(s) with {num_iterations} iteration(s) each")
    print(f"Device: {'CPU' if use_cpu else 'GPU'}")
    print("=" * 80)
    print()
    
    results = {
        'passed': [],
        'failed': [],
        'total': len(tasks),
        'start_time': time.time()
    }
    
    for i, task_name in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] Testing task: {task_name}...", end=" ", flush=True)
        
        result = run_task_subprocess(
            task_name=task_name,
            num_iterations=num_iterations,
            use_cpu=use_cpu,
            headless=headless
        )
        
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} ({result['duration']:.1f}s)")
        
        if result['success']:
            results['passed'].append({
                'task': task_name,
                'message': result['message'],
                'duration': result['duration']
            })
        else:
            results['failed'].append({
                'task': task_name,
                'message': result['message'],
                'duration': result['duration']
            })
            print(f"    Error: {result['message']}")
            
            if not continue_on_error:
                print("\nStopping on first failure as requested.")
                break
    
    results['end_time'] = time.time()
    results['total_duration'] = results['end_time'] - results['start_time']
    
    return results


def print_summary(results: Dict):
    """Print test summary report."""
    print()
    print("=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    
    total = results['total']
    passed = len(results['passed'])
    failed = len(results['failed'])
    
    print(f"\nTotal Tasks: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Total Duration: {results['total_duration']:.1f}s")
    
    if results['failed']:
        print("\n❌ FAILED TASKS:")
        print("-" * 80)
        for failure in results['failed']:
            print(f"  • {failure['task']}: {failure['message']}")
    
    if results['passed']:
        print("\n✅ PASSED TASKS:")
        print("-" * 80)
        for success in results['passed']:
            print(f"  • {success['task']} ({success['duration']:.1f}s)")
    
    print()
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All smoke tests passed!")
    else:
        print(f"⚠️  {failed} task(s) failed. Check output above for details.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Run smoke tests on LeggedGym-Ex training tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all registered tasks (default)
  python scripts/test_all_tasks.py
  
  # Test specific tasks
  python scripts/test_all_tasks.py --tasks go2 go2_stage2 go2_stage3
  
  # Run 3 iterations instead of 5
  python scripts/test_all_tasks.py --iterations 3
  
  # Use CPU instead of GPU
  python scripts/test_all_tasks.py --cpu
  
  # Stop on first failure
  python scripts/test_all_tasks.py --stop-on-error
        """
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        help='Specific task names to test (default: all registered tasks)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of training iterations per task (default: 5)'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run in headless mode (default: True)'
    )
    
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop testing on first failure instead of continuing'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all registered tasks and exit'
    )
    
    args = parser.parse_args()
    
    # List tasks and exit if requested
    if args.list:
        tasks = get_registered_tasks()
        print("Registered tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task}")
        return 0
    
    # Run smoke tests
    results = run_smoke_tests(
        tasks=args.tasks,
        num_iterations=args.iterations,
        use_cpu=args.cpu,
        headless=args.headless,
        continue_on_error=not args.stop_on_error
    )
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    return 0 if len(results['failed']) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
