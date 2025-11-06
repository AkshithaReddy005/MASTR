import os
import subprocess
import argparse
import re
from pathlib import Path

def run_evaluation(data_path, num_vehicles=25, num_episodes=5, penalty_late=1.0, max_speed=17):
    """Run evaluation for a single data file and return the results."""
    cmd = [
        'python', 'evaluate_qlearning_improved.py',
        '--data_path', str(data_path),
        '--num_vehicles', str(num_vehicles),
        '--num_episodes', str(num_episodes),
        '--penalty_late', str(penalty_late),
        '--max_speed', str(max_speed)
    ]
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {data_path.name}")
    print(' '.join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {data_path.name}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Run evaluation on all instance files.')
    parser.add_argument('--num_vehicles', type=int, default=25, help='Number of vehicles')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes per instance')
    parser.add_argument('--penalty_late', type=float, default=1.0, help='Penalty for late arrivals')
    parser.add_argument('--max_speed', type=float, default=17.0, help='Maximum vehicle speed')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing instance files')
    args = parser.parse_args()
    
    # Find all instance files in the data directory
    instance_files = list(Path(args.data_dir).glob('c*.txt'))
    
    if not instance_files:
        print(f"No instance files found in {args.data_dir}")
        return
    
    print(f"Found {len(instance_files)} instance files in {args.data_dir}")
    
    # Run evaluation for each instance
    success_count = 0
    for instance_file in sorted(instance_files):
        success = run_evaluation(
            instance_file,
            args.num_vehicles,
            args.num_episodes,
            args.penalty_late,
            args.max_speed
        )
        if success:
            success_count += 1
    
    print(f"\nEvaluations completed! {success_count} out of {len(instance_files)} instances processed successfully.")

if __name__ == "__main__":
    main()
