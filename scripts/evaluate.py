"""
Evaluation script for MASTR
Compare MAAM with OR-Tools baseline on test instances
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
from tqdm import tqdm

from env.mvrp_env import MVRPSTWEnv
from model.maam_model import MAAM
from train.train_rl import REINFORCETrainer
from utils.metrics import evaluate_solution, compare_solutions, print_solution_summary, plot_routes
from utils.ortools_baseline import solve_with_ortools


def evaluate_maam(
    model_path: str,
    num_customers: int = 20,
    num_vehicles: int = 3,
    num_episodes: int = 100,
    seed: int = 42
):
    """
    Evaluate trained MAAM model
    """
    print("\n" + "="*60)
    print("EVALUATING MAAM MODEL")
    print("="*60)
    
    # Create environment
    env = MVRPSTWEnv(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=100.0,
        grid_size=100.0,
        seed=seed
    )
    
    # Create model
    model = MAAM(
        input_dim=8,
        embed_dim=128,
        num_heads=8,
        num_encoder_layers=3,
        ff_dim=512,
        dropout=0.1
    )
    
    # Create trainer and load checkpoint
    trainer = REINFORCETrainer(model=model, env=env)
    
    try:
        trainer.load_checkpoint(model_path)
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model checkpoint not found at {model_path}")
        print("  Training a new model for demonstration...")
        trainer.train(num_iterations=50, episodes_per_iter=16, eval_interval=10)
    
    # Evaluate
    print(f"\nEvaluating on {num_episodes} episodes...")
    metrics = trainer.evaluate(num_episodes=num_episodes)
    
    print("\nMAAM Evaluation Results:")
    print(f"  Average Cost:  {metrics['avg_cost']:.2f}")
    print(f"  Std Cost:      {metrics['std_cost']:.2f}")
    print(f"  Best Cost:     {metrics['best_cost']:.2f}")
    print(f"  Worst Cost:    {metrics['worst_cost']:.2f}")
    
    return metrics, env, trainer


def evaluate_ortools(
    env: MVRPSTWEnv,
    num_episodes: int = 100,
    time_limit: int = 30
):
    """
    Evaluate OR-Tools baseline
    """
    print("\n" + "="*60)
    print("EVALUATING OR-TOOLS BASELINE")
    print("="*60)
    
    costs = []
    
    for episode in tqdm(range(num_episodes), desc="OR-Tools"):
        # Reset environment
        env.reset(seed=42 + episode)
        
        # Solve with OR-Tools
        routes, cost, info = solve_with_ortools(
            locations=env.customer_locs,
            depot=env.depot,
            demands=env.demands,
            start_times=env.start_times,
            end_times=env.end_times,
            num_vehicles=env.num_vehicles,
            vehicle_capacity=env.vehicle_capacity,
            time_limit=time_limit
        )
        
        if info['status'] == 'solved':
            # Evaluate with time window penalties
            metrics = evaluate_solution(
                routes=routes,
                locations=env.customer_locs,
                depot=env.depot,
                demands=env.demands,
                start_times=env.start_times,
                end_times=env.end_times,
                penalties_early=env.penalties_early,
                penalties_late=env.penalties_late,
                vehicle_capacity=env.vehicle_capacity
            )
            costs.append(metrics['total_cost'])
        else:
            costs.append(float('inf'))
    
    results = {
        'avg_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'best_cost': np.min(costs),
        'worst_cost': np.max(costs)
    }
    
    print("\nOR-Tools Evaluation Results:")
    print(f"  Average Cost:  {results['avg_cost']:.2f}")
    print(f"  Std Cost:      {results['std_cost']:.2f}")
    print(f"  Best Cost:     {results['best_cost']:.2f}")
    print(f"  Worst Cost:    {results['worst_cost']:.2f}")
    
    return results


def compare_methods(maam_metrics, ortools_metrics):
    """
    Compare MAAM and OR-Tools
    """
    print("\n" + "="*60)
    print("COMPARISON: MAAM vs OR-Tools")
    print("="*60)
    
    metrics = ['avg_cost', 'std_cost', 'best_cost', 'worst_cost']
    
    for metric in metrics:
        maam_val = maam_metrics[metric]
        ortools_val = ortools_metrics[metric]
        diff = maam_val - ortools_val
        diff_pct = (diff / ortools_val * 100) if ortools_val != 0 else 0
        
        print(f"{metric.replace('_', ' ').title():.<20} MAAM: {maam_val:>8.2f} | OR-Tools: {ortools_val:>8.2f} | Diff: {diff:>+8.2f} ({diff_pct:>+6.1f}%)")
    
    # Determine winner
    if maam_metrics['avg_cost'] < ortools_metrics['avg_cost']:
        improvement = (ortools_metrics['avg_cost'] - maam_metrics['avg_cost']) / ortools_metrics['avg_cost'] * 100
        print(f"\n🏆 MAAM wins with {improvement:.1f}% improvement!")
    elif maam_metrics['avg_cost'] > ortools_metrics['avg_cost']:
        gap = (maam_metrics['avg_cost'] - ortools_metrics['avg_cost']) / ortools_metrics['avg_cost'] * 100
        print(f"\n📊 OR-Tools wins (MAAM is {gap:.1f}% worse)")
    else:
        print("\n🤝 Tie!")
    
    print("="*60)


def visualize_example(env, trainer):
    """
    Generate and visualize an example solution
    """
    print("\n" + "="*60)
    print("GENERATING EXAMPLE SOLUTION")
    print("="*60)
    
    # Reset environment
    env.reset(seed=123)
    
    # Generate MAAM solution
    done = False
    obs = env._get_obs()
    
    while not done:
        customer_features, vehicle_state = trainer._parse_observation(obs)
        mask = trainer._get_mask(obs)
        
        with torch.no_grad():
            action, _ = trainer.model.sample_action(
                customer_features, vehicle_state, mask, greedy=True
            )
        
        obs, reward, done, truncated, info = env.step(action.item())
    
    routes_maam = env.get_routes()
    
    # Evaluate MAAM solution
    metrics_maam = evaluate_solution(
        routes=routes_maam,
        locations=env.customer_locs,
        depot=env.depot,
        demands=env.demands,
        start_times=env.start_times,
        end_times=env.end_times,
        penalties_early=env.penalties_early,
        penalties_late=env.penalties_late,
        vehicle_capacity=env.vehicle_capacity
    )
    
    print("\nMAAM Solution:")
    print_solution_summary(metrics_maam)
    
    # Generate OR-Tools solution
    routes_ortools, cost_ortools, info = solve_with_ortools(
        locations=env.customer_locs,
        depot=env.depot,
        demands=env.demands,
        start_times=env.start_times,
        end_times=env.end_times,
        num_vehicles=env.num_vehicles,
        vehicle_capacity=env.vehicle_capacity,
        time_limit=30
    )
    
    # Evaluate OR-Tools solution
    metrics_ortools = evaluate_solution(
        routes=routes_ortools,
        locations=env.customer_locs,
        depot=env.depot,
        demands=env.demands,
        start_times=env.start_times,
        end_times=env.end_times,
        penalties_early=env.penalties_early,
        penalties_late=env.penalties_late,
        vehicle_capacity=env.vehicle_capacity
    )
    
    print("\nOR-Tools Solution:")
    print_solution_summary(metrics_ortools)
    
    # Compare
    compare_solutions(metrics_maam, metrics_ortools, names=("MAAM", "OR-Tools"))
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_routes(routes_maam, env.customer_locs, env.depot, title="MAAM Solution", save_path="results/maam_routes.png")
    plot_routes(routes_ortools, env.customer_locs, env.depot, title="OR-Tools Solution", save_path="results/ortools_routes.png")
    print("✓ Visualizations saved to results/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MASTR model")
    parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--num-customers', type=int, default=20, help='Number of customers')
    parser.add_argument('--num-vehicles', type=int, default=3, help='Number of vehicles')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compare-ortools', action='store_true', help='Compare with OR-Tools baseline')
    parser.add_argument('--visualize', action='store_true', help='Generate example visualization')
    
    args = parser.parse_args()
    
    # Evaluate MAAM
    maam_metrics, env, trainer = evaluate_maam(
        model_path=args.model_path,
        num_customers=args.num_customers,
        num_vehicles=args.num_vehicles,
        num_episodes=args.num_episodes,
        seed=args.seed
    )
    
    # Compare with OR-Tools
    if args.compare_ortools:
        ortools_metrics = evaluate_ortools(
            env=env,
            num_episodes=args.num_episodes,
            time_limit=30
        )
        compare_methods(maam_metrics, ortools_metrics)
    
    # Visualize example
    if args.visualize:
        visualize_example(env, trainer)
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
