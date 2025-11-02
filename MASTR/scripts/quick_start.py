"""
Quick Start Script for MASTR
Demonstrates basic usage in a single script
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

from env.mvrp_env import MVRPSTWEnv
from model.maam_model import MAAM
from train.train_rl import REINFORCETrainer


def main():
    print("="*60)
    print("MASTR: Multi-Agent Soft Time Routing - Quick Start")
    print("="*60)
    
    # 1. Create Environment
    print("\n[1/5] Creating environment...")
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=100.0,
        grid_size=100.0,
        seed=42
    )
    print(f"✓ Environment created: {env.num_customers} customers, {env.num_vehicles} vehicles")
    
    # 2. Create Model
    print("\n[2/5] Creating MAAM model...")
    model = MAAM(
        input_dim=8,
        embed_dim=128,
        num_heads=8,
        num_encoder_layers=3,
        ff_dim=512,
        dropout=0.1
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    
    # 3. Create Trainer
    print("\n[3/5] Creating trainer...")
    trainer = REINFORCETrainer(
        model=model,
        env=env,
        learning_rate=1e-4,
        baseline_lr=1e-3,
        gamma=0.99
    )
    print(f"✓ Trainer initialized on device: {trainer.device}")
    
    # 4. Train Model (small demo)
    print("\n[4/5] Training model (demo with 100 iterations)...")
    print("     For full training, use train_rl.py with 1000+ iterations")
    trainer.train(
        num_iterations=100,
        episodes_per_iter=16,
        eval_interval=20
    )
    print("✓ Training complete!")
    
    # 5. Evaluate
    print("\n[5/5] Evaluating model...")
    eval_metrics = trainer.evaluate(num_episodes=10)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Average Cost:  {eval_metrics['avg_cost']:.2f}")
    print(f"Std Cost:      {eval_metrics['std_cost']:.2f}")
    print(f"Best Cost:     {eval_metrics['best_cost']:.2f}")
    print(f"Worst Cost:    {eval_metrics['worst_cost']:.2f}")
    print("="*60)
    
    # 6. Generate Example Solution
    print("\n[Bonus] Generating example solution...")
    env.reset(seed=123)
    done = False
    obs = env._get_obs()
    
    step_count = 0
    while not done:
        customer_features, vehicle_state = trainer._parse_observation(obs)
        mask = trainer._get_mask(obs)
        
        with torch.no_grad():
            action, log_prob = model.sample_action(
                customer_features, vehicle_state, mask, greedy=True
            )
        
        obs, reward, done, truncated, info = env.step(action.item())
        step_count += 1
    
    routes = env.get_routes()
    total_cost = env.get_solution_cost()
    
    print(f"\n✓ Solution generated in {step_count} steps")
    print(f"  Total Cost: {total_cost:.2f}")
    print(f"  Routes:")
    for i, route in enumerate(routes):
        print(f"    Vehicle {i+1}: {len(route)} customers - {route}")
    
    print("\n" + "="*60)
    print("QUICK START COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("  1. Train longer: python MASTR/train/train_rl.py")
    print("  2. Evaluate: python MASTR/scripts/evaluate.py --compare-ortools --visualize")
    print("  3. Explore notebook: jupyter notebook MASTR/notebooks/demo.ipynb")
    print("  4. Monitor training: tensorboard --logdir runs/maam_training")
    print("="*60)


if __name__ == '__main__':
    main()
