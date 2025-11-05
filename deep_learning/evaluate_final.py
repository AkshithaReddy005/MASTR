"""
Final Evaluation Script for MAAM Model
Compatible with PyTorch 2.6+
"""
import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import after path is set
from model.maam_model import MAAM
from env.mvrp_env import MVRPSTWEnv

def load_model_weights(model, model_path):
    """Safely load model weights with PyTorch 2.6+ compatibility"""
    try:
        # Try loading with weights_only=False first (for PyTorch 2.6+)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Fallback for older PyTorch
        checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    return model

def evaluate():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate MAAM Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()
    
    # Environment configuration (must match training config)
    env_config = {
        'num_customers': 20,
        'num_vehicles': 3,
        'vehicle_capacity': 200,
        'data_path': 'data/raw/C101.csv',
        'max_time': 1236,
        'penalty_early': 100,
        'penalty_late': 200,
        'seed': 42
    }
    
    # Model configuration (must match training config)
    model_config = {
        'input_dim': 8,
        'embed_dim': 64,
        'num_heads': 4,
        'num_encoder_layers': 2,
        'ff_dim': 128,
        'dropout': 0.1,
        'tanh_clipping': 10.0,
        'init_gain': 0.1
    }
    
    # Create environment and model
    print("Creating environment...")
    env = MVRPSTWEnv(**env_config)
    
    print("Initializing model...")
    model = MAAM(**model_config)
    
    # Load model weights
    print(f"Loading model from {args.model}...")
    model = load_model_weights(model, args.model)
    model.eval()
    
    # Evaluation loop
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL")
    print(f"Episodes: {args.episodes}")
    print(f"Rendering: {args.render}")
    print(f"{'='*70}\n")
    
    total_reward = 0.0
    
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 200:  # Prevent infinite loops
            # Get model action (greedy)
            with torch.no_grad():
                # Extract customer and vehicle features
                num_customers = env.num_customers
                customer_features = obs[8:8 + num_customers * 8].reshape(num_customers, 8)
                vehicle_features = obs[8 + num_customers * 8:].reshape(env.num_vehicles, 4)
                
                # Convert to tensors
                customer_tensor = torch.FloatTensor(customer_features).unsqueeze(0)
                vehicle_tensor = torch.FloatTensor(vehicle_features[env.current_vehicle]).unsqueeze(0)
                
                # Create mask for visited customers
                visited_mask = customer_features[:, -1] > 0.5
                mask = torch.BoolTensor(visited_mask).unsqueeze(0)
                
                # Get action
                action, _ = model.sample_action(
                    customer_tensor,
                    vehicle_tensor,
                    mask,
                    greedy=True
                )
                action = action.item()
            
            # Take step
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            step += 1
            
            # Render if requested
            if args.render:
                env.render()
        
        # Episode summary
        total_reward += episode_reward
        print(f"Episode {episode+1}/{args.episodes} | Reward: {episode_reward:.2f} | Steps: {step}")
    
    # Final results
    print("\n" + "="*70)
    print(f"EVALUATION COMPLETE")
    print(f"Average Reward: {total_reward/args.episodes:.2f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    evaluate()
