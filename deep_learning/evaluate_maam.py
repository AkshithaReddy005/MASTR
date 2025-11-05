"""
Evaluation script for trained MAAM model on MVRPSTW
"""
import os
import sys
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model.maam_model import MAAM
from env.mvrp_env import MVRPSTWEnv

def safe_load_model(model_path):
    """Safely load a PyTorch model with compatibility for different versions"""
    try:
        # For PyTorch < 2.6
        return torch.load(model_path, map_location='cpu')
    except TypeError:
        try:
            # For PyTorch 2.6+ with weights_only=False
            return torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative loading method...")
            # Last resort - load just the state dict
            checkpoint = {'model_state_dict': torch.load(model_path, map_location='cpu', weights_only=False)['model_state_dict']}
            return checkpoint

def evaluate_model(model_path: str, num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained MAAM model
    
    Args:
        model_path: Path to the trained model checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    # Load the model
    checkpoint = safe_load_model(model_path)
    
    # Extract configuration
    model_config = checkpoint.get('model_config', {})
    env_config = checkpoint.get('env_config', {})
    
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    print(f"Model config: {model_config}")
    print(f"Environment config: {env_config}")
    
    # Create environment
    env = MVRPSTWEnv(**env_config)
    
    # Create model
    model = MAAM(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluation metrics
    total_rewards = []
    total_distances = []
    episode_lengths = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    
    with torch.no_grad():
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            
            while not (done or truncated):
                # Parse observation
                depot_features = obs[:8]
                num_customers = env.num_customers
                customer_features = obs[8:8 + num_customers * 8].reshape(num_customers, 8)
                vehicle_features = obs[8 + num_customers * 8:].reshape(env.num_vehicles, 4)
                
                # Convert to tensors
                customer_features_tensor = torch.FloatTensor(customer_features).unsqueeze(0)
                vehicle_state_tensor = torch.FloatTensor(vehicle_features[env.current_vehicle]).unsqueeze(0)
                
                # Create mask for visited customers
                visited_mask = customer_features[:, -1] > 0.5
                mask = torch.BoolTensor(visited_mask).unsqueeze(0)
                
                # Get action (greedy)
                action, _ = model.sample_action(
                    customer_features_tensor,
                    vehicle_state_tensor,
                    mask,
                    greedy=True
                )
                
                # Take step
                obs, reward, done, truncated, info = env.step(action.item())
                episode_reward += reward
                episode_steps += 1
            
            # Calculate total distance
            total_distance = env.get_solution_cost()
            
            total_rewards.append(episode_reward)
            total_distances.append(total_distance)
            episode_lengths.append(episode_steps)
            
            print(f"Episode {ep+1}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Distance: {total_distance:.2f} | "
                  f"Steps: {episode_steps}")
            
            # Render if requested
            if render and ep == 0:
                env.render()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Distance: {np.mean(total_distances):.2f} ± {np.std(total_distances):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Best Reward: {np.max(total_rewards):.2f}")
    print(f"Worst Reward: {np.min(total_rewards):.2f}")
    print("="*60)
    
    return {
        'rewards': total_rewards,
        'distances': total_distances,
        'episode_lengths': episode_lengths,
        'mean_reward': np.mean(total_rewards),
        'mean_distance': np.mean(total_distances)
    }

def compare_models(model_paths: list, num_episodes: int = 10):
    """
    Compare multiple trained models
    
    Args:
        model_paths: List of paths to model checkpoints
        num_episodes: Number of episodes to evaluate each model
    """
    results = {}
    
    for i, path in enumerate(model_paths):
        print(f"\n{'='*60}")
        print(f"Evaluating Model {i+1}: {path}")
        print(f"{'='*60}")
        
        results[path] = evaluate_model(path, num_episodes, render=False)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rewards comparison
    model_names = [f"Model {i+1}" for i in range(len(model_paths))]
    rewards_data = [results[path]['rewards'] for path in model_paths]
    
    axes[0].boxplot(rewards_data, labels=model_names)
    axes[0].set_title('Reward Distribution')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True)
    
    # Distances comparison
    distances_data = [results[path]['distances'] for path in model_paths]
    
    axes[1].boxplot(distances_data, labels=model_names)
    axes[1].set_title('Distance Distribution')
    axes[1].set_ylabel('Total Distance')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print(f"\nComparison plot saved to model_comparison.png")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MAAM model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--compare', nargs='+', help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.episodes)
    else:
        evaluate_model(args.model, args.episodes, args.render)
