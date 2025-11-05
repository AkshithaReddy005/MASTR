"""
Evaluation Script for Enhanced MAAM Model
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from env.mvrp_env import MVRPSTWEnv
from train_enhanced import EnhancedRouter  # Import the same model class

def evaluate_model(model_path, num_episodes=10, render=False):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = EnhancedRouter(hidden_dim=128).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Create environment
    env = MVRPSTWEnv(
        num_customers=20,  # Match training config
        num_vehicles=3,
        vehicle_capacity=200,
        data_path='data/raw/C101.csv',
        max_time=1236,
        penalty_early=100,
        penalty_late=200,
        seed=42
    )
    
    # Run evaluation
    all_rewards = []
    all_distances = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        if render:
            env.render()
        
        while not done and episode_steps < 200:  # Match max_steps from training
            # Convert observation to tensors
            customer_features = torch.FloatTensor(
                obs[8:8+20*8].reshape(1, 20, 8)  # 20 customers * 8 features
            ).to(device)
            
            vehicle_state = torch.FloatTensor(
                obs[8+20*8:].reshape(3, 4)  # 3 vehicles * 4 features
            )[env.current_vehicle].unsqueeze(0).to(device)
            
            # Create mask for visited customers
            visited_mask = customer_features[0, :, -1] > 0.5
            mask = visited_mask.unsqueeze(0).unsqueeze(0).to(device)
            
            # Get greedy action (no exploration during evaluation)
            with torch.no_grad():
                action = model.get_action(customer_features, vehicle_state, mask, epsilon=0.0)
            
            # Take step
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if render:
                env.render()
        
        # Store results
        all_rewards.append(episode_reward)
        if 'total_distance' in info:
            all_distances.append(info['total_distance'])
        distance = info.get('total_distance', None)
        distance_str = f"{distance:.2f}" if distance is not None else 'N/A'
        print(f"Episode {episode+1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Distance: {distance_str} | "
              f"Steps: {episode_steps}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    if all_distances:
        print(f"Average Distance: {np.mean(all_distances):.2f} ± {np.std(all_distances):.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards, 'b.-')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    if all_distances:
        plt.subplot(1, 2, 2)
        plt.plot(all_distances, 'r.-')
        plt.title('Total Distance')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("\nSaved evaluation results to evaluation_results.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Enhanced MAAM Model')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.episodes, args.render)
