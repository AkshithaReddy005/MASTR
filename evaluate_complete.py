"""
Complete Evaluation Script
"""
import torch
import numpy as np
from env.mvrp_env import MVRPSTWEnv
from train_complete import MAAM

def evaluate(model_path, num_episodes=5, render=False):
    print("Loading model and environment...")
    
    # Load model
    model = MAAM()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Training episode: {checkpoint['episode']}, Reward: {checkpoint['reward']:.2f}")
    
    # Create environment
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=200,
        data_path='data/raw/C101.csv',
        max_time=1236,
        seed=None  # Different seed for evaluation
    )
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("-" * 80)
    
    all_rewards = []
    all_steps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        if render:
            env.render()
        
        while not done and steps < 100:
            # Parse observation
            customers = torch.FloatTensor(obs[8:168].reshape(1, 20, 8))
            vehicle_idx = env.current_vehicle
            vehicle = torch.FloatTensor(obs[168:].reshape(3, 4)[vehicle_idx]).unsqueeze(0)
            
            # Create mask
            visited = customers[0, :, -1] > 0.5
            mask = visited.unsqueeze(0)
            
            # Get greedy action
            with torch.no_grad():
                action = model.get_action(customers, vehicle, mask, greedy=True)
            
            # Step
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        all_rewards.append(episode_reward)
        all_steps.append(steps)
        
        print(f"Episode {episode+1:2d} | Reward: {episode_reward:8.2f} | Steps: {steps:3d}")
    
    print("-" * 80)
    print(f"\nEvaluation Results:")
    print(f"  Average Reward: {np.mean(all_rewards):8.2f} ± {np.std(all_rewards):6.2f}")
    print(f"  Average Steps:  {np.mean(all_steps):8.2f} ± {np.std(all_steps):6.2f}")
    print(f"  Best Reward:    {np.max(all_rewards):8.2f}")
    print(f"  Worst Reward:   {np.min(all_rewards):8.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints_complete/best_model.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--render', action='store_true')
    
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.render)
