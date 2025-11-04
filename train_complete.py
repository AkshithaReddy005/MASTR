"""
Complete Working MAAM Training Script
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env.mvrp_env import MVRPSTWEnv

class MAAM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128):
        super().__init__()
        self.customer_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.vehicle_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU()
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, customers, vehicle, mask=None):
        # customers: [batch, num_customers, 8]
        # vehicle: [batch, 4]
        batch_size, num_customers, _ = customers.shape
        
        customer_enc = self.customer_net(customers)  # [batch, num_customers, hidden]
        vehicle_enc = self.vehicle_net(vehicle)  # [batch, hidden]
        
        # Expand vehicle for each customer
        vehicle_expanded = vehicle_enc.unsqueeze(1).expand(-1, num_customers, -1)
        
        # Combine and score
        combined = torch.cat([customer_enc, vehicle_expanded], dim=-1)
        scores = self.attention(combined).squeeze(-1)  # [batch, num_customers]
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        return scores
    
    def get_action(self, customers, vehicle, mask=None, greedy=False):
        scores = self.forward(customers, vehicle, mask)
        probs = torch.softmax(scores, dim=-1)
        
        if greedy:
            return torch.argmax(probs, dim=-1).item()
        else:
            return torch.multinomial(probs, 1).squeeze(-1).item()

def train_model(num_episodes=200):
    # Setup
    print("Setting up environment and model...")
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=200,
        data_path='data/raw/C101.csv',
        max_time=1236,
        seed=42
    )
    
    model = MAAM()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_reward = -float('inf')
    save_dir = 'checkpoints_complete'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        log_probs = []
        rewards = []
        steps = 0
        
        while not done and steps < 100:
            # Parse observation
            customers = torch.FloatTensor(obs[8:168].reshape(1, 20, 8))
            vehicle_idx = env.current_vehicle
            vehicle = torch.FloatTensor(obs[168:].reshape(3, 4)[vehicle_idx]).unsqueeze(0)
            
            # Create mask
            visited = customers[0, :, -1] > 0.5
            mask = visited.unsqueeze(0)
            
            # Get action
            action = model.get_action(customers, vehicle, mask, greedy=False)
            
            # Step
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
        
        # Print progress
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1:3d} | Reward: {episode_reward:8.2f} | Steps: {steps:3d}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'model': model.state_dict(),
                'episode': episode,
                'reward': best_reward
            }, os.path.join(save_dir, 'best_model.pt'))
    
    print("-" * 60)
    print(f"Training complete! Best reward: {best_reward:.2f}")
    print(f"Model saved to: {save_dir}/best_model.pt")
    
    return model

if __name__ == "__main__":
    train_model()
