"""
Training script for MAAM on MVRPSTW
Uses REINFORCE algorithm with baseline
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.maam_model import MAAM
from env.mvrp_env import MVRPSTWEnv


class REINFORCETrainer:
    """
    REINFORCE trainer for MAAM
    Policy gradient with baseline for variance reduction
    """
    def __init__(
        self,
        model: MAAM,
        env: MVRPSTWEnv,
        learning_rate: float = 1e-4,
        baseline_lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.env = env
        self.device = device
        self.gamma = gamma
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Baseline (value function) for variance reduction
        self.baseline = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=baseline_lr)
        
        # Logging
        self.writer = SummaryWriter('runs/maam_training')
    
    def rollout(self, greedy: bool = False):
        """
        Perform one episode rollout
        
        Returns:
            total_reward: episode return
            log_probs: list of log probabilities
            states: list of states
            actions: list of actions
        """
        obs, _ = self.env.reset()
        done = False
        
        log_probs = []
        rewards = []
        states = []
        
        while not done:
            # Parse observation into customer features and vehicle state
            customer_features, vehicle_state = self._parse_observation(obs)
            
            # Create mask for visited customers
            mask = self._get_mask(obs)
            
            # Sample action
            with torch.no_grad():
                action, log_prob = self.model.sample_action(
                    customer_features,
                    vehicle_state,
                    mask,
                    greedy=greedy
                )
            
            action_np = action.item()
            
            # Step environment
            obs, reward, done, truncated, info = self.env.step(action_np)
            
            # Store
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append((customer_features, vehicle_state))
        
        total_reward = sum(rewards)
        
        return total_reward, log_probs, states, rewards
    
    def _parse_observation(self, obs):
        """Parse flat observation into structured tensors"""
        # Observation format: [depot_features(8), customer_features(N*8), vehicle_states(M*4)]
        num_customers = self.env.num_customers
        num_vehicles = self.env.num_vehicles
        
        # Extract customer features (including depot)
        customer_end_idx = 8 + num_customers * 8
        customer_flat = obs[:customer_end_idx]
        customer_features = customer_flat.reshape(num_customers + 1, 8)
        
        # Extract current vehicle state
        vehicle_start_idx = customer_end_idx
        vehicle_flat = obs[vehicle_start_idx:]
        vehicle_states = vehicle_flat.reshape(num_vehicles, 4)
        current_vehicle_state = vehicle_states[self.env.current_vehicle]
        
        # Convert to tensors
        customer_features = torch.FloatTensor(customer_features).unsqueeze(0).to(self.device)
        vehicle_state = torch.FloatTensor(current_vehicle_state).unsqueeze(0).to(self.device)
        
        return customer_features, vehicle_state
    
    def _get_mask(self, obs):
        """Create mask for visited customers"""
        num_customers = self.env.num_customers
        
        # Extract visited flags from observation
        visited = []
        for i in range(num_customers + 1):
            start_idx = i * 8
            visited_flag = obs[start_idx + 7]  # Last feature is visited flag
            visited.append(visited_flag > 0.5)
        
        mask = torch.BoolTensor(visited).unsqueeze(0).to(self.device)
        return mask
    
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        return returns
    
    def train_step(self, num_episodes: int = 32):
        """
        Perform one training step with multiple episodes
        """
        all_log_probs = []
        all_returns = []
        all_baselines = []
        episode_rewards = []
        
        # Collect episodes
        for _ in range(num_episodes):
            total_reward, log_probs, states, rewards = self.rollout(greedy=False)
            returns = self.compute_returns(rewards)
            
            # Compute baselines
            baselines = []
            for customer_features, vehicle_state in states:
                # Use encoder output as state representation
                with torch.no_grad():
                    _, encoder_output = self.model(customer_features, vehicle_state)
                    state_repr = encoder_output.mean(dim=1)  # [B, embed_dim]
                    baseline_value = self.baseline(state_repr).squeeze(-1)
                    baselines.append(baseline_value)
            
            baselines = torch.stack(baselines)
            
            all_log_probs.extend(log_probs)
            all_returns.append(returns)
            all_baselines.append(baselines)
            episode_rewards.append(total_reward)
        
        # Concatenate all episodes
        all_log_probs = torch.cat(all_log_probs)
        all_returns = torch.cat(all_returns)
        all_baselines = torch.cat(all_baselines)
        
        # Normalize returns
        returns_mean = all_returns.mean()
        returns_std = all_returns.std() + 1e-8
        normalized_returns = (all_returns - returns_mean) / returns_std
        
        # Compute advantages
        advantages = normalized_returns - all_baselines.detach()
        
        # Policy loss (REINFORCE)
        policy_loss = -(all_log_probs * advantages).mean()
        
        # Baseline loss
        baseline_loss = nn.MSELoss()(all_baselines, all_returns)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update baseline
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()
        
        # Metrics
        avg_reward = np.mean(episode_rewards)
        
        return {
            'policy_loss': policy_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'avg_reward': avg_reward,
            'avg_cost': -avg_reward
        }
    
    def train(self, num_iterations: int = 1000, episodes_per_iter: int = 32, eval_interval: int = 50):
        """
        Main training loop
        """
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_cost = float('inf')
        
        for iteration in tqdm(range(num_iterations), desc="Training"):
            # Training step
            metrics = self.train_step(num_episodes=episodes_per_iter)
            
            # Logging
            self.writer.add_scalar('Loss/Policy', metrics['policy_loss'], iteration)
            self.writer.add_scalar('Loss/Baseline', metrics['baseline_loss'], iteration)
            self.writer.add_scalar('Reward/Train', metrics['avg_reward'], iteration)
            self.writer.add_scalar('Cost/Train', metrics['avg_cost'], iteration)
            
            # Evaluation
            if (iteration + 1) % eval_interval == 0:
                eval_metrics = self.evaluate(num_episodes=10)
                
                self.writer.add_scalar('Cost/Eval', eval_metrics['avg_cost'], iteration)
                self.writer.add_scalar('Cost/Best', eval_metrics['best_cost'], iteration)
                
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print(f"  Train Cost: {metrics['avg_cost']:.2f}")
                print(f"  Eval Cost: {eval_metrics['avg_cost']:.2f}")
                print(f"  Best Cost: {eval_metrics['best_cost']:.2f}")
                
                # Save best model
                if eval_metrics['avg_cost'] < best_cost:
                    best_cost = eval_metrics['avg_cost']
                    self.save_checkpoint('checkpoints/best_model.pt')
                    print(f"  ✓ New best model saved!")
        
        self.writer.close()
        print("\nTraining complete!")
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate policy"""
        self.model.eval()
        
        costs = []
        for _ in range(num_episodes):
            total_reward, _, _, _ = self.rollout(greedy=True)
            costs.append(-total_reward)
        
        self.model.train()
        
        return {
            'avg_cost': np.mean(costs),
            'std_cost': np.std(costs),
            'best_cost': np.min(costs),
            'worst_cost': np.max(costs)
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'baseline_state_dict': self.baseline.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline_optimizer_state_dict': self.baseline_optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])


def main():
    """Main training function"""
    # Hyperparameters
    num_customers = 20
    num_vehicles = 3
    embed_dim = 128
    num_heads = 8
    num_encoder_layers = 3
    
    # Create environment
    env = MVRPSTWEnv(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=100.0,
        grid_size=100.0
    )
    
    # Create model
    model = MAAM(
        input_dim=8,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        ff_dim=512,
        dropout=0.1
    )
    
    # Create trainer
    trainer = REINFORCETrainer(
        model=model,
        env=env,
        learning_rate=1e-4,
        baseline_lr=1e-3,
        gamma=0.99
    )
    
    # Train
    trainer.train(
        num_iterations=1000,
        episodes_per_iter=32,
        eval_interval=50
    )


if __name__ == '__main__':
    main()
