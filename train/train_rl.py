"""
Training script for MAAM on MVRPSTW
Uses REINFORCE algorithm with baseline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Initialize tracking variables
        self.episode_counter = 0
        self.best_reward = -float('inf')
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Baseline (value function) for variance reduction
        embed_dim = model.embed_dim
        self.baseline = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        ).to(device)
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=baseline_lr)
        
        # Logging
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs/maam_training')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
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
        
        # Skip depot (first 8 features), extract only customer features
        customer_start_idx = 8
        customer_end_idx = 8 + num_customers * 8
        customer_flat = obs[customer_start_idx:customer_end_idx]
        customer_features = customer_flat.reshape(num_customers, 8)
        
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
        """Create mask for visited customers (True = mask out, False = available)"""
        num_customers = self.env.num_customers
        
        # Extract visited flags from customer features only (skip depot at index 0)
        visited = []
        for i in range(num_customers):
            # Customer i is at position: depot(8) + i*8
            start_idx = 8 + i * 8
            visited_flag = obs[start_idx + 7]  # Last feature is visited flag
            # Cast to Python bool to avoid numpy.bool_ deprecation
            visited.append(bool(visited_flag > 0.5))

        # Convert to tensor: True = visited (mask out), False = available
        mask = torch.tensor(visited, dtype=torch.bool, device=self.device).unsqueeze(0)
        return mask
    
    def compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns"""
        returns = []
        R = 0
        if not isinstance(rewards, list):
            rewards = [rewards] if not isinstance(rewards, (list, tuple)) else rewards
            
        for r in reversed(rewards):
            if not isinstance(r, (int, float)):
                r = r.item() if hasattr(r, 'item') else float(r)
            R = r + gamma * R
            returns.insert(0, R)
            
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns_tensor.unsqueeze(-1) if len(returns_tensor.shape) == 1 else returns_tensor
    
    def train_step(self, num_episodes: int = 32):
        """
        Perform one training step with multiple episodes
        """
        self.model.train()
        
        # Collect data with current policy
        all_log_probs = []
        all_returns = []
        all_states = []
        episode_rewards = []
        
        # Phase 1: Collect rollouts
        for _ in range(num_episodes):
            # Collect episode
            total_reward, log_probs, states, rewards = self.rollout()
            episode_rewards.append(total_reward)
            
            # Store data
            all_log_probs.extend(log_probs)
            all_states.extend(states)
            
            # Compute and store returns
            returns = self.compute_returns(rewards)
            all_returns.append(returns)
        
        # Convert to tensors
        all_log_probs = torch.stack(all_log_probs) if all_log_probs else torch.tensor([], device=self.device)
        all_returns = torch.cat(all_returns) if all_returns else torch.tensor([], device=self.device)
        
        # Phase 2: Compute baselines (no gradients)
        with torch.no_grad():
            baselines = []
            for customer_features, vehicle_state in all_states:
                customer_tensor = customer_features  # already a tensor on device
                vehicle_tensor = vehicle_state       # already a tensor on device
                
                # Forward pass through model (no gradients)
                _, encoder_output = self.model(customer_tensor, vehicle_tensor)
                state_repr = encoder_output.mean(dim=1)
                baseline_value = self.baseline(state_repr)
                baselines.append(baseline_value.squeeze(-1))
            
            baselines = torch.cat(baselines) if baselines else torch.zeros_like(all_returns)
        
        # Ensure shapes match
        if len(baselines.shape) > 1:
            baselines = baselines.squeeze(-1)
        if len(all_returns.shape) > 1:
            all_returns = all_returns.squeeze(-1)
        
        # Normalize returns
        if len(all_returns) > 1:
            returns_mean = all_returns.mean()
            returns_std = all_returns.std() + 1e-8
            normalized_returns = (all_returns - returns_mean) / returns_std
        else:
            normalized_returns = all_returns
        
        # Compute advantages
        advantages = normalized_returns - baselines.detach()
        
        # Phase 3: Update policy
        self.optimizer.zero_grad()
        policy_loss = -(all_log_probs * advantages).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Phase 4: Update baseline
        self.baseline_optimizer.zero_grad()
        
        # Recompute baselines with updated policy (with gradients)
        baseline_preds = []
        for customer_features, vehicle_state in all_states:
            customer_tensor = customer_features  # already a tensor on device
            vehicle_tensor = vehicle_state       # already a tensor on device
            
            with torch.no_grad():
                _, encoder_output = self.model(customer_tensor, vehicle_tensor)
                state_repr = encoder_output.mean(dim=1)
            
            baseline_pred = self.baseline(state_repr).squeeze(-1)
            baseline_preds.append(baseline_pred)
        
        baseline_preds = torch.cat(baseline_preds) if baseline_preds else torch.tensor([], device=self.device)
        baseline_loss = F.mse_loss(baseline_preds, normalized_returns.detach())
        baseline_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.baseline.parameters(), max_norm=1.0)
        self.baseline_optimizer.step()
        
        # Calculate average reward
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        # Update best model
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.save_checkpoint('best_model.pt')
        
        # Print progress
        print(f"Episode: {self.episode_counter}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Policy Loss: {policy_loss.item():.4f}")
        print(f"  Baseline Loss: {baseline_loss.item():.4f}")
        
        self.episode_counter += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'avg_reward': avg_reward,
            'avg_cost': -avg_reward
        }
    
    def update_policy(self, log_probs, rewards, states):
        # Ensure log_probs is a tensor
        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.stack(log_probs)
        
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Compute baselines (need gradients for baseline loss)
        baselines = []
        for customer_features, vehicle_state in states:
            # Ensure tensors are on correct device
            customer_features = torch.FloatTensor(customer_features).to(self.device)
            vehicle_state = torch.FloatTensor(vehicle_state).to(self.device)
            
            # Use encoder output as state representation
            with torch.no_grad():
                # Make sure to detach the computation graph
                customer_features = customer_features.detach()
                vehicle_state = vehicle_state.detach()
                
                # Forward pass through model
                _, encoder_output = self.model(customer_features, vehicle_state)
                state_repr = encoder_output.mean(dim=1)  # [B, embed_dim]
                
                # Forward through baseline network
                with torch.no_grad():
                    baseline_value = self.baseline(state_repr)  # [B, 1]
                baselines.append(baseline_value.squeeze(-1))  # [B]
        
        baselines = torch.cat(baselines) if baselines else torch.zeros_like(returns)
        
        # Ensure shapes match
        if len(baselines.shape) > 1:
            baselines = baselines.squeeze(-1)
        
        # Normalize returns
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        normalized_returns = (returns - returns_mean) / returns_std
        
        # Compute advantages
        advantages = normalized_returns - baselines.detach()
        
        # Ensure shapes match for multiplication
        if len(advantages.shape) > 1:
            advantages = advantages.squeeze(-1)
        
        # Policy loss (REINFORCE)
        policy_loss = -(log_probs * advantages).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return policy_loss.item()
    
    def train(self, num_iterations: int = 100, episodes_per_iter: int = 32, eval_interval: int = 50):
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
                eval_metrics = self.evaluate(num_episodes=3)
                
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
                    print(f"  New best model saved!")
        
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
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Join with the provided filename
        full_path = os.path.join(checkpoint_dir, os.path.basename(path))
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'baseline_state_dict': self.baseline.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline_optimizer_state_dict': self.baseline_optimizer.state_dict(),
        }, full_path)
        
        print(f"\nModel saved to {full_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        # If path is not absolute, assume it's in the checkpoints directory
        if not os.path.isabs(path):
            checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
            path = os.path.join(checkpoint_dir, os.path.basename(path))
        
        print(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])
        
        # Only load optimizer states if they exist in the checkpoint
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'baseline_optimizer_state_dict' in checkpoint:
            self.baseline_optimizer.load_state_dict(checkpoint['baseline_optimizer_state_dict'])
            
        print("Model loaded successfully")


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Training parameters (FAST MODE for 100 customers)
    # Increase later for higher accuracy (e.g., 200-500 iters, 8-16 episodes)
    num_iterations = 60     # Fast training
    episodes_per_iter = 2   # Small batch per update
    eval_interval = 10      # Evaluate every 10 iterations
    embed_dim = 128
    num_heads = 8
    num_encoder_layers = 3
    
    # Path to C101 dataset
    data_path = os.path.join("data", "raw", "C101.csv")
    
    # Ensure data path exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the C101.csv file is in the correct location.")
        return
    
    print(f"Using data from: {data_path}")
    
    try:
        # Create environment with 100 customers from the CSV
        env = MVRPSTWEnv(
            num_customers=100,  # Use all customers
            num_vehicles=10,    # More vehicles for larger instance
            vehicle_capacity=200.0,
            grid_size=100.0,
            data_path=data_path,
            max_time=1236.0,
            penalty_early=1.0,
            penalty_late=2.0
        )
        
        # Create model
        model = MAAM(
            input_dim=8,  # 8 features per customer
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            ff_dim=512,
            dropout=0.1
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} trainable parameters")
        
        # Create trainer
        trainer = REINFORCETrainer(
            model=model,
            env=env,
            learning_rate=1e-4,
            baseline_lr=1e-3,
            gamma=0.99
        )
        
        # Create checkpoints directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Train
        print("\nStarting training...")
        trainer.train(
            num_iterations=num_iterations,
            episodes_per_iter=episodes_per_iter,
            eval_interval=eval_interval
        )
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()


def visualize_training():
    """Visualize training progress using TensorBoard"""
    print("\nTo visualize training progress, run in a new terminal:")
    print("tensorboard --logdir=runs/")
    print("Then open http://localhost:6006 in your browser")

if __name__ == "__main__":
    try:
        main()
        visualize_training()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        # Create a dummy trainer to save the model
        try:
            trainer.save_checkpoint('interrupted_model.pt')
        except NameError:
            print("Could not save model: trainer not initialized")
        visualize_training()
