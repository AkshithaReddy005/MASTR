"""
Enhanced Training script for MAAM on MVRPSTW
With improved monitoring and logging
"""
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.maam_model import MAAM
from env.mvrp_env import MVRPSTWEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values to tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.training_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # Log additional metrics every 10 steps
        if self.n_calls % 10 == 0:
            if len(self.training_rewards) > 0:
                self.logger.record('train/mean_reward', np.mean(self.training_rewards[-100:]))
                self.logger.record('train/mean_episode_length', np.mean(self.episode_lengths[-100:]))
            self.logger.record('time/steps_per_second', int(self.num_timesteps / (time.time() - self.start_time)))
        return True
    
    def on_rollout_end(self) -> None:
        # Log at the end of each rollout
        if len(self.training_rewards) > 0:
            self.logger.record('rollout/ep_rew_mean', np.mean(self.training_rewards[-10:]))
            self.logger.record('rollout/ep_len_mean', np.mean(self.episode_lengths[-10:]))
            self.logger.record('time/total_steps', self.num_timesteps)
            self.logger.record('time/elapsed_time', int(time.time() - self.start_time))

class MAAMTrainer:
    def __init__(
        self,
        model: MAAM,
        env: MVRPSTWEnv,
        training_config: dict = None,
        model_config: dict = None,
        input_dim: int = 8,
        log_dir: str = 'logs/maam'
    ):
        # Store configurations
        self.model = model
        self.env = env
        self.training_config = training_config or {}
        self.model_config = model_config or {}
        self.input_dim = input_dim
        self.log_dir = log_dir
        self.global_step = 0
        
        # Set up logging
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Initialize target network (optional, for future use)
        self.target_net = None
        
        # Training state
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.no_improve_epochs = 0
        self.gradient_accumulation_steps = 4
        
        # Initialize model and move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize optimizer with weight decay and gradient clipping
        self.optimizer = torch.optim.AdamW(
            [
                {'params': [p for n, p in self.model.named_parameters() if p.requires_grad]},
            ],
            lr=1e-4,
            weight_decay=1e-5,
            eps=1e-8  # For numerical stability
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Logging to: {os.path.abspath(self.log_dir)}")
        
        try:
            # Training loop
            for step in range(1, self.training_config['total_steps'] + 1):
                # Reset environment at the start of each episode
                if step == 1:
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                    
                # Parse observation into customer features and vehicle state
                customer_features, vehicle_state = self._parse_observation(obs)
                
                # Create mask for visited customers
                mask = self._get_mask(obs)
                
                # Get action from model
                with torch.no_grad():
                    action, _ = self.model.sample_action(
                        customer_features,
                        vehicle_state,
                        mask,
                        greedy=False
                    )
                
                # Take a step in the environment
                next_obs, reward, done, truncated, info = self.env.step(action.item())
                
                # Store transition in buffer (if using experience replay)
                # self.replay_buffer.add(obs, action, reward, next_obs, done)
                
                # Update episode stats
                episode_reward += reward
                episode_steps += 1
                
                # Calculate loss
                loss = self.model.calculate_loss(
                    customer_features,
                    vehicle_state,
                    mask,
                    reward
                )
                
                # Normalize loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with gradient accumulation
                loss.backward()
                
                # Update weights every gradient_accumulation_steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients for stability
                    if 'gradient_clip' in self.training_config:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.training_config['gradient_clip']
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Check if episode is done
                if done or truncated:
                    # Log episode completion
                    if step % self.training_config['log_interval'] == 0 or True:
                        print(f"Episode completed at step {step} | "
                              f"Episode Reward: {episode_reward:.2f} | "
                              f"Episode Steps: {episode_steps}")
                    
                    # Reset environment for next episode
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                else:
                    # Update observation
                    obs = next_obs
                
                # Log progress
                if step % self.training_config['log_interval'] == 0:
                    # Print to console
                    print(f"Step {step}/{self.training_config['total_steps']} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Episode Reward: {episode_reward:.2f} | "
                          f"Episode Steps: {episode_steps}")
                    
                    # Log to TensorBoard
                    self.writer.add_scalar('train/loss', loss.item(), step)
                    self.writer.add_scalar('train/episode_reward', episode_reward, step)
                    self.writer.add_scalar('train/episode_length', episode_steps, step)
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], step)
                    
                    # Log gradients (optional, can be commented out if not needed)
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'grads/{name}', param.grad, step)
                    
                    # Log model weights (optional, can be commented out if not needed)
                    for name, param in self.model.named_parameters():
                        self.writer.add_histogram(f'weights/{name}', param, step)
                        
                # Save checkpoint and evaluate
                if step % self.training_config['checkpoint_interval'] == 0:
                    # Save checkpoint
                    self.save_checkpoint(step)
                    
                    # Evaluate model
                    eval_reward = self.evaluate()
                    print(f"Evaluation Reward: {eval_reward:.2f}")
                    
                    # Log evaluation metrics to TensorBoard
                    self.writer.add_scalar('eval/reward', eval_reward, step)
                    
                    # Reset environment after evaluation to ensure training can continue
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                    
                    # Early stopping based on evaluation reward
                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        self.no_improve_epochs = 0
                        # Save best model
                        self.save_checkpoint(step, is_best=True)
                        # Log best reward
                        self.writer.add_scalar('eval/best_reward', self.best_reward, step)
                    else:
                        self.no_improve_epochs += 1
                        
                    if self.no_improve_epochs >= self.training_config.get('patience', 10):
                        print(f"Early stopping at step {step} (no improvement for {self.no_improve_epochs} evaluations)")
                        self.writer.add_scalar('train/early_stopping', 1, step)
                        break
                    
                # Early stopping if loss is NaN or Inf
                if not torch.isfinite(loss):
                    print(f"Training stopped at step {step} due to non-finite loss")
                    break
                    
        except KeyboardInterrupt:
            print("Training interrupted by user")
        except Exception as e:
            print(f"Training error at step {step}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Save final model
        print("Training completed. Saving final model...")
        self.save_checkpoint(self.global_step, is_final=True)
        
    def evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate the model on the environment
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Average reward over evaluation episodes
        """
        self.model.eval()
        total_reward = 0.0
        
        with torch.no_grad():
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                done = False
                truncated = False
                episode_reward = 0.0
                
                while not (done or truncated):
                    # Parse observation into customer features and vehicle state
                    customer_features, vehicle_state = self._parse_observation(obs)
                    
                    # Create mask for visited customers
                    mask = self._get_mask(obs)
                    
                    # Get action from model (greedy for evaluation)
                    action, _ = self.model.sample_action(
                        customer_features,
                        vehicle_state,
                        mask,
                        greedy=True
                    )
                    
                    # Take a step in the environment
                    obs, reward, done, truncated, info = self.env.step(action.item())
                    episode_reward += reward
                
                total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        self.model.train()
        return avg_reward
    
    def _parse_observation(self, obs):
        """Parse flat observation into structured tensors"""
        # Get the actual environment (unwrap from Monitor)
        env = self.env.env if hasattr(self.env, 'env') else self.env
        
        # Observation format: [depot_features(8), customer_features(N*8), vehicle_states(M*4)]
        num_customers = env.num_customers
        num_vehicles = env.num_vehicles
        
        # Skip depot (first 8 features), extract only customer features
        customer_start_idx = 8
        customer_end_idx = 8 + num_customers * 8
        customer_flat = obs[customer_start_idx:customer_end_idx]
        customer_features = customer_flat.reshape(num_customers, 8)
        
        # Extract current vehicle state
        vehicle_start_idx = customer_end_idx
        vehicle_flat = obs[vehicle_start_idx:]
        vehicle_states = vehicle_flat.reshape(num_vehicles, 4)
        current_vehicle_state = vehicle_states[env.current_vehicle]
        
        # Convert to tensors and add batch dimension
        customer_features = torch.FloatTensor(customer_features).unsqueeze(0).to(self.device)
        vehicle_state = torch.FloatTensor(current_vehicle_state).unsqueeze(0).to(self.device)
        
        return customer_features, vehicle_state
    
    def _get_mask(self, obs):
        """Create mask for visited customers (True = mask out, False = available)"""
        # Get the actual environment (unwrap from Monitor)
        env = self.env.env if hasattr(self.env, 'env') else self.env
        
        num_customers = env.num_customers
        
        # Extract visited flags from customer features (skip depot at index 0)
        visited = []
        for i in range(num_customers):
            # Customer i is at position: depot(8) + i*8
            start_idx = 8 + i * 8
            visited_flag = obs[start_idx + 7]  # Last feature is visited flag
            visited.append(bool(visited_flag > 0.5))

        # Convert to tensor: True = visited (mask out), False = available
        mask = torch.tensor(visited, dtype=torch.bool, device=self.device).unsqueeze(0)
        return mask
        
    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        # Ensure checkpoint directory exists
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints/maam')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
            'training_config': self.training_config
        }
        
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, "maam_best.pt")
        elif is_final:
            checkpoint_path = os.path.join(checkpoint_dir, "maam_final.pt")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"maam_step_{step}.pt")
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        if is_final:
            # Save a copy as final_model.pt
            final_path = os.path.join(checkpoint_dir, 'final_model.pt')
            torch.save(checkpoint, final_path)
            print(f"Final model saved to {final_path}")
def main():
    # Environment configuration - Start with 50 customers for stable training
    env_config = {
        'num_customers': 50,   # Start with 50 customers for faster convergence
        'num_vehicles': 5,     # Proportional to customer count
        'vehicle_capacity': 200,  # From C101 dataset
        'data_path': 'data/raw/C101.csv',
        'max_time': 1236,  # From C101 dataset (20.6 hours in minutes)
        'penalty_early': 100,  # Reduced penalty for better reward scaling
        'penalty_late': 200,   # Reduced penalty for better reward scaling
        'seed': 42
    }
    
    # Model configuration - Optimized for stable training
    model_config = {
        'input_dim': 8,  # [x, y, demand, ready_time, due_time, service_time, is_depot, is_visited]
        'embed_dim': 128,  # Good balance of capacity and speed
        'num_heads': 8,    # Multi-head attention
        'num_encoder_layers': 2,  # Simpler architecture for stability
        'ff_dim': 256,     # Feed-forward dimension
        'dropout': 0.15,   # Increased dropout for regularization
        'tanh_clipping': 10.0,
        'init_gain': 0.05  # Slightly larger initialization
    }
    
    # Training configuration - Optimized for convergence
    training_config = {
        'total_steps': 20000,  # Reasonable number of steps
        'checkpoint_interval': 500,  # Frequent checkpoints
        'log_interval': 50,   # Frequent logging
        'checkpoint_dir': 'checkpoints/maam',
        'lr': 1e-4,  # Standard learning rate
        'batch_size': 16,  # Larger batch for stability
        'gamma': 0.99,
        'target_update_freq': 1000,
        'seed': 42,
        'verbose': 1,
        'patience': 20,  # More patience for convergence
        'gradient_clip': 0.5,  # Strict gradient clipping
        'warmup_steps': 1000  # Learning rate warmup
    }
    
    # Set random seeds for reproducibility
    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    random.seed(training_config['seed'])
    
    # Create environment
    env = MVRPSTWEnv(**env_config)
    
    # Wrap environment with Monitor for logging
    env = Monitor(env)
    
    # Create model
    model = MAAM(**model_config)
    
    # Print model summary
    print("Model architecture:")
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    trainer = MAAMTrainer(
        model=model,
        env=env,
        training_config=training_config,
        model_config=model_config,
        input_dim=model_config['input_dim'],
        log_dir=f"logs/maam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
