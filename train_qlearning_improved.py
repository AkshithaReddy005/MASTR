"""
Improved Q-Learning Training Script with Proper RL Implementation
"""
import os
import numpy as np
from env.mvrp_env import MVRPSTWEnv
from qlearning_improved import ImprovedQLearningAgent

def train_improved_qlearning(num_episodes=1000, save_dir='checkpoints_qlearning_improved'):
    """
    Train Improved Q-Learning agent with proper RL implementation
    
    Args:
        num_episodes: Number of training episodes
        save_dir: Directory to save checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*80)
    print("IMPROVED Q-LEARNING VRP TRAINING")
    print("="*80)
    print(f"Episodes: {num_episodes}")
    print("\nHyperparameters:")
    print("  Learning Rate (alpha): 0.1 (with decay)")
    print("  Discount Factor (gamma): 0.95")
    print("  Initial Epsilon: 1.0")
    print("  Min Epsilon: 0.01")
    print("  Epsilon Decay: 0.995")
    print("  Alpha Decay: 0.9995")
    print("="*80)
    
    # Create environment with real data
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=100.0,
        data_path='data/raw/C101.csv'
    )
    
    # Create improved agent with optimal hyperparameters
    agent = ImprovedQLearningAgent(
        learning_rate=0.1,      # Initial learning rate
        discount_factor=0.95,   # Balance between immediate and future rewards
        epsilon=1.0,            # Start with full exploration
        epsilon_min=0.01,       # Minimum exploration
        epsilon_decay=0.995,    # Gradual decay
        alpha_decay=0.9995,     # Learning rate decay
        alpha_min=0.01          # Minimum learning rate
    )
    
    print("\nStarting training...")
    print("-" * 80)
    
    # Training tracking
    episode_rewards = []
    episode_steps = []
    episode_customers = []
    best_reward = float('-inf')
    no_improvement = 0
    patience = 50  # Stop if no improvement for 50 episodes
    
    # For visualization
    window_size = 10
    moving_avg = []
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        episode_step = 0
        customers_visited = 0
        done = False
        
        state = agent.get_state_features(obs, env)
        
        while not done:
            # Select action
            action = agent.get_action(state, obs, env, training=True)
            
            if action is None:
                break
            
            # Take action in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Get next state
            next_state = agent.get_state_features(next_obs, env)
            
            # Update Q-values (proper RL: use environment reward directly)
            agent.update(state, action, reward, next_state, done, next_obs, env)
            
            # Track metrics
            episode_reward += reward
            episode_step += 1
            customers_visited += 1
            
            # Move to next state
            obs = next_obs
            state = next_state
        
        # Decay exploration and learning rate
        agent.decay_epsilon()
        agent.decay_alpha()
        agent.episode_count += 1
        
        # Track episode statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        episode_customers.append(customers_visited)
        
        # Calculate moving average
        if len(episode_rewards) >= window_size:
            moving_avg.append(np.mean(episode_rewards[-window_size:]))
        else:
            moving_avg.append(np.mean(episode_rewards))
        
        # Save best agent
        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improvement = 0
            agent.save(os.path.join(save_dir, 'best_agent.pkl'))
            print(f"\nNew best reward: {best_reward:.2f} at episode {episode}")
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= patience and episode > 100:  # Ensure minimum 100 episodes
            print(f"\nEarly stopping at episode {episode} - no improvement for {patience} episodes")
            break
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_steps = np.mean(episode_steps[-10:])
            avg_customers = np.mean(episode_customers[-10:])
            
            # ANSI escape codes for colored output
            R = '\033[91m'  # Red
            G = '\033[92m'  # Green
            Y = '\033[93m'  # Yellow
            B = '\033[94m'  # Blue
            C = '\033[0m'   # Reset
            
            # Color code reward based on performance
            reward_color = G if avg_reward > 0 else R if avg_reward < -5 else Y
            
            print(f"{B}Episode {episode:4d} | "
                  f"{reward_color}Reward: {avg_reward:7.2f}{C} | "
                  f"Steps: {avg_steps:4.1f} | "
                  f"Customers: {avg_customers:3.1f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Alpha: {agent.alpha:.4f} | "
                  f"Best: {best_reward:7.2f}{C}")
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Final alpha: {agent.alpha:.4f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print(f"Total updates: {agent.total_updates}")
    print("="*80 + "\n")
    
    # Save final agent
    agent.save(os.path.join(save_dir, 'final_agent.pkl'))
    
    # Save training statistics
    stats = {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'episode_customers': episode_customers,
        'moving_avg': moving_avg,
        'best_reward': best_reward,
        'final_epsilon': agent.epsilon,
        'final_alpha': agent.alpha,
        'q_table_size': len(agent.q_table),
        'total_updates': agent.total_updates
    }
    
    # Save as both numpy and json for flexibility
    np.save(os.path.join(save_dir, 'training_stats.npy'), stats)
    
    # Also save a human-readable summary
    with open(os.path.join(save_dir, 'training_summary.txt'), 'w') as f:
        f.write("Training Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Total episodes: {len(episode_rewards)}\n")
        f.write(f"Best reward: {best_reward:.2f}\n")
        f.write(f"Final epsilon: {agent.epsilon:.4f}\n")
        f.write(f"Final alpha: {agent.alpha:.6f}\n")
        f.write(f"Q-table size: {len(agent.q_table)} states\n")
        f.write(f"Total updates: {agent.total_updates}\n\n")
        
        # Last 10 episodes summary
        f.write("\nLast 10 episodes:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Episode':<10}{'Reward':<15}{'Steps':<10}{'Customers'}\n")
        for i in range(max(0, len(episode_rewards)-10), len(episode_rewards)):
            f.write(f"{i+1:<10}{episode_rewards[i]:<15.2f}{episode_steps[i]:<10}{episode_customers[i]}\n")
    
    print(f"\nTraining statistics saved to {save_dir}/training_stats.npy")
    print("="*80)
    print("Training complete! Use evaluate_qlearning_improved.py to test the agent.")
    print("="*80)

def main():
    """Main training function"""
    train_improved_qlearning(num_episodes=1000)

if __name__ == "__main__":
    main()
