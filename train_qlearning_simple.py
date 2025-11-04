"""
Simplified Q-Learning Training Script for VRP
"""
import os
import numpy as np
from env.mvrp_env import MVRPSTWEnv
from qlearning_simple import SimpleQLearningAgent

def train_simple_qlearning(num_episodes=500, penalty_config=None, save_dir='checkpoints_qlearning_simple'):
    """
    Train Simple Q-Learning agent on VRP
    
    Args:
        num_episodes: Number of training episodes
        penalty_config: Dictionary with penalty values
        save_dir: Directory to save checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Default penalty configuration
    if penalty_config is None:
        penalty_config = {
            'distance_weight': 1.0,
            'early_penalty': 50.0,
            'late_penalty': 100.0,
            'capacity_penalty': 200.0
        }
    
    print("="*80)
    print("SIMPLE Q-LEARNING VRP TRAINING")
    print("="*80)
    print(f"Episodes: {num_episodes}")
    print(f"Penalty Configuration:")
    for key, value in penalty_config.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Create environment
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=200,
        data_path='data/raw/C101.csv',
        max_time=1236,
        seed=42
    )
    
    # Create agent
    agent = SimpleQLearningAgent(
        penalty_config=penalty_config,
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=0.3
    )
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_customers_served = []
    best_reward = -float('inf')
    
    print("\nStarting training...")
    print("-"*80)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.get_state_features(obs, env)
        
        episode_reward = 0.0
        steps = 0
        done = False
        max_steps = 100
        
        while not done and steps < max_steps:
            # Get action
            action = agent.get_action(state, obs, env, training=True)
            
            if action is None:
                # No valid actions available
                break
            
            # Take action
            next_obs, env_reward, done, truncated, info = env.step(action)
            
            # Calculate custom reward
            reward = agent.calculate_reward(obs, action, next_obs, env, env_reward)
            
            # Get next state
            next_state = agent.get_state_features(next_obs, env)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done, next_obs, env)
            
            # Update tracking variables
            obs = next_obs
            state = next_state
            episode_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        # Decay exploration rate
        agent.decay_epsilon()
        agent.episode_count += 1
        
        # Count customers served
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        customers_served = int(np.sum(customers_data[:, -1] > 0.5))
        
        # Track metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_customers_served.append(customers_served)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = os.path.join(save_dir, 'best_agent.pkl')
            agent.save(save_path)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_steps = np.mean(episode_steps[-50:])
            avg_customers = np.mean(episode_customers_served[-50:])
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Avg Steps: {avg_steps:5.1f} | "
                  f"Avg Customers: {avg_customers:4.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Best: {best_reward:8.2f}")
    
    print("-"*80)
    print("\nTraining completed!")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print(f"Total updates: {agent.total_updates}")
    
    # Save final agent
    final_path = os.path.join(save_dir, 'final_agent.pkl')
    agent.save(final_path)
    
    # Save training statistics
    stats = {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'episode_customers_served': episode_customers_served,
        'best_reward': best_reward,
        'penalty_config': penalty_config
    }
    
    stats_path = os.path.join(save_dir, 'training_stats.npy')
    np.save(stats_path, stats)
    print(f"\nTraining statistics saved to {stats_path}")
    
    return agent, stats

def main():
    """Main training function with default parameters"""
    
    # Default penalty configuration
    penalty_config = {
        'distance_weight': 1.0,
        'early_penalty': 50.0,
        'late_penalty': 100.0,
        'capacity_penalty': 200.0
    }
    
    # Train agent
    agent, stats = train_simple_qlearning(
        num_episodes=500,
        penalty_config=penalty_config,
        save_dir='checkpoints_qlearning_simple'
    )
    
    print("\n" + "="*80)
    print("Training complete! Use evaluate_qlearning_simple.py to test the agent.")
    print("="*80)

if __name__ == "__main__":
    main()
