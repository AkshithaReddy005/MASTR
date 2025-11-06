"""
Improved Q-Learning Training Script with Proper RL Implementation
"""
import os
import numpy as np
from env.mvrp_env import MVRPSTWEnv
from qlearning_improved import ImprovedQLearningAgent

def get_solomon_instances():
    """Get list of all Solomon instance file paths."""
    data_dir = 'data'
    instances = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.TXT') or f.endswith('.txt')
    ]
    return instances

def train_improved_qlearning(num_episodes=3000, save_dir='checkpoints_qlearning_improved'):
    """
    Train Improved Q-Learning agent with enhanced curriculum learning and exploration
    
    Args:
        num_episodes: Number of training episodes
        save_dir: Directory to save checkpoints
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment with smaller problem size to start
    num_customers = 5
    num_vehicles = 2
    env = MVRPSTWEnv(
        num_customers=num_customers,
        num_vehicles=num_vehicles,
        vehicle_capacity=25.0,
        max_time=240.0,
        max_speed=5.0,  # Explicitly set max_speed
        seed=42
    )
    
    # Optimized agent parameters for effective learning
    agent = ImprovedQLearningAgent(
        learning_rate=0.5,      # Balanced learning rate
        discount_factor=0.95,   # Standard discount factor
        epsilon=1.0,            # Start with full exploration
        epsilon_min=0.05,       # Minimal exploration at the end
        epsilon_decay=0.995,    # Moderate decay of exploration
        alpha_decay=0.995,      # Moderate learning rate decay
        alpha_min=0.01,         # Minimum learning rate
        replay_buffer_size=10000 # Reasonable replay buffer
    )
    
    # Progressive curriculum - gradually increase difficulty
    curriculum_stages = [
        # Stage 1: Very simple (3 customers, 1 vehicle)
        {'min_episodes': 0, 'customers': 3, 'vehicles': 1, 'max_time': 500.0, 'instance': 'c101.txt'},
        
        # Stage 2: Add more customers
        {'min_episodes': 300, 'customers': 5, 'vehicles': 1, 'max_time': 800.0, 'instance': 'c101.txt'},
        
        # Stage 3: More customers
        {'min_episodes': 600, 'customers': 8, 'vehicles': 2, 'max_time': 1000.0, 'instance': 'c101.txt'},
        
        # Stage 4: Moderate problem
        {'min_episodes': 1000, 'customers': 10, 'vehicles': 2, 'max_time': 1200.0, 'instance': 'c101.txt'},
        
        # Stage 5: Larger problem
        {'min_episodes': 1500, 'customers': 15, 'vehicles': 3, 'max_time': 1500.0, 'instance': 'c101.txt'},
        
        # Stage 6: Full problem
        {'min_episodes': 2000, 'customers': 20, 'vehicles': 5, 'max_time': 2000.0, 'instance': 'c101.txt'}
    ]
    
    # Get all Solomon instances
    instances = get_solomon_instances()
    if not instances:
        raise FileNotFoundError("No Solomon instance files found in data/ directory")
    
    instances = sorted(instances, key=lambda x: int(os.path.basename(x).split('.')[0][1:]) 
                      if os.path.basename(x)[0].isdigit() else float('inf'))
    
    print(f"Found {len(instances)} Solomon instances (sorted by size):")
    for i, instance in enumerate(instances, 1):
        print(f"  {i}. {os.path.basename(instance)}")
    
    # Curriculum learning stages
    curriculum_stages = [
        {'min_episodes': 0, 'max_customers': 20, 'num_vehicles': 5},
        {'min_episodes': 200, 'max_customers': 50, 'num_vehicles': 10},
        {'min_episodes': 500, 'max_customers': 100, 'num_vehicles': 25},
    ]
    
    # Training statistics
    best_reward = -float('inf')
    best_customers_served = 0
    no_improve = 0
    max_no_improve = 50  # More patience for curriculum learning
    
    # Initialize agent with better parameters
    agent = ImprovedQLearningAgent(
        learning_rate=0.2,
        discount_factor=0.98,
        epsilon=1.0,
        epsilon_min=0.1,  # Keep some exploration
        epsilon_decay=0.999,
        alpha_decay=0.999,
        alpha_min=0.01
    )
    
    # Training tracking
    episode_rewards = []
    episode_steps = []
    episode_customers = []
    
    # Initialize tracking variables
    best_reward = -np.inf
    best_avg_reward = -np.inf
    best_customers_served = 0
    best_episode = 0
    no_improve = 0
    
    print("\nStarting training with curriculum learning...")
    print("-" * 80)
    
    for episode in range(1, num_episodes + 1):
        # Apply epsilon and alpha decay
        if episode > 1:
            agent.decay_epsilon()
            agent.decay_alpha()
        
        # Determine current curriculum stage
        current_stage = curriculum_stages[-1]  # Default to last stage
        for stage in curriculum_stages:
            if episode >= stage['min_episodes']:
                current_stage = stage
        
        # Select instance (cycle through them)
        instance_idx = (episode - 1) % len(instances)
        current_instance_path = instances[instance_idx]
        
        # Adjust number of customers based on curriculum
        num_customers = min(current_stage['max_customers'], 100)  # Cap at 100
        num_vehicles = current_stage['num_vehicles']
        
        # Reset environment with current instance and curriculum settings
        env = MVRPSTWEnv(
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            vehicle_capacity=200,  # Increased capacity
            data_path=current_instance_path,
            penalty_early=0.3,     # Reduced penalties
            penalty_late=0.5
        )
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # Episode loop
        while not done:
            # Get state features for the agent
            state_features = agent.get_state_features(state, env)
            
            # Get action from agent with exploration
            action = agent.get_action(state_features, state, env, training=True)
            
            # If no valid action, break the episode
            if action is None:
                break
                
            # Take action
            next_state, env_reward, done, truncated, info = env.step(action)
            
            # Get next state features for the agent
            next_state_features = agent.get_state_features(next_state, env)
            
            # Compute shaped reward
            shaped_reward = agent.calculate_reward(state, next_state, env, env_reward)
            
            # Update Q-values using the experience
            agent.update(
                state=state_features,
                action=action,
                reward=shaped_reward,
                next_state=next_state_features,
                done=done,
                next_obs=next_state,
                env=env
            )
            
            state = next_state
            total_reward += shaped_reward
            steps += 1
            
            if done or truncated:
                break
        
        # Track statistics
        customers_served = np.sum(env.visited)
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_customers.append(customers_served)
        
        # Calculate moving averages for better progress tracking
        window = min(10, episode)  # Use a window of 10 or the number of episodes so far
        avg_reward = np.mean(episode_rewards[-window:])
        avg_customers = np.mean(episode_customers[-window:])
        
        # Update best model based on both reward and customers served
        improvement = False
        if (avg_reward > best_reward * 1.01 or  # At least 1% improvement
            (np.abs(avg_reward - best_reward) < 1e-6 and avg_customers > best_customers_served)):
            best_reward = avg_reward
            best_customers_served = avg_customers
            agent.save(os.path.join(save_dir, 'best_agent.pkl'))
            no_improve = 0
            improvement = True
            
            # Save detailed info about the best model
            with open(os.path.join(save_dir, 'best_model_info.txt'), 'w') as f:
                f.write(f'Episode: {episode}\n')
                f.write(f'Average Reward: {avg_reward:.2f}\n')
                f.write(f'Average Customers Served: {avg_customers:.1f}\n')
                f.write(f'Epsilon: {agent.epsilon:.4f}\n')
                f.write(f'Learning Rate: {agent.alpha:.6f}\n')
        else:
            no_improve += 1
            
        # Save checkpoint every 50 episodes
        if episode % 50 == 0 or episode == 1:
            agent.save(os.path.join(save_dir, f'checkpoint_ep{episode}.pkl'))
            
            # Save training statistics
            stats = {
                'episode': episode,
                'rewards': episode_rewards,
                'customers_served': episode_customers,
                'steps': episode_steps,
                'epsilon': agent.epsilon,
                'alpha': agent.alpha,
                'best_reward': best_reward,
                'best_customers': best_customers_served
            }
            np.save(os.path.join(save_dir, 'training_stats.npy'), stats)
            
        # Print progress every episode for better monitoring
        if episode % 5 == 0 or improvement or episode == 1:
            stage_info = f"Stage: {current_stage['max_customers']}c/{current_stage['num_vehicles']}v"
            improve_flag = "*" if improvement else " "
            
            print(f"Episode {episode:4d}{improve_flag} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Cust: {avg_customers:3.1f}/{num_customers} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"LR: {agent.alpha:.4f} | "
                  f"{stage_info} | "
                  f"Inst: {os.path.basename(current_instance_path)[:8]}...")
        
        # Early stopping only if no improvement for a very long time
        if episode >= 500 and no_improve >= 500:  # No improvement for 500 episodes
            print(f"\nEarly stopping at episode {episode} - no improvement for 500 episodes")
            print(f"Best average reward: {best_reward:.2f}, Best customers served: {best_customers_served}")
            break
            
        # Update best average reward and episode
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_episode = episode
    
    # Save final model
    agent.save(os.path.join(save_dir, 'final_agent.pkl'))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Best customers served: {best_customers_served}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print("="*80)
    
    return agent, episode_rewards, episode_customers

def main():
    """Main training function."""
    train_improved_qlearning(num_episodes=3000)

if __name__ == "__main__":
    main()
