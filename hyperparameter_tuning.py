"""
Hyperparameter Tuning for Q-Learning VRP Agent
Tests different hyperparameter combinations to find optimal settings
"""
import os
import numpy as np
from env.mvrp_env import MVRPSTWEnv
from qlearning_improved import ImprovedQLearningAgent
import json
from datetime import datetime

def evaluate_agent(agent, env, num_episodes=20):
    """
    Evaluate agent performance
    
    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []
    distances = []
    customers_served = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_distance = 0
        customers_visited = 0
        done = False
        
        state = agent.get_state_features(obs, env)
        
        while not done:
            action = agent.get_action(state, obs, env, training=False)
            
            if action is None:
                break
            
            # Get vehicle position before action
            vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            old_pos = vehicle_data[env.current_vehicle][:2]
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Calculate distance traveled
            next_vehicle_data = next_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            new_pos = next_vehicle_data[env.current_vehicle][:2]
            distance = np.linalg.norm(new_pos - old_pos)
            
            episode_reward += reward
            episode_distance += distance
            customers_visited += 1
            
            obs = next_obs
            state = agent.get_state_features(obs, env)
        
        rewards.append(episode_reward)
        distances.append(episode_distance)
        customers_served.append(customers_visited)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_distance': np.mean(distances),
        'mean_customers': np.mean(customers_served)
    }

def train_with_hyperparams(hyperparams, num_episodes=1000, eval_interval=100):
    """
    Train agent with specific hyperparameters
    
    Args:
        hyperparams: Dictionary of hyperparameters
        num_episodes: Number of training episodes
        eval_interval: Evaluate every N episodes
    
    Returns:
        Dictionary with training results
    """
    print(f"\nTraining with hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Create environment
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=100.0,
        data_path='data/raw/C101.csv'
    )
    
    # Create agent with hyperparameters
    agent = ImprovedQLearningAgent(
        learning_rate=hyperparams['learning_rate'],
        discount_factor=hyperparams['discount_factor'],
        epsilon=hyperparams['epsilon_start'],
        epsilon_min=hyperparams['epsilon_min'],
        epsilon_decay=hyperparams['epsilon_decay'],
        alpha_decay=hyperparams.get('alpha_decay', 0.9995),
        alpha_min=hyperparams.get('alpha_min', 0.01)
    )
    
    # Training tracking
    eval_rewards = []
    eval_episodes = []
    best_reward = float('-inf')
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        state = agent.get_state_features(obs, env)
        
        while not done:
            action = agent.get_action(state, obs, env, training=True)
            
            if action is None:
                break
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_state = agent.get_state_features(next_obs, env)
            
            # Update Q-values (use environment reward directly - no double counting)
            agent.update(state, action, reward, next_state, done, next_obs, env)
            
            episode_reward += reward
            obs = next_obs
            state = next_state
        
        # Decay exploration and learning rate
        agent.decay_epsilon()
        agent.decay_alpha()
        agent.episode_count += 1
        
        # Evaluate periodically
        if episode % eval_interval == 0:
            eval_results = evaluate_agent(agent, env, num_episodes=10)
            eval_rewards.append(eval_results['mean_reward'])
            eval_episodes.append(episode)
            
            if eval_results['mean_reward'] > best_reward:
                best_reward = eval_results['mean_reward']
            
            print(f"Episode {episode:4d} | "
                  f"Eval Reward: {eval_results['mean_reward']:8.2f} | "
                  f"Best: {best_reward:8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Alpha: {agent.alpha:.4f}")
    
    # Final evaluation
    final_eval = evaluate_agent(agent, env, num_episodes=50)
    
    return {
        'hyperparams': hyperparams,
        'final_reward': final_eval['mean_reward'],
        'final_reward_std': final_eval['std_reward'],
        'final_distance': final_eval['mean_distance'],
        'final_customers': final_eval['mean_customers'],
        'best_reward': best_reward,
        'eval_rewards': eval_rewards,
        'eval_episodes': eval_episodes,
        'q_table_size': len(agent.q_table)
    }

def hyperparameter_search():
    """
    Perform grid search over hyperparameter combinations
    """
    print("="*80)
    print("HYPERPARAMETER TUNING FOR Q-LEARNING VRP")
    print("="*80)
    
    # Define hyperparameter search space
    hyperparameter_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'discount_factor': [0.9, 0.95, 0.99],
        'epsilon_start': [1.0],
        'epsilon_min': [0.01, 0.05],
        'epsilon_decay': [0.995, 0.999],
        'alpha_decay': [0.9995, 1.0],  # 1.0 means no decay
        'alpha_min': [0.01]
    }
    
    # Generate combinations to test
    test_configurations = [
        # Configuration 1: Baseline (medium learning, high exploration decay)
        {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'alpha_decay': 0.9995,
            'alpha_min': 0.01
        },
        # Configuration 2: Higher learning rate
        {
            'learning_rate': 0.2,
            'discount_factor': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'alpha_decay': 0.9995,
            'alpha_min': 0.01
        },
        # Configuration 3: Lower learning rate, slower exploration decay
        {
            'learning_rate': 0.05,
            'discount_factor': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.999,
            'alpha_decay': 0.9995,
            'alpha_min': 0.01
        },
        # Configuration 4: High discount factor (more future-oriented)
        {
            'learning_rate': 0.1,
            'discount_factor': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'alpha_decay': 0.9995,
            'alpha_min': 0.01
        },
        # Configuration 5: No learning rate decay
        {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'alpha_decay': 1.0,  # No decay
            'alpha_min': 0.01
        },
        # Configuration 6: Conservative exploration
        {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.05,  # Higher minimum
            'epsilon_decay': 0.999,  # Slower decay
            'alpha_decay': 0.9995,
            'alpha_min': 0.01
        },
    ]
    
    results = []
    
    # Test each configuration
    for i, config in enumerate(test_configurations, 1):
        print(f"\n{'='*80}")
        print(f"Testing Configuration {i}/{len(test_configurations)}")
        print(f"{'='*80}")
        
        result = train_with_hyperparams(config, num_episodes=1000, eval_interval=100)
        results.append(result)
    
    def convert_numpy(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(x) for x in obj]
        return obj

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = 'hyperparameter_results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'tuning_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("="*80)
    
    # Sort by final reward
    sorted_results = sorted(results, key=lambda x: x['final_reward'], reverse=True)
    
    print(f"\n{'Rank':<6}{'LR':<8}{'Gamma':<8}{'Eps Min':<10}{'Eps Decay':<12}"
          f"{'Alpha Decay':<12}{'Final Reward':<15}{'Best Reward':<15}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results, 1):
        hp = result['hyperparams']
        print(f"{i:<6}"
              f"{hp['learning_rate']:<8.3f}"
              f"{hp['discount_factor']:<8.2f}"
              f"{hp['epsilon_min']:<10.3f}"
              f"{hp['epsilon_decay']:<12.4f}"
              f"{hp['alpha_decay']:<12.4f}"
              f"{result['final_reward']:<15.2f}"
              f"{result['best_reward']:<15.2f}")
    
    # Print best configuration
    best_result = sorted_results[0]
    print("\n" + "="*80)
    print("BEST HYPERPARAMETER CONFIGURATION")
    print("="*80)
    print(f"\nHyperparameters:")
    for key, value in best_result['hyperparams'].items():
        print(f"  {key}: {value}")
    print(f"\nPerformance:")
    print(f"  Final Reward: {best_result['final_reward']:.2f} ± {best_result['final_reward_std']:.2f}")
    print(f"  Best Reward: {best_result['best_reward']:.2f}")
    print(f"  Final Distance: {best_result['final_distance']:.2f}")
    print(f"  Customers Served: {best_result['final_customers']:.1f}")
    print(f"  Q-table Size: {best_result['q_table_size']} states")
    
    print(f"\nResults saved to: {results_file}")
    print("="*80)
    
    return sorted_results

if __name__ == "__main__":
    best_configs = hyperparameter_search()
