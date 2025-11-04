"""
Penalty Variation Analysis for Simple Q-Learning VRP
Trains and evaluates agents with different penalty configurations
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from env.mvrp_env import MVRPSTWEnv
from qlearning_simple import SimpleQLearningAgent
import pandas as pd
from datetime import datetime

def train_with_penalty_config(penalty_config, num_episodes=300, config_name="default"):
    """
    Train an agent with a specific penalty configuration
    
    Args:
        penalty_config: Dictionary with penalty values
        num_episodes: Number of training episodes
        config_name: Name for this configuration
    
    Returns:
        Trained agent and training statistics
    """
    print(f"\n{'='*80}")
    print(f"Training with configuration: {config_name}")
    print(f"{'='*80}")
    print(f"Penalty Configuration:")
    for key, value in penalty_config.items():
        print(f"  {key}: {value}")
    
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
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.get_state_features(obs, env)
        
        episode_reward = 0.0
        steps = 0
        done = False
        max_steps = 100
        
        while not done and steps < max_steps:
            action = agent.get_action(state, obs, env, training=True)
            if action is None:
                break
            
            next_obs, env_reward, done, truncated, info = env.step(action)
            reward = agent.calculate_reward(obs, action, next_obs, env, env_reward)
            next_state = agent.get_state_features(next_obs, env)
            agent.update(state, action, reward, next_state, done, next_obs, env)
            
            obs = next_obs
            state = next_state
            episode_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        agent.decay_epsilon()
        agent.episode_count += 1
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:8.2f}")
    
    return agent, episode_rewards

def evaluate_agent(agent, num_episodes=20):
    """
    Evaluate an agent and return detailed metrics
    
    Returns:
        Dictionary with evaluation metrics
    """
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=200,
        data_path='data/raw/C101.csv',
        max_time=1236,
        seed=42
    )
    
    all_rewards = []
    all_distances = []
    all_customers_served = []
    all_violations = {'early': [], 'late': [], 'capacity': []}
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.get_state_features(obs, env)
        
        episode_reward = 0.0
        total_distance = 0.0
        episode_violations = {'early': 0, 'late': 0, 'capacity': 0}
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = agent.get_action(state, obs, env, training=False)
            if action is None:
                break
            
            # Calculate distance
            vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            current_pos = vehicle_data[env.current_vehicle][:2]
            
            next_obs, env_reward, done, truncated, info = env.step(action)
            
            next_vehicle_data = next_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            next_pos = next_vehicle_data[env.current_vehicle][:2]
            distance = np.linalg.norm(next_pos - current_pos)
            total_distance += distance
            
            # Track violations
            if action < env.num_customers:
                customers_data = next_obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
                customer = customers_data[action]
                
                ready_time = customer[3]
                due_date = customer[4]
                arrival_time = next_vehicle_data[env.current_vehicle][3]
                
                if arrival_time < ready_time:
                    episode_violations['early'] += 1
                elif arrival_time > due_date:
                    episode_violations['late'] += 1
                
                remaining_capacity = next_vehicle_data[env.current_vehicle][2]
                if remaining_capacity < 0:
                    episode_violations['capacity'] += 1
            
            reward = agent.calculate_reward(obs, action, next_obs, env, env_reward)
            episode_reward += reward
            
            obs = next_obs
            state = agent.get_state_features(obs, env)
            steps += 1
            
            if truncated:
                done = True
        
        # Count customers served
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        customers_served = int(np.sum(customers_data[:, -1] > 0.5))
        
        all_rewards.append(episode_reward)
        all_distances.append(total_distance)
        all_customers_served.append(customers_served)
        for key in episode_violations:
            all_violations[key].append(episode_violations[key])
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_distance': np.mean(all_distances),
        'std_distance': np.std(all_distances),
        'mean_customers_served': np.mean(all_customers_served),
        'std_customers_served': np.std(all_customers_served),
        'mean_early_violations': np.mean(all_violations['early']),
        'mean_late_violations': np.mean(all_violations['late']),
        'mean_capacity_violations': np.mean(all_violations['capacity']),
        'total_violations': np.mean(all_violations['early']) + 
                           np.mean(all_violations['late']) + 
                           np.mean(all_violations['capacity'])
    }

def run_penalty_analysis(num_episodes=300, eval_episodes=20):
    """
    Run comprehensive penalty analysis with different configurations
    """
    print("="*80)
    print("PENALTY VARIATION ANALYSIS - SIMPLE Q-LEARNING")
    print("="*80)
    print(f"Training episodes per configuration: {num_episodes}")
    print(f"Evaluation episodes per configuration: {eval_episodes}")
    print("="*80)
    
    # Define penalty configurations to test
    penalty_configs = [
        # Baseline
        {
            'name': 'Baseline',
            'config': {
                'distance_weight': 1.0,
                'early_penalty': 50.0,
                'late_penalty': 100.0,
                'capacity_penalty': 200.0
            }
        },
        # Low penalties
        {
            'name': 'Low Penalties',
            'config': {
                'distance_weight': 1.0,
                'early_penalty': 25.0,
                'late_penalty': 50.0,
                'capacity_penalty': 100.0
            }
        },
        # High penalties
        {
            'name': 'High Penalties',
            'config': {
                'distance_weight': 1.0,
                'early_penalty': 100.0,
                'late_penalty': 200.0,
                'capacity_penalty': 400.0
            }
        },
        # Strict time windows
        {
            'name': 'Strict Time Windows',
            'config': {
                'distance_weight': 1.0,
                'early_penalty': 150.0,
                'late_penalty': 300.0,
                'capacity_penalty': 200.0
            }
        },
        # Strict capacity
        {
            'name': 'Strict Capacity',
            'config': {
                'distance_weight': 1.0,
                'early_penalty': 50.0,
                'late_penalty': 100.0,
                'capacity_penalty': 500.0
            }
        },
        # Distance priority
        {
            'name': 'Distance Priority',
            'config': {
                'distance_weight': 2.0,
                'early_penalty': 30.0,
                'late_penalty': 60.0,
                'capacity_penalty': 150.0
            }
        }
    ]
    
    results = []
    
    # Train and evaluate each configuration
    for idx, config_data in enumerate(penalty_configs):
        print(f"\n\nConfiguration {idx+1}/{len(penalty_configs)}")
        
        # Train agent
        agent, training_rewards = train_with_penalty_config(
            config_data['config'],
            num_episodes=num_episodes,
            config_name=config_data['name']
        )
        
        # Evaluate agent
        print(f"\nEvaluating {config_data['name']}...")
        eval_metrics = evaluate_agent(agent, num_episodes=eval_episodes)
        
        # Store results
        result = {
            'name': config_data['name'],
            'penalty_config': config_data['config'],
            'eval_metrics': eval_metrics,
            'training_rewards': training_rewards
        }
        results.append(result)
        
        # Print evaluation summary
        print(f"\nEvaluation Results for {config_data['name']}:")
        print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
        print(f"  Mean Distance: {eval_metrics['mean_distance']:.2f} ± {eval_metrics['std_distance']:.2f}")
        print(f"  Mean Customers Served: {eval_metrics['mean_customers_served']:.2f} ± {eval_metrics['std_customers_served']:.2f}")
        print(f"  Early Violations: {eval_metrics['mean_early_violations']:.2f}")
        print(f"  Late Violations: {eval_metrics['mean_late_violations']:.2f}")
        print(f"  Capacity Violations: {eval_metrics['mean_capacity_violations']:.2f}")
        print(f"  Total Violations: {eval_metrics['total_violations']:.2f}")
    
    # Generate comparison report
    generate_comparison_report(results)
    
    # Generate visualizations
    generate_visualizations(results)
    
    return results

def generate_comparison_report(results):
    """Generate a detailed comparison report"""
    print("\n\n" + "="*80)
    print("PENALTY ANALYSIS - COMPARISON REPORT")
    print("="*80)
    
    # Create comparison table
    data = []
    for result in results:
        metrics = result['eval_metrics']
        config = result['penalty_config']
        data.append({
            'Configuration': result['name'],
            'Early Penalty': config['early_penalty'],
            'Late Penalty': config['late_penalty'],
            'Capacity Penalty': config['capacity_penalty'],
            'Mean Reward': f"{metrics['mean_reward']:.2f}",
            'Mean Distance': f"{metrics['mean_distance']:.2f}",
            'Customers Served': f"{metrics['mean_customers_served']:.2f}",
            'Early Violations': f"{metrics['mean_early_violations']:.2f}",
            'Late Violations': f"{metrics['mean_late_violations']:.2f}",
            'Capacity Violations': f"{metrics['mean_capacity_violations']:.2f}",
            'Total Violations': f"{metrics['total_violations']:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'penalty_analysis_report_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    
    print("\n" + df.to_string(index=False))
    print(f"\nReport saved to: {csv_filename}")
    print("="*80)

def generate_visualizations(results):
    """Generate visualization plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Penalty Variation Analysis - Simple Q-Learning', fontsize=16, fontweight='bold')
    
    names = [r['name'] for r in results]
    
    # Plot 1: Mean Rewards
    rewards = [r['eval_metrics']['mean_reward'] for r in results]
    reward_stds = [r['eval_metrics']['std_reward'] for r in results]
    axes[0, 0].bar(range(len(names)), rewards, yerr=reward_stds, capsize=5, color='skyblue')
    axes[0, 0].set_title('Mean Reward by Configuration')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Mean Distance
    distances = [r['eval_metrics']['mean_distance'] for r in results]
    distance_stds = [r['eval_metrics']['std_distance'] for r in results]
    axes[0, 1].bar(range(len(names)), distances, yerr=distance_stds, capsize=5, color='lightcoral')
    axes[0, 1].set_title('Mean Distance by Configuration')
    axes[0, 1].set_ylabel('Mean Distance')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Customers Served
    customers = [r['eval_metrics']['mean_customers_served'] for r in results]
    customers_stds = [r['eval_metrics']['std_customers_served'] for r in results]
    axes[0, 2].bar(range(len(names)), customers, yerr=customers_stds, capsize=5, color='lightgreen')
    axes[0, 2].set_title('Mean Customers Served by Configuration')
    axes[0, 2].set_ylabel('Mean Customers Served')
    axes[0, 2].set_xticks(range(len(names)))
    axes[0, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 2].axhline(y=20, color='red', linestyle='--', label='Total Customers')
    axes[0, 2].legend()
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Plot 4: Violations breakdown
    early_viols = [r['eval_metrics']['mean_early_violations'] for r in results]
    late_viols = [r['eval_metrics']['mean_late_violations'] for r in results]
    capacity_viols = [r['eval_metrics']['mean_capacity_violations'] for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    axes[1, 0].bar(x - width, early_viols, width, label='Early', color='gold')
    axes[1, 0].bar(x, late_viols, width, label='Late', color='orange')
    axes[1, 0].bar(x + width, capacity_viols, width, label='Capacity', color='red')
    axes[1, 0].set_title('Violations by Type and Configuration')
    axes[1, 0].set_ylabel('Mean Violations')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 5: Training curves
    for result in results:
        # Smooth training rewards
        window = 30
        if len(result['training_rewards']) >= window:
            smoothed = np.convolve(result['training_rewards'], 
                                  np.ones(window)/window, mode='valid')
            axes[1, 1].plot(smoothed, label=result['name'], alpha=0.7)
    
    axes[1, 1].set_title('Training Progress')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Smoothed Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 6: Total violations vs Distance
    total_viols = [r['eval_metrics']['total_violations'] for r in results]
    axes[1, 2].scatter(distances, total_viols, s=100, alpha=0.6)
    for i, name in enumerate(names):
        axes[1, 2].annotate(name, (distances[i], total_viols[i]), 
                           fontsize=8, ha='right')
    axes[1, 2].set_title('Trade-off: Distance vs Violations')
    axes[1, 2].set_xlabel('Mean Distance')
    axes[1, 2].set_ylabel('Total Violations')
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'penalty_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {filename}")
    
    plt.close()

def main():
    """Main function"""
    print("\nStarting Penalty Variation Analysis...")
    print("This will train and evaluate multiple Q-Learning agents with different penalty configurations.")
    print("This may take some time...\n")
    
    results = run_penalty_analysis(num_episodes=300, eval_episodes=20)
    
    print("\n\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - penalty_analysis_report_*.csv (comparison table)")
    print("  - penalty_analysis_*.png (visualization)")
    print("\nReview these files to understand how different penalty configurations")
    print("affect the routing solution quality and constraint violations.")
    print("="*80)

if __name__ == "__main__":
    main()
