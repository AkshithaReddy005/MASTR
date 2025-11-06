"""
Evaluation Script for Improved Q-Learning VRP Agent
"""
import os
import json
import numpy as np
from datetime import datetime
from env.mvrp_env import MVRPSTWEnv
from qlearning_improved import ImprovedQLearningAgent

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

def evaluate_improved_qlearning(agent_path='checkpoints_qlearning_improved/best_agent.pkl', 
                                 num_episodes=50):
    """
    Evaluate the improved Q-learning agent with enhanced metrics
    
    Args:
        agent_path: Path to saved agent
        num_episodes: Number of evaluation episodes
    """
    print("="*80)
    print("IMPROVED Q-LEARNING VRP EVALUATION")
    print("="*80)
    
    # Create environment with parameters matching training
    env = MVRPSTWEnv(
        num_customers=100,  # Match training configuration
        num_vehicles=25,    # Match final training stage (100c/25v)
        vehicle_capacity=200.0,  # Capacity from Solomon header
        data_path='data/c101.txt',  # Use same instance family
        penalty_early=0.3,   # Match training penalties
        penalty_late=0.5
    )
    
    # Load agent
    agent = ImprovedQLearningAgent()
    agent.load(agent_path)
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("-" * 80)
    
    # ANSI color codes
    R = '\033[91m'  # Red
    G = '\033[92m'  # Green
    Y = '\033[93m'  # Yellow
    B = '\033[94m'  # Blue
    C = '\033[0m'   # Reset
    
    # Evaluation tracking
    episode_rewards = []
    episode_steps = []
    episode_distances = []
    episode_customers = []
    early_violations = []
    late_violations = []
    capacity_violations = []
    vehicle_utilization = []
    
    # For visualization
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Evaluation loop
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        
        # Get depot position from the environment
        if hasattr(env, 'depot_loc'):
            depot = env.depot_loc
        else:
            # Fallback: Use default depot position (center of grid)
            depot = np.array([env.grid_size / 2, env.grid_size / 2])
            print(f"[WARNING] Using default depot position: {depot}")
        
        episode_reward = 0
        episode_step = 0
        episode_distance = 0
        customers_visited = 0
        done = False
        
        early_count = 0
        late_count = 0
        capacity_count = 0
        vehicle_usage = [0] * env.num_vehicles
        
        state = agent.get_state_features(obs, env)
        
        # Track route for visualization
        if episode <= 3:  # Only visualize first 3 episodes
            routes = [[] for _ in range(env.num_vehicles)]
            # Use the depot position we already got at the start of the episode
            for v in range(env.num_vehicles):
                routes[v].append(depot.copy())
        
        while not done:
            # Get action (no exploration during evaluation)
            action = agent.get_action(state, obs, env, training=False)
            
            if action is None:
                break
            
            # Get vehicle position before action
            vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            old_pos = vehicle_data[env.current_vehicle][:2]
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Calculate distance traveled
            next_vehicle_data = next_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            new_pos = next_vehicle_data[env.current_vehicle][:2]
            distance = np.linalg.norm(new_pos - old_pos)
            
            # Track vehicle usage
            vehicle_usage[env.current_vehicle] = 1
            
            # Track route for visualization
            if episode <= 3:
                routes[env.current_vehicle].append(new_pos.copy())
            
            # Check for violations
            if action < env.num_customers:
                customers_data = next_obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
                customer = customers_data[action]
                
                ready_time = customer[3]
                due_date = customer[4]
                arrival_time = next_vehicle_data[env.current_vehicle][3]
                
                if arrival_time < ready_time:
                    early_count += 1
                elif arrival_time > due_date:
                    late_count += 1
                
                remaining_capacity = next_vehicle_data[env.current_vehicle][2]
                if remaining_capacity < 0:
                    capacity_count += 1
            
            # Track metrics
            episode_reward += reward
            episode_step += 1
            episode_distance += distance
            if action < env.num_customers:
                # Count as served only if environment marked it visited
                customers_data_after = next_obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
                if customers_data_after[action, -1] > 0.5:
                    customers_visited += 1
            
            # Move to next state
            obs = next_obs
            state = agent.get_state_features(next_obs, env)
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        episode_distances.append(episode_distance)
        episode_customers.append(customers_visited)
        early_violations.append(early_count)
        late_violations.append(late_count)
        capacity_violations.append(capacity_count)
        vehicle_utilization.append(sum(vehicle_usage) / env.num_vehicles * 100)  # Percentage
        
        # Color code based on performance
        reward_color = G if episode_reward > 0 else R if episode_reward < -5 else Y
        customer_color = G if customers_visited > 15 else R if customers_visited < 10 else Y
        
        # Print episode results with colors
        print(f"{B}Episode {episode:2d}{C} | "
              f"{reward_color}Reward: {episode_reward:8.2f}{C} | "
              f"Steps: {episode_step:3d} | "
              f"Dist: {episode_distance:7.2f} | "
              f"{customer_color}Customers: {customers_visited:2d}/{env.num_customers}{C} | "
              f"Early: {Y if early_count > 0 else C}{early_count:2d}{C} | "
              f"Late: {R if late_count > 0 else C}{late_count:2d}{C} | "
              f"Cap: {R if capacity_count > 0 else C}{capacity_count:2d}{C} | "
              f"Vehicles: {sum(vehicle_usage)}/{env.num_vehicles}")
        
        # Visualize routes for first 3 episodes
        if episode <= 3:
            plt.figure(figsize=(10, 8))
            plt.scatter([depot[0]], [depot[1]], c='red', marker='s', s=100, label='Depot')
            
            # Plot customers
            try:
                customers = env.customers
                plt.scatter(customers[:, 0], customers[:, 1], c='blue', label='Customers')
            except AttributeError:
                # Fallback: Extract customer positions from observation
                customer_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
                plt.scatter(customer_data[:, 0], customer_data[:, 1], c='blue', label='Customers')
            
            # Plot routes
            colors = ['green', 'purple', 'orange', 'brown', 'pink']
            for v in range(env.num_vehicles):
                if len(routes[v]) > 1:  # If vehicle was used
                    route = np.array(routes[v])
                    plt.plot(route[:, 0], route[:, 1], 'o-', color=colors[v % len(colors)], 
                             label=f'Vehicle {v+1}')
            
            plt.title(f'Episode {episode} - Reward: {episode_reward:.2f}, Customers: {customers_visited}/{env.num_customers}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            os.makedirs('evaluation_plots', exist_ok=True)
            plt.savefig(f'evaluation_plots/episode_{episode}_routes.png')
            plt.close()
    
    print("\n" + "="*80)
    
    # Calculate statistics
    stats = {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'distances': episode_distances,
        'customers': episode_customers,
        'early_violations': early_violations,
        'late_violations': late_violations,
        'capacity_violations': capacity_violations,
        'vehicle_utilization': vehicle_utilization
    }
    
    # Print summary statistics with colors
    print("\n" + "="*80)
    print(f"{'EVALUATION SUMMARY':^80}")
    print("="*80)
    
    # Performance Metrics
    print(f"\n{G}{'PERFORMANCE METRICS':^80}{C}")
    print("-"*80)
    print(f"{'Average Reward:':<25} {G if np.mean(episode_rewards) > 0 else R}{np.mean(episode_rewards):8.2f} ± {np.std(episode_rewards):6.2f}{C}")
    print(f"{'Average Steps:':<25} {np.mean(episode_steps):8.2f} ± {np.std(episode_steps):6.2f}")
    print(f"{'Average Distance:':<25} {np.mean(episode_distances):8.2f} ± {np.std(episode_distances):6.2f}")
    print(f"{'Average Customers Served:':<25} {G if np.mean(episode_customers) > 15 else R if np.mean(episode_customers) < 10 else Y}{np.mean(episode_customers):8.2f} ± {np.std(episode_customers):6.2f}{C}")
    print(f"{'Best Reward:':<25} {G}{np.max(episode_rewards):8.2f}{C}")
    print(f"{'Worst Reward:':<25} {R if np.min(episode_rewards) < -5 else Y}{np.min(episode_rewards):8.2f}{C}")
    
    # Violation Statistics
    print(f"\n{R}{'VIOLATION STATISTICS':^80}{C}")
    print("-"*80)
    print(f"{'Early Arrivals:':<25} {Y if np.mean(early_violations) > 0 else C}{np.mean(early_violations):6.2f} ± {np.std(early_violations):5.2f}{C}")
    print(f"{'Late Arrivals:':<25} {R if np.mean(late_violations) > 0 else C}{np.mean(late_violations):6.2f} ± {np.std(late_violations):5.2f}{C}")
    print(f"{'Capacity Violations:':<25} {R if np.mean(capacity_violations) > 0 else C}{np.mean(capacity_violations):6.2f} ± {np.std(capacity_violations):5.2f}{C}")
    
    # Resource Utilization
    print(f"\n{B}{'RESOURCE UTILIZATION':^80}{C}")
    print("-"*80)
    print(f"{'Vehicle Utilization:':<25} {G if np.mean(vehicle_utilization) > 70 else Y if np.mean(vehicle_utilization) > 30 else R}{np.mean(vehicle_utilization):6.2f}% ± {np.std(vehicle_utilization):5.2f}%{C}")
    
    print("\n" + "="*80)
    
    # Save detailed report
    save_evaluation_report(stats, num_episodes, env.num_customers)
    
    return stats

def save_evaluation_report(stats, num_episodes, total_customers):
    """Save detailed evaluation report to file"""
    report = []
    report.append("="*80)
    report.append("DETAILED EVALUATION REPORT".center(80))
    report.append("="*80)
    
    # Summary
    report.append("\nSUMMARY".ljust(80, '-'))
    report.append(f"{'Total Episodes:':<25} {num_episodes}")
    report.append(f"{'Total Customers:':<25} {total_customers}")
    report.append(f"{'Average Customers Served:':<25} {np.mean(stats['customers']):.2f} ± {np.std(stats['customers']):.2f}")
    report.append(f"{'Average Reward:':<25} {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
    
    # Detailed Statistics
    report.append("\nDETAILED STATISTICS".ljust(80, '-'))
    report.append(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    report.append("-"*65)
    
    metrics = [
        ('Reward', stats['rewards']),
        ('Steps', stats['steps']),
        ('Distance', stats['distances']),
        ('Customers Served', stats['customers']),
        ('Early Violations', stats['early_violations']),
        ('Late Violations', stats['late_violations']),
        ('Capacity Violations', stats['capacity_violations']),
        ('Vehicle Utilization (%)', stats['vehicle_utilization'])
    ]
    
    for name, values in metrics:
        report.append(f"{name:<25} {np.mean(values):10.2f} {np.std(values):10.2f} {np.min(values):10.2f} {np.max(values):10.2f}")
    
    # Save to file
    os.makedirs('evaluation_reports', exist_ok=True)
    timestamp = np.datetime64('now').astype(str).replace(':', '-')
    report_path = f'evaluation_reports/evaluation_report_{timestamp}.txt'
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nDetailed evaluation report saved to: {report_path}")
    
    # Save raw data
    data_path = f'evaluation_reports/evaluation_data_{timestamp}.json'
    with open(data_path, 'w') as f:
        # Convert all numpy types to native Python types before JSON serialization
        json.dump(convert_numpy(stats), f, indent=2)

def main():
    """Main evaluation function"""
    # ANSI color codes for error messages
    R = '\033[91m'  # Red
    C = '\033[0m'   # Reset
    
    try:
        # Create necessary directories
        os.makedirs('evaluation_plots', exist_ok=True)
        os.makedirs('evaluation_reports', exist_ok=True)
        
        # Run evaluation
        results = evaluate_improved_qlearning()
        
        # Generate and save plots
        try:
            generate_evaluation_plots(results)
        except Exception as e:
            print(f"\n{R}Error generating plots: {e}{C}")
            
    except Exception as e:
        print(f"\n{R}Evaluation failed: {e}{C}")
        raise

def generate_evaluation_plots(stats):
    """Generate and save evaluation plots"""
    import matplotlib.pyplot as plt
    
    # Plot rewards over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(stats['rewards'], 'b-', alpha=0.7)
    plt.axhline(y=np.mean(stats['rewards']), color='r', linestyle='--', 
                label=f'Mean: {np.mean(stats["rewards"]):.2f}')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('evaluation_plots/rewards_over_episodes.png')
    plt.close()
    
    # Plot customers served over episodes
    plt.figure(figsize=(12, 6))
    plt.plot(stats['customers'], 'g-', alpha=0.7)
    plt.axhline(y=np.mean(stats['customers']), color='r', linestyle='--', 
                label=f'Mean: {np.mean(stats["customers"]):.2f}')
    plt.title('Customers Served per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Customers Served')
    plt.legend()
    plt.grid(True)
    plt.savefig('evaluation_plots/customers_served.png')
    plt.close()
    
    # Plot violations
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    
    ax[0].plot(stats['early_violations'], 'y-')
    ax[0].set_title('Early Arrivals per Episode')
    ax[0].set_ylabel('Count')
    ax[0].grid(True)
    
    ax[1].plot(stats['late_violations'], 'r-')
    ax[1].set_title('Late Arrivals per Episode')
    ax[1].set_ylabel('Count')
    ax[1].grid(True)
    
    ax[2].plot(stats['capacity_violations'], 'm-')
    ax[2].set_title('Capacity Violations per Episode')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Count')
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/violations.png')
    plt.close()
    
    print("\nEvaluation plots saved to the 'evaluation_plots' directory.")

if __name__ == "__main__":
    main()
