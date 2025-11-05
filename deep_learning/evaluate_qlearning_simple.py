"""
Evaluation Script for Simple Q-Learning VRP
"""
import os
import argparse
import numpy as np
from env.mvrp_env import MVRPSTWEnv
from qlearning_simple import SimpleQLearningAgent

def evaluate_simple_qlearning(agent_path, num_episodes=10, verbose=False):
    """
    Evaluate trained Simple Q-Learning agent
    
    Args:
        agent_path: Path to saved agent
        num_episodes: Number of episodes to evaluate
        verbose: Whether to print detailed step information
    """
    print("="*80)
    print("SIMPLE Q-LEARNING VRP EVALUATION")
    print("="*80)
    
    # Load agent
    agent = SimpleQLearningAgent()
    agent.load(agent_path)
    
    print(f"\nPenalty Configuration:")
    for key, value in agent.penalty_config.items():
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
    
    # Evaluation metrics
    all_rewards = []
    all_steps = []
    all_distances = []
    all_customers_served = []
    all_violations = {'early': [], 'late': [], 'capacity': []}
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("-"*80)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = agent.get_state_features(obs, env)
        
        episode_reward = 0.0
        steps = 0
        done = False
        max_steps = 100
        
        episode_violations = {'early': 0, 'late': 0, 'capacity': 0}
        total_distance = 0.0
        
        while not done and steps < max_steps:
            # Get action (greedy, no exploration)
            action = agent.get_action(state, obs, env, training=False)
            
            if action is None:
                break
            
            if verbose:
                print(f"Step {steps+1}: Visiting customer {action}")
            
            # Store previous state
            prev_obs = obs.copy()
            vehicle_data = prev_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            current_pos = vehicle_data[env.current_vehicle][:2]
            
            # Take action
            next_obs, env_reward, done, truncated, info = env.step(action)
            
            # Calculate distance
            next_vehicle_data = next_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            next_pos = next_vehicle_data[env.current_vehicle][:2]
            distance = np.linalg.norm(next_pos - current_pos)
            total_distance += distance
            
            # Calculate custom reward
            reward = agent.calculate_reward(prev_obs, action, next_obs, env, env_reward)
            
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
            
            # Update state
            obs = next_obs
            state = agent.get_state_features(obs, env)
            episode_reward += reward
            steps += 1
            
            if truncated:
                done = True
        
        # Count customers served
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        customers_served = int(np.sum(customers_data[:, -1] > 0.5))
        
        # Store episode metrics
        all_rewards.append(episode_reward)
        all_steps.append(steps)
        all_distances.append(total_distance)
        all_customers_served.append(customers_served)
        for key in all_violations:
            all_violations[key].append(episode_violations[key])
        
        print(f"Episode {episode+1:2d} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Steps: {steps:3d} | "
              f"Distance: {total_distance:7.2f} | "
              f"Customers: {customers_served:2d}/20 | "
              f"Early: {episode_violations['early']:2d} | "
              f"Late: {episode_violations['late']:2d} | "
              f"Cap: {episode_violations['capacity']:2d}")
    
    print("-"*80)
    print("\nEvaluation Results:")
    print(f"  Average Reward:        {np.mean(all_rewards):8.2f} ± {np.std(all_rewards):6.2f}")
    print(f"  Average Steps:         {np.mean(all_steps):8.2f} ± {np.std(all_steps):6.2f}")
    print(f"  Average Distance:      {np.mean(all_distances):8.2f} ± {np.std(all_distances):6.2f}")
    print(f"  Average Customers:     {np.mean(all_customers_served):8.2f} ± {np.std(all_customers_served):6.2f}")
    print(f"  Best Reward:           {np.max(all_rewards):8.2f}")
    print(f"  Worst Reward:          {np.min(all_rewards):8.2f}")
    print("\nViolation Statistics:")
    print(f"  Early Arrivals:        {np.mean(all_violations['early']):6.2f} ± {np.std(all_violations['early']):5.2f}")
    print(f"  Late Arrivals:         {np.mean(all_violations['late']):6.2f} ± {np.std(all_violations['late']):5.2f}")
    print(f"  Capacity Exceeded:     {np.mean(all_violations['capacity']):6.2f} ± {np.std(all_violations['capacity']):5.2f}")
    print("="*80)
    
    return {
        'rewards': all_rewards,
        'steps': all_steps,
        'distances': all_distances,
        'customers_served': all_customers_served,
        'violations': all_violations,
        'mean_reward': np.mean(all_rewards),
        'mean_distance': np.mean(all_distances),
        'mean_customers_served': np.mean(all_customers_served)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate Simple Q-Learning VRP Agent')
    parser.add_argument('--agent', type=str, default='checkpoints_qlearning_simple/best_agent.pkl',
                        help='Path to saved agent')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed step information')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.agent):
        print(f"Error: Agent file not found at {args.agent}")
        print("Please train an agent first using train_qlearning_simple.py")
        return
    
    evaluate_simple_qlearning(args.agent, args.episodes, args.verbose)

if __name__ == "__main__":
    main()
