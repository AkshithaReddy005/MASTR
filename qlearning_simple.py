"""
Simplified Q-Learning Agent for VRP
Uses a simpler state representation and nearest-neighbor heuristic with learning
"""
import numpy as np
import pickle
from collections import defaultdict

class SimpleQLearningAgent:
    def __init__(self, penalty_config=None, learning_rate=0.2, discount_factor=0.9, epsilon=0.3):
        """
        Initialize Simple Q-Learning Agent
        
        Args:
            penalty_config: Dictionary with penalty values
            learning_rate: Alpha parameter for Q-learning
            discount_factor: Gamma parameter for Q-learning
            epsilon: Exploration rate
        """
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        # Penalty configuration
        self.penalty_config = penalty_config or {
            'distance_weight': 1.0,
            'early_penalty': 50.0,
            'late_penalty': 100.0,
            'capacity_penalty': 200.0
        }
        
        # Statistics
        self.episode_count = 0
        self.total_updates = 0
        
    def get_state_features(self, obs, env):
        """
        Extract simple state features
        
        Returns:
            Tuple representing the state
        """
        # Get customer and vehicle data
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        current_vehicle = vehicle_data[env.current_vehicle]
        
        # Count unserved customers
        num_unserved = int(np.sum(customers_data[:, -1] < 0.5))
        
        # Discretize remaining capacity (4 bins: 0-25%, 25-50%, 50-75%, 75-100%)
        capacity_ratio = current_vehicle[2] / env.vehicle_capacity
        capacity_bin = min(3, int(capacity_ratio * 4))
        
        # Simple state: (num_unserved, capacity_bin)
        state = (num_unserved, capacity_bin)
        
        return state
    
    def get_valid_customers(self, obs, env):
        """Get list of valid customer indices that can be visited"""
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        current_vehicle = vehicle_data[env.current_vehicle]
        
        valid_customers = []
        
        for i in range(env.num_customers):
            is_served = customers_data[i, -1] > 0.5
            demand = customers_data[i, 2]
            
            # Check if customer can be served
            if not is_served and demand <= current_vehicle[2]:
                valid_customers.append(i)
        
        return valid_customers
    
    def get_nearest_neighbor_action(self, obs, env, valid_customers):
        """Get nearest unvisited customer (heuristic baseline)"""
        if not valid_customers:
            return None
        
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        current_pos = vehicle_data[env.current_vehicle][:2]
        
        # Find nearest customer
        min_dist = float('inf')
        nearest = valid_customers[0]
        
        for customer_idx in valid_customers:
            customer_pos = customers_data[customer_idx][:2]
            dist = np.linalg.norm(current_pos - customer_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = customer_idx
        
        return nearest
    
    def get_action(self, state, obs, env, training=True):
        """
        Select action using epsilon-greedy with nearest neighbor bias
        
        Args:
            state: Current state tuple
            obs: Full observation
            env: Environment instance
            training: Whether in training mode
        
        Returns:
            Selected customer index
        """
        valid_customers = self.get_valid_customers(obs, env)
        
        if not valid_customers:
            return None
        
        # Initialize Q-values for valid actions
        for customer_idx in valid_customers:
            if customer_idx not in self.q_table[state]:
                # Initialize with nearest neighbor heuristic value
                self.q_table[state][customer_idx] = 0.0
        
        # Epsilon-greedy selection
        if training and np.random.random() < self.epsilon:
            # Exploration: 50% random, 50% nearest neighbor
            if np.random.random() < 0.5:
                return np.random.choice(valid_customers)
            else:
                return self.get_nearest_neighbor_action(obs, env, valid_customers)
        
        # Exploitation: choose action with highest Q-value
        q_values = [(customer_idx, self.q_table[state][customer_idx]) 
                    for customer_idx in valid_customers]
        
        # If all Q-values are equal, use nearest neighbor
        if len(set(q for _, q in q_values)) == 1:
            return self.get_nearest_neighbor_action(obs, env, valid_customers)
        
        # Select action with highest Q-value
        best_customer = max(q_values, key=lambda x: x[1])[0]
        return best_customer
    
    def calculate_reward(self, obs, action, next_obs, env, env_reward):
        """
        Calculate custom reward based on penalty configuration
        
        Returns:
            reward: Float value
        """
        # Start with environment reward (already includes distance and time penalties)
        reward = env_reward
        
        # Extract vehicle data
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        next_vehicle_data = next_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        
        current_pos = vehicle_data[env.current_vehicle][:2]
        next_pos = next_vehicle_data[env.current_vehicle][:2]
        
        # Calculate distance traveled
        distance = np.linalg.norm(next_pos - current_pos)
        
        # Apply distance penalty from config
        reward -= self.penalty_config['distance_weight'] * distance
        
        # Check if customer was visited
        if action < env.num_customers:
            customers_data = next_obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
            customer = customers_data[action]
            
            # Time window info
            ready_time = customer[3]
            due_date = customer[4]
            arrival_time = next_vehicle_data[env.current_vehicle][3]
            
            # Apply custom time window penalties
            if arrival_time < ready_time:
                early_violation = ready_time - arrival_time
                reward -= self.penalty_config['early_penalty'] * early_violation / 100.0
            elif arrival_time > due_date:
                late_violation = arrival_time - due_date
                reward -= self.penalty_config['late_penalty'] * late_violation / 100.0
            
            # Check capacity
            remaining_capacity = next_vehicle_data[env.current_vehicle][2]
            if remaining_capacity < 0:
                reward -= self.penalty_config['capacity_penalty'] * abs(remaining_capacity) / 100.0
        
        return reward
    
    def update(self, state, action, reward, next_state, done, next_obs, env):
        """
        Update Q-value using Q-learning update rule
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        # Initialize Q-value if not present
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        current_q = self.q_table[state][action]
        
        if done:
            max_next_q = 0.0
        else:
            # Get valid actions for next state
            valid_next_customers = self.get_valid_customers(next_obs, env)
            
            if not valid_next_customers:
                max_next_q = 0.0
            else:
                # Initialize Q-values for next state
                for customer_idx in valid_next_customers:
                    if customer_idx not in self.q_table[next_state]:
                        self.q_table[next_state][customer_idx] = 0.0
                
                # Get maximum Q-value
                next_q_values = [self.q_table[next_state][c] for c in valid_next_customers]
                max_next_q = max(next_q_values)
        
        # Q-learning update
        target = reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[state][action] = new_q
        
        self.total_updates += 1
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save Q-table and agent parameters"""
        data = {
            'q_table': dict(self.q_table),
            'penalty_config': self.penalty_config,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_updates': self.total_updates
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table and agent parameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Convert back to defaultdict
        loaded_q_table = data['q_table']
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in loaded_q_table.items():
            for action, value in actions.items():
                self.q_table[state][action] = value
        
        self.penalty_config = data['penalty_config']
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.episode_count = data.get('episode_count', 0)
        self.total_updates = data.get('total_updates', 0)
        
        print(f"Agent loaded from {filepath}")
        print(f"Q-table size: {len(self.q_table)} states")
