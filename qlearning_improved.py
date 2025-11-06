"""
Improved Q-Learning Agent for VRP with Proper RL Implementation
Follows standard Q-learning principles with enhanced state representation
"""
import numpy as np
import pickle
from collections import defaultdict

class ImprovedQLearningAgent:
    def __init__(self, penalty_config=None, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 alpha_decay=0.9995, alpha_min=0.01):
        """
        Initialize Improved Q-Learning Agent with proper RL implementation
        
        Args:
            penalty_config: Dictionary with penalty values (for compatibility)
            learning_rate: Alpha parameter for Q-learning (initial)
            discount_factor: Gamma parameter for Q-learning
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            alpha_decay: Decay rate for learning rate
            alpha_min: Minimum learning rate
        """
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = learning_rate
        self.alpha_init = learning_rate
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Penalty configuration (for compatibility)
        self.penalty_config = penalty_config or {
            'distance_weight': 1.0,
            'early_penalty': 50.0,
            'late_penalty': 100.0,
            'capacity_penalty': 200.0
        }
        
        # Statistics
        self.episode_count = 0
        self.total_updates = 0
        self.visit_counts = defaultdict(int)  # State-action visit counts for UCB
        
    def get_state_features(self, obs, env):
        """
        Extract enhanced state features for better learning
        
        Returns:
            Tuple representing the state with more informative features
        """
        # Get customer and vehicle data
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        current_vehicle = vehicle_data[env.current_vehicle]
        
        # Count unserved customers
        num_unserved = int(np.sum(customers_data[:, -1] < 0.5))
        
        # Discretize remaining capacity into more bins for better granularity
        capacity_ratio = current_vehicle[2] / env.vehicle_capacity
        capacity_bin = min(9, int(capacity_ratio * 10))  # 10 bins: 0-10%, 10-20%, ..., 90-100%
        
        # Discretize current time into bins
        time_ratio = current_vehicle[3] / env.max_time
        time_bin = min(4, int(time_ratio * 5))  # 5 bins: 0-20%, 20-40%, ..., 80-100%
        
        # Vehicle index
        vehicle_idx = env.current_vehicle
        
        # Enhanced state: (num_unserved, capacity_bin, time_bin, vehicle_idx)
        state = (num_unserved, capacity_bin, time_bin, vehicle_idx)
        
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
    
    def get_second_best_action(self, state, valid_customers):
        """Get the second best action to encourage exploration"""
        if len(valid_customers) <= 1:
            return valid_customers[0] if valid_customers else None
            
        q_values = [(customer_idx, self.q_table[state].get(customer_idx, 0.0)) 
                   for customer_idx in valid_customers]
        sorted_actions = [x[0] for x in sorted(q_values, key=lambda x: x[1], reverse=True)]
        return sorted_actions[1] if len(sorted_actions) > 1 else sorted_actions[0]
    
    def calculate_reward(self, obs, next_obs, env, env_reward):
        """
        Calculate enhanced reward for the transition with better shaping
        
        Args:
            obs: Current observation
            next_obs: Next observation
            env: Environment instance
            env_reward: Reward from environment
            
        Returns:
            float: Calculated reward
        """
        # Start with environment reward (already includes distance and time penalties)
        reward = env_reward
        
        # Calculate customers served in this step
        prev_customers_served = np.sum(obs[8+env.num_customers*8+2::4])
        curr_customers_served = np.sum(next_obs[8+env.num_customers*8+2::4])
        customers_served = curr_customers_served - prev_customers_served
        
        # Reward for serving customers
        if customers_served > 0:
            reward += 0.5 * customers_served  # Bonus for serving customers
        else:
            # Small penalty for not serving any customer
            reward -= 0.2
            
            # Additional penalty if vehicle is not moving much
            vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            next_vehicle_data = next_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
            
            current_pos = vehicle_data[env.current_vehicle][:2]
            next_pos = next_vehicle_data[env.current_vehicle][:2]
            distance = np.linalg.norm(next_pos - current_pos)
            
            if distance < 1.0:  # If vehicle is not moving much
                reward -= 0.3
                
        # Check for time window violations
        if env_reward < -100:  # High penalty in environment
            reward -= 1.0  # Additional penalty for time window violations
            
        return np.clip(reward, -10.0, 10.0)  # Clip rewards to prevent instability
    
    def get_action(self, state, obs, env, training=True):
        """
        Select action using enhanced epsilon-greedy policy with exploration strategies
        
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
            
        # Initialize Q-values for valid actions if not present
        for customer_idx in valid_customers:
            if customer_idx not in self.q_table[state]:
                self.q_table[state][customer_idx] = 0.0
        
        # Epsilon-greedy with adaptive exploration
        if training and np.random.random() < self.epsilon:
            # Exploration strategy: 40% random, 30% nearest neighbor, 30% second best Q
            rand_val = np.random.random()
            if rand_val < 0.4:  # 40% random action
                return np.random.choice(valid_customers)
            elif rand_val < 0.7:  # 30% nearest neighbor
                return self.get_nearest_neighbor_action(obs, env, valid_customers)
            else:  # 30% second best Q-value to encourage exploration
                return self.get_second_best_action(state, valid_customers)
        
        # Exploitation: Choose action with highest Q-value
        q_values = [(customer_idx, self.q_table[state].get(customer_idx, 0.0)) 
                   for customer_idx in valid_customers]
        
        # If all Q-values are equal (early in training), use nearest neighbor
        if len(set(q for _, q in q_values)) == 1:
            return self.get_nearest_neighbor_action(obs, env, valid_customers)
        
        # Select action with highest Q-value
        best_customer = max(q_values, key=lambda x: x[1])[0]
        return best_customer
    
    def update(self, state, action, reward, next_state, done, next_obs, env):
        """
        Update Q-value using standard Q-learning update rule
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        
        This is the proper Bellman equation for Q-learning
        """
        # Initialize Q-value if not present
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        current_q = self.q_table[state][action]
        
        # Track visit counts
        state_action = (state, action)
        self.visit_counts[state_action] += 1
        
        if done:
            # Terminal state has no future rewards
            max_next_q = 0.0
        else:
            # Get valid actions for next state
            valid_next_customers = self.get_valid_customers(next_obs, env)
            
            if not valid_next_customers:
                max_next_q = 0.0
            else:
                # Initialize Q-values for next state if needed
                for customer_idx in valid_next_customers:
                    if customer_idx not in self.q_table[next_state]:
                        self.q_table[next_state][customer_idx] = 0.0
                
                # Get maximum Q-value for next state (Bellman optimality)
                next_q_values = [self.q_table[next_state][c] for c in valid_next_customers]
                max_next_q = max(next_q_values)
        
        # Q-learning update (Temporal Difference learning)
        # TD target = r + γ * max Q(s',a')
        # TD error = target - current Q
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        
        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.q_table[state][action] = new_q
        
        self.total_updates += 1
    
    def decay_epsilon(self):
        """Decay exploration rate (standard epsilon decay)"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def decay_alpha(self):
        """Decay learning rate for better convergence"""
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
    
    def save(self, filepath):
        """Save Q-table and agent parameters"""
        data = {
            'q_table': dict(self.q_table),
            'penalty_config': self.penalty_config,
            'alpha': self.alpha,
            'alpha_init': self.alpha_init,
            'alpha_decay': self.alpha_decay,
            'alpha_min': self.alpha_min,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'episode_count': self.episode_count,
            'total_updates': self.total_updates,
            'visit_counts': dict(self.visit_counts)
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
        self.alpha_init = data.get('alpha_init', self.alpha)
        self.alpha_decay = data.get('alpha_decay', 0.9995)
        self.alpha_min = data.get('alpha_min', 0.01)
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_min = data.get('epsilon_min', 0.01)
        self.epsilon_decay = data.get('epsilon_decay', 0.995)
        self.episode_count = data.get('episode_count', 0)
        self.total_updates = data.get('total_updates', 0)
        
        # Load visit counts
        loaded_visits = data.get('visit_counts', {})
        self.visit_counts = defaultdict(int)
        for key, count in loaded_visits.items():
            self.visit_counts[key] = count
        
        print(f"Agent loaded from {filepath}")
        print(f"Q-table size: {len(self.q_table)} states")
