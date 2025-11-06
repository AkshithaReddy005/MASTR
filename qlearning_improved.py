"""
Improved Q-Learning Agent for VRP with Proper RL Implementation
Follows standard Q-learning principles with enhanced state representation
"""
import os
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
        Simplified state representation for better learning
        Focuses on key features for routing decisions
        """
        # Extract customer data
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        
        # Extract vehicle data
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        current_vehicle = vehicle_data[env.current_vehicle]
        
        # Current position and status
        current_pos = current_vehicle[:2]
        current_load = current_vehicle[2]
        current_time = current_vehicle[3]
        
        # Get depot position (first customer)
        depot_pos = env.depot_loc
        dist_to_depot = np.linalg.norm(current_pos - depot_pos)
        
        # Count unserved customers
        unserved = np.sum([1 for i in range(env.num_customers) if customers_data[i, -1] < 0.5])
        
        # Distance to nearest unserved customer
        min_dist = float('inf')
        for i in range(env.num_customers):  # Skip depot
            if customers_data[i, -1] < 0.5:  # If unserved
                dist = np.linalg.norm(current_pos - customers_data[i, :2])
                min_dist = min(min_dist, dist)
        
        # Create state tuple (simplified)
        nearest_cust_bin = min(4, int((min_dist / 100.0) * 5)) if min_dist != float('inf') else 4
        
        # Enhanced state representation
        state = (
            min(int(unserved), 4),  # Cap at 4 for state space
            int(current_load > 0),   # Has load or not
            min(int(dist_to_depot / 20.0), 4),  # Distance to depot bucket
            nearest_cust_bin,        # Distance to nearest customer bucket
            min(int(current_time / (env.max_time / 5)), 4)  # Time bucket
        )
        
        return state
        
    def get_valid_customers(self, obs, env):
        """Get list of valid customer indices that can be visited"""
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        current_vehicle = vehicle_data[env.current_vehicle]
        
        unserved = []
        feasible = []
        current_pos = current_vehicle[:2]
        current_time = current_vehicle[3]
        
        for i in range(env.num_customers):
            is_served = customers_data[i, -1] > 0.5
            if not is_served:
                unserved.append(i)
                dest = customers_data[i, :2]
                ready = customers_data[i, 3]
                due = customers_data[i, 4]
                dist = np.linalg.norm(current_pos - dest)
                travel = dist / max(1e-6, env.max_speed)
                start = current_time + travel
                if start < ready:
                    start = ready
                if start <= due:
                    feasible.append(i)
        
        return feasible if feasible else unserved
    
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
    
    def __init__(self, penalty_config=None, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 alpha_decay=0.9995, alpha_min=0.01, replay_buffer_size=10000):
        """
        Initialize Improved Q-Learning Agent with experience replay
        
        Args:
            penalty_config: Dictionary with penalty values (for compatibility)
            learning_rate: Alpha parameter for Q-learning (initial)
            discount_factor: Gamma parameter for Q-learning
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            alpha_decay: Decay rate for learning rate
            alpha_min: Minimum learning rate
            replay_buffer_size: Size of experience replay buffer
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
        
        # Experience replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = 32
        
        # Penalty configuration (for compatibility)
        self.penalty_config = penalty_config or {
            'distance_weight': 1.0,
            'early_penalty': 50.0,
            'late_penalty': 100.0,
            'capacity_penalty': 200.0,
            'idle_penalty': 0.3,
            'efficiency_bonus': 0.5
        }
        
        # Statistics
        self.episode_count = 0
        self.total_updates = 0
        self.visit_counts = defaultdict(int)  # State-action visit counts for UCB
        
    def calculate_reward(self, obs, next_obs, env, env_reward):
        """
        Simplified and effective reward function to maximize customer service
        """
        # Get customer data from observations
        customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        next_customers_data = next_obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
        
        # Count served customers (customers_data already excludes depot)
        prev_served = int(np.sum(customers_data[:, -1] > 0.5))
        curr_served = int(np.sum(next_customers_data[:, -1] > 0.5))
        new_customers_served = curr_served - prev_served
        total_customers = env.num_customers
        
        # Initialize reward
        reward = 0.0
        
        # 1. MASSIVE reward for serving new customers
        if new_customers_served > 0:
            reward += 100.0 * new_customers_served
        
        # 2. Cumulative reward based on total customers served
        if total_customers > 0:
            service_ratio = curr_served / total_customers
            reward += 50.0 * service_ratio
        
        # 3. HUGE bonus for serving all customers
        if total_customers > 0 and curr_served == total_customers:
            reward += 1000.0
        
        # 4. Small step penalty to encourage efficiency
        reward -= 1.0
        
        # 5. Penalty for not serving any customers yet
        if curr_served == 0:
            reward -= 10.0
        
        # Get vehicle data
        vehicle_data = obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        next_vehicle_data = next_obs[8+env.num_customers*8:].reshape(env.num_vehicles, 4)
        
        # Current vehicle information
        current_pos = vehicle_data[env.current_vehicle][:2]
        next_pos = next_vehicle_data[env.current_vehicle][:2]
        distance = np.linalg.norm(next_pos - current_pos)
        current_time = next_vehicle_data[env.current_vehicle][3]
        
        # Small penalty for distance traveled (encourage efficiency)
        # Reduced penalty to not discourage exploration
        reward -= distance * 0.001
        
        # Track if we're making progress toward any customer
        making_progress = False
        
        # Time window and customer proximity bonuses
        for i in range(env.num_customers):
            if customers_data[i, -1] < 0.5:  # If customer not served
                ready_time = customers_data[i, 3]
                due_date = customers_data[i, 4]
                customer_pos = customers_data[i, :2]
                dist_to_customer = np.linalg.norm(next_pos - customer_pos)
                
                # Larger bonus for moving toward customers (inverse distance)
                if distance > 1.0:  # If vehicle is moving
                    prev_dist = np.linalg.norm(current_pos - customer_pos)
                    if dist_to_customer < prev_dist:  # If getting closer to a customer
                        progress_bonus = 1.0 * (1 - dist_to_customer / env.grid_size)
                        reward += progress_bonus
                        making_progress = True
                
                # Time window urgency bonus/penalty
                time_to_arrive = dist_to_customer / max(0.1, env.max_speed)  # Avoid division by zero
                time_until_due = due_date - (current_time + time_to_arrive)
                
                # Larger bonus for being in the time window
                if current_time + time_to_arrive >= ready_time and time_until_due > 0:
                    time_window_center = ready_time + (due_date - ready_time) / 2
                    time_diff = abs((current_time + time_to_arrive) - time_window_center)
                    time_window_bonus = 5.0 / (1 + time_diff)  # Increased base bonus
                    reward += time_window_bonus
                
                # Penalty for being too early or too late (reduced penalties)
                if time_until_due < 0:  # Too late
                    reward -= 0.1 * abs(time_until_due)  # Reduced penalty
                elif current_time + time_to_arrive < ready_time:  # Too early
                    reward -= 0.05 * (ready_time - (current_time + time_to_arrive))  # Reduced penalty
        
        # Small penalty for idle vehicles (reduced penalty)
        if distance < 1.0 and not making_progress:
            reward -= 0.01  # Reduced idle penalty
        
        # Bonus for returning to depot when nearly out of capacity (encourage resupply/route reset)
        if next_vehicle_data[env.current_vehicle, 2] <= env.vehicle_capacity * 0.1:  # Low remaining capacity
            dist_to_depot = np.linalg.norm(next_pos - env.depot_loc)
            depot_bonus = 2.0 * (1 - dist_to_depot / (env.grid_size * 0.7))
            reward += depot_bonus
        
        # Add a small positive reward for each time step to encourage exploration
        reward += 0.1
        
        # Normalize reward to a reasonable range (wider range)
        return np.clip(reward, -10.0, 200.0)  # Increased max reward
        
    def _get_ucb_scores(self, state, valid_customers, c=2.0):
        """Calculate Upper Confidence Bound scores for actions"""
        state_visits = sum(self.visit_counts.get((state, a), 0) for a in valid_customers) + 1e-6
        ucb_scores = []
        
        for a in valid_customers:
            q_val = self.q_table[state].get(a, 0.0)
            n = self.visit_counts.get((state, a), 0)
            exploration_bonus = c * np.sqrt(np.log(state_visits) / (1 + n))
            ucb_scores.append(q_val + exploration_bonus)
            
        return ucb_scores
    
    def get_action(self, state, obs, env, training=True):
        """
        Select action using UCB-enhanced epsilon-greedy policy
        
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
        
        # UCB-enhanced exploration
        if training and np.random.random() < self.epsilon:
            # Exploration strategy: 40% UCB, 30% nearest neighbor, 30% second best Q
            exploration_strategy = np.random.choice(['ucb', 'nn', 'second_best'], 
                                                 p=[0.4, 0.3, 0.3])
            
            if exploration_strategy == 'ucb':
                # UCB exploration
                ucb_scores = self._get_ucb_scores(state, valid_customers)
                action = valid_customers[np.argmax(ucb_scores)]
            elif exploration_strategy == 'nn':
                # Nearest neighbor heuristic
                action = self.get_nearest_neighbor_action(obs, env, valid_customers)
            else:
                # Second best Q-value
                action = self.get_second_best_action(state, valid_customers)
                
            return action
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
            customers_data = obs[8:8+env.num_customers*8].reshape(env.num_customers, 8)
            ordered = sorted(valid_customers, key=lambda i: customers_data[i, 4])
            return ordered[0]
        
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
            if valid_next_customers:
                # Ensure Q-values exist for next state actions
                for a_next in valid_next_customers:
                    if a_next not in self.q_table[next_state]:
                        self.q_table[next_state][a_next] = 0.0
                next_q_values = [self.q_table[next_state].get(a_next, 0.0) for a_next in valid_next_customers]
                max_next_q = max(next_q_values) if next_q_values else 0.0
            else:
                max_next_q = 0.0

        # Compute target and update Q-value
        target_q = reward + self.gamma * max_next_q
        self.q_table[state][action] = (1 - self.alpha) * current_q + self.alpha * target_q

        # Count total updates
        self.total_updates += 1
    
    def _sample_from_replay_buffer(self):
        """Sample a batch of experiences from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return self.replay_buffer
        return np.random.choice(self.replay_buffer, self.batch_size, replace=False)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-values using experience replay and Q-learning
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        experience = (state, action, reward, next_state, done)
        self._add_to_replay_buffer(experience)
        
        # Sample a batch from replay buffer
        batch = self._sample_from_replay_buffer()
        
        # Update Q-values using the batch
        for exp in batch:
            s, a, r, s_next, d = exp
            
            # Get current Q-value
            current_q = self.q_table[s].get(a, 0.0)
            
            # Calculate target Q-value
            if d:
                target_q = r
            else:
                # Use max Q-value for next state
                next_q_values = [self.q_table[s_next].get(a, 0.0) for a in self.q_table[s_next]]
                max_next_q = max(next_q_values) if next_q_values else 0
                target_q = r + self.gamma * max_next_q
            
            # Update Q-value with adaptive learning rate
            self.q_table[s][a] = (1 - self.alpha) * current_q + self.alpha * target_q
            
            # Update visit counts
            self.visit_counts[(s, a)] += 1
            self.total_updates += 1
        
        # Increment episode counter if this is the end of an episode
        if done:
            self.episode_count += 1
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def decay_alpha(self):
        """Decay learning rate"""
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
    
    def save(self, filepath):
        """
        Save agent state including Q-table, replay buffer, and training parameters
        """
        # Convert defaultdict to regular dict for serialization
        q_table_dict = {}
        for state, actions in self.q_table.items():
            q_table_dict[state] = dict(actions)
        
        data = {
            'q_table': q_table_dict,
            'replay_buffer': self.replay_buffer,
            'visit_counts': dict(self.visit_counts),
            'penalty_config': self.penalty_config,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates,
            'episode_count': self.episode_count,
            'replay_buffer_size': self.replay_buffer_size,
            'batch_size': self.batch_size,
            'alpha_init': self.alpha_init,
            'alpha_decay': self.alpha_decay,
            'alpha_min': self.alpha_min,
            'gamma': self.gamma,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """
        Load agent state including Q-table, replay buffer, and training parameters
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Load Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in data['q_table'].items():
            self.q_table[state].update(actions)
        
        # Load other parameters
        self.replay_buffer = data.get('replay_buffer', [])
        self.visit_counts = defaultdict(int, data.get('visit_counts', {}))
        self.penalty_config = data.get('penalty_config', self.penalty_config)
        self.alpha = data.get('alpha', self.alpha)
        self.epsilon = data.get('epsilon', self.epsilon)
        self.total_updates = data.get('total_updates', 0)
        self.episode_count = data.get('episode_count', 0)
        self.replay_buffer_size = data.get('replay_buffer_size', 10000)
        self.batch_size = data.get('batch_size', 32)
        self.alpha_init = data.get('alpha_init', self.alpha)
        self.alpha_decay = data.get('alpha_decay', self.alpha_decay)
        self.alpha_min = data.get('alpha_min', self.alpha_min)
        self.gamma = data.get('gamma', self.gamma)
        self.epsilon_min = data.get('epsilon_min', self.epsilon_min)
        self.epsilon_decay = data.get('epsilon_decay', self.epsilon_decay)
        
        # Load visit counts
        loaded_visits = data.get('visit_counts', {})
        self.visit_counts = defaultdict(int)
        for key, count in loaded_visits.items():
            self.visit_counts[key] = count
        
        print(f"Agent loaded from {filepath}")
        print(f"Q-table size: {len(self.q_table)} states")
