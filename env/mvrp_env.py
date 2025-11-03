"""
Multi-Vehicle Routing Problem with Soft Time Windows (MVRPSTW) Environment

This module implements a custom Gym environment for the MVRPSTW problem.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union

class MVRPSTWEnv(gym.Env):
    """
    A custom Gym environment for Multi-Vehicle Routing Problem with Soft Time Windows.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, num_customers: int = 20, num_vehicles: int = 3, 
                 vehicle_capacity: float = 100.0, grid_size: float = 100.0,
                 max_time: float = 480.0, penalty_early: float = 1.0,
                 penalty_late: float = 2.0, seed: Optional[int] = None):
        """
        Initialize the MVRPSTW environment.
        
        Args:
            num_customers: Number of customers to serve
            num_vehicles: Number of vehicles available
            vehicle_capacity: Capacity of each vehicle
            grid_size: Size of the grid
            max_time: Maximum time horizon
            penalty_early: Penalty multiplier for early arrival
            penalty_late: Penalty multiplier for late arrival
            seed: Random seed for reproducibility
        """
        super(MVRPSTWEnv, self).__init__()
        
        # Store configuration
        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.grid_size = grid_size
        self.max_time = max_time
        self.penalty_early = penalty_early
        self.penalty_late = penalty_late
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Current vehicle index
        self.current_vehicle = 0
        
        # Initialize state variables
        self.customer_locations = None
        self.demands = None
        self.time_windows = None
        self.vehicle_states = None
        self.visited = None
        self.routes = None
        self.depot_loc = np.array([0.0, 0.0])
            
        # Initialize state
        self.reset()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(num_customers)
        self.observation_space = self._get_flat_observation_space()
        
    def _get_flat_observation_space(self) -> spaces.Box:
        """Define the flat observation space."""
        # Observation: depot(8) + customers(N*8) + vehicles(M*4)
        obs_size = 8 + self.num_customers * 8 + self.num_vehicles * 4
        return spaces.Box(
            low=0.0,
            high=max(self.grid_size, self.max_time),
            shape=(obs_size,),
            dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        elif self.seed is not None:
            np.random.seed(self.seed)
            
        # Generate random customer locations and demands
        self.customer_locations = np.random.uniform(
            0, self.grid_size, 
            size=(self.num_customers, 2)
        )
        self.demands = np.random.uniform(
            5, 25, 
            size=self.num_customers
        )
        
        # Generate time windows (start_time, end_time) for each customer
        time_starts = np.random.uniform(0, self.max_time * 0.6, size=self.num_customers)
        time_durations = np.random.uniform(60, 120, size=self.num_customers)
        self.time_windows = np.column_stack([time_starts, time_starts + time_durations])
        
        # Initialize vehicle states [x, y, capacity_remaining, current_time]
        self.vehicle_states = np.zeros((self.num_vehicles, 4))
        self.vehicle_states[:, 2] = self.vehicle_capacity  # Set initial capacity
        
        # Track visited customers
        self.visited = np.zeros(self.num_customers, dtype=bool)
        
        # Track routes
        self.routes = [[] for _ in range(self.num_vehicles)]
        
        # Current vehicle
        self.current_vehicle = 0
        # Safety: step counter to avoid endless episodes
        self._steps = 0
        self._max_steps = max(50, self.num_customers * self.num_vehicles * 5)
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Customer index to visit (0 to num_customers-1)
            
        Returns:
            observation: The new observation
            reward: The reward for the action
            terminated: Whether the episode is complete
            truncated: Whether the episode was truncated
            info: Additional information
        """
        customer_idx = action
        vehicle_idx = self.current_vehicle
        
        # Increment step counter
        self._steps += 1

        # Check if action is valid
        if customer_idx < 0 or customer_idx >= self.num_customers:
            return self._get_obs(), -100.0, False, False, {'error': 'invalid_action'}
            
        if self.visited[customer_idx]:
            return self._get_obs(), -50.0, False, False, {'error': 'already_visited'}
            
        # Get vehicle and customer info
        vehicle = self.vehicle_states[vehicle_idx]
        customer_loc = self.customer_locations[customer_idx]
        demand = self.demands[customer_idx]
        
        # Check capacity constraint
        if vehicle[2] < demand:
            # Need to return to depot and use next vehicle
            distance_to_depot = np.linalg.norm(vehicle[:2] - self.depot_loc)
            reward = -distance_to_depot  # Penalty for returning to depot
            
            # Switch to next vehicle
            self.current_vehicle = (self.current_vehicle + 1) % self.num_vehicles
            if self.current_vehicle == 0:
                # All vehicles used, episode ends
                return self._get_obs(), reward, True, False, {'reason': 'capacity_exceeded'}
            
            return self._get_obs(), reward, False, False, {'action': 'returned_to_depot'}
        
        # Calculate travel distance and time
        distance = np.linalg.norm(vehicle[:2] - customer_loc)
        travel_time = distance  # Assume speed = 1 unit per minute
        
        # Update vehicle state
        arrival_time = vehicle[3] + travel_time
        
        # Calculate time window penalties
        start_time, end_time = self.time_windows[customer_idx]
        if arrival_time < start_time:
            # Early arrival
            penalty = (start_time - arrival_time) * self.penalty_early
            arrival_time = start_time  # Wait until window opens
        elif arrival_time > end_time:
            # Late arrival
            penalty = (arrival_time - end_time) * self.penalty_late
        else:
            penalty = 0.0
            
        # Service time
        service_time = 10.0
        
        # Update vehicle state
        self.vehicle_states[vehicle_idx, :2] = customer_loc  # Update location
        self.vehicle_states[vehicle_idx, 2] -= demand  # Reduce capacity
        self.vehicle_states[vehicle_idx, 3] = arrival_time + service_time  # Update time
        
        # Mark customer as visited
        self.visited[customer_idx] = True
        
        # Add to route
        self.routes[vehicle_idx].append(customer_idx)
        
        # Calculate reward (negative of total cost: distance + penalty)
        reward = -(distance + penalty)
        
        # Check if all customers are served
        done = np.all(self.visited)

        # Safety truncation
        truncated = False
        if not done and self._steps >= self._max_steps:
            truncated = True
            done = True

        return self._get_obs(), reward, done, truncated, {}
    
    def _get_obs(self) -> np.ndarray:
        """Get the current environment observation as a flat array."""
        # Depot features: [x, y, demand=0, start_time=0, end_time=max, penalty_early, penalty_late, visited=0]
        depot_features = np.array([0.0, 0.0, 0.0, 0.0, self.max_time, 
                                   self.penalty_early, self.penalty_late, 0.0])
        
        # Customer features: [x, y, demand, start_time, end_time, penalty_early, penalty_late, visited]
        customer_features = np.column_stack([
            self.customer_locations,
            self.demands,
            self.time_windows,
            np.full(self.num_customers, self.penalty_early),
            np.full(self.num_customers, self.penalty_late),
            self.visited.astype(np.float32)
        ])
        
        # Flatten customer features
        customer_features_flat = customer_features.flatten()
        
        # Vehicle states: [x, y, capacity_remaining, current_time]
        vehicle_features_flat = self.vehicle_states.flatten()
        
        # Concatenate all features
        obs = np.concatenate([depot_features, customer_features_flat, vehicle_features_flat])
        
        return obs.astype(np.float32)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment."""
        plt.figure(figsize=(10, 8))
        
        # Plot customers
        plt.scatter(
            self.customer_locations[:, 0], 
            self.customer_locations[:, 1],
            c='blue', 
            label='Unvisited Customers'
        )
        
        # Plot visited customers
        if np.any(self.visited):
            plt.scatter(
                self.customer_locations[self.visited, 0],
                self.customer_locations[self.visited, 1],
                c='green',
                label='Visited Customers'
            )
        
        # Plot vehicles and routes
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
        for i, (vehicle, route) in enumerate(zip(self.vehicle_states, self.routes)):
            if len(route) > 0:
                # Plot route
                route_locs = np.vstack([
                    np.array([0, 0]),  # Start at depot
                    self.customer_locations[route]
                ])
                plt.plot(
                    route_locs[:, 0], 
                    route_locs[:, 1], 
                    '--',
                    color=colors[i % len(colors)],
                    alpha=0.5,
                    label=f'Vehicle {i+1} Route'
                )
            
            # Plot vehicle position
            plt.scatter(
                vehicle[0],
                vehicle[1],
                marker='*',
                s=200,
                color=colors[i % len(colors)],
                edgecolors='black',
                label=f'Vehicle {i+1}'
            )
        
        # Plot depot
        plt.scatter(0, 0, c='black', marker='s', s=100, label='Depot')
        
        plt.title('MVRPSTW Environment')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        
        if mode == 'human':
            plt.show()
            return None
        else:  # rgb_array
            fig = plt.gcf()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return img
    
    def close(self):
        """Close the environment and any open windows."""
        plt.close()
        
    def get_routes(self) -> List[List[int]]:
        """Get the current routes for all vehicles."""
        return self.routes
    
    def get_solution_cost(self) -> float:
        """Calculate the total cost of the current solution."""
        total_cost = 0.0
        
        for vehicle, route in zip(self.vehicle_states, self.routes):
            if not route:
                continue
                
            # Add cost from depot to first customer
            first_customer = self.customer_locations[route[0]]
            total_cost += np.linalg.norm(first_customer)
            
            # Add costs between customers
            for i in range(len(route) - 1):
                loc1 = self.customer_locations[route[i]]
                loc2 = self.customer_locations[route[i+1]]
                total_cost += np.linalg.norm(loc2 - loc1)
            
            # Add cost from last customer back to depot
            last_customer = self.customer_locations[route[-1]]
            total_cost += np.linalg.norm(last_customer)
            
        return total_cost


if __name__ == "__main__":
    # Example usage
    env = MVRPSTWEnv({
        'num_customers': 10,
        'num_vehicles': 2,
        'seed': 42
    })
    
    obs = env.reset()
    done = False
    
    while not done:
        # Simple random policy for demonstration
        valid_actions = np.where(obs['mask'].flatten())[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            vehicle_idx = action // (env.config['num_customers'] + 1)
            customer_idx = action % (env.config['num_customers'] + 1) - 1
            if customer_idx == -1:  # Depot
                customer_idx = 0  # Simple workaround for depot
            obs, reward, done, _ = env.step((vehicle_idx, customer_idx))
            env.render()
        else:
            break
    
    print("Routes:", env.get_routes())
    print("Total cost:", env.get_solution_cost())
    env.close()
