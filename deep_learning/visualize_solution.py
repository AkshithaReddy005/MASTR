"""
Visualize the solution by saving it to an image file
"""
import numpy as np
import matplotlib.pyplot as plt
from env.mvrp_env import MVRPSTWEnv
from train_complete import MAAM
import torch

def plot_solution(env, solution, filename='solution.png'):
    plt.figure(figsize=(12, 10))
    
    # Get customer locations from the environment's data
    customer_locations = []
    obs, _ = env.reset()
    
    # Extract customer locations from observation
    # First 8 values are vehicle states, then 8 values per customer (x, y, demand, ready, due, service, is_served, is_routed)
    for i in range(20):  # 20 customers
        x = obs[8 + i*8]     # x-coordinate
        y = obs[8 + i*8 + 1] # y-coordinate
        customer_locations.append((x, y))
    
    # Depot coordinates (from C101.csv)
    depot_x, depot_y = 40, 50
    
    # Plot customers
    for i, (x, y) in enumerate(customer_locations):
        plt.scatter(x, y, c='blue', s=100, alpha=0.7)
        plt.text(x, y, str(i+1), fontsize=8, ha='center', va='center')
    
    # Plot depot
    plt.scatter(depot_x, depot_y, c='red', s=200, marker='s')
    plt.text(depot_x, depot_y, 'Depot', fontsize=10, ha='right')
    
    # Plot routes
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for v in range(env.num_vehicles):
        if v < len(solution['routes']):
            route = solution['routes'][v]
            if route:
                # Start from depot
                x_coords = [depot_x]
                y_coords = [depot_y]
                
                # Add customer locations in the order they are visited
                for cust_idx in route:
                    if cust_idx < len(customer_locations):  # Safety check
                        x, y = customer_locations[cust_idx]
                        x_coords.append(x)
                        y_coords.append(y)
                
                # Return to depot
                x_coords.append(depot_x)
                y_coords.append(depot_y)
                
                # Plot route
                plt.plot(x_coords, y_coords, 'o-', color=colors[v % len(colors)], 
                        linewidth=2, markersize=5, label=f'Vehicle {v+1}')
    
    plt.title(f"Vehicle Routes (Reward: {solution['reward']:.2f})")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Solution saved to {filename}")

def get_solution(model_path):
    # Load model
    model = MAAM()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create environment
    env = MVRPSTWEnv(
        num_customers=20,
        num_vehicles=3,
        vehicle_capacity=200,
        data_path='data/raw/C101.csv',
        max_time=1236
    )
    
    obs, _ = env.reset()
    done = False
    total_reward = 0
    routes = [[] for _ in range(env.num_vehicles)]
    
    while not done:
        # Parse observation
        customers = torch.FloatTensor(obs[8:168].reshape(1, 20, 8))
        vehicle_idx = env.current_vehicle
        vehicle = torch.FloatTensor(obs[168:].reshape(3, 4)[vehicle_idx]).unsqueeze(0)
        
        # Create mask
        visited = customers[0, :, -1] > 0.5
        mask = visited.unsqueeze(0)
        
        # Get action
        with torch.no_grad():
            action = model.get_action(customers, vehicle, mask, greedy=True)
        
        # Store action in route
        if action < 20:  # If not a depot action
            routes[vehicle_idx].append(action)
        
        # Step
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    
    return {
        'routes': routes,
        'reward': total_reward,
        'num_customers': 20,
        'num_vehicles': 3
    }

def main():
    print("Generating solution visualization...")
    
    try:
        # Create environment
        env = MVRPSTWEnv(
            num_customers=20,
            num_vehicles=3,
            vehicle_capacity=200,
            data_path='data/raw/C101.csv',
            max_time=1236
        )
        
        solution = get_solution('checkpoints_complete/best_model.pt')
        print(f"Solution found with reward: {solution['reward']:.2f}")
        
        # Save visualization
        output_file = 'vehicle_routes.png'
        plot_solution(env, solution, output_file)
        
        print("\n[SUCCESS] Visualization complete!")
        print(f"Saved to: {output_file}")
        print("\nThe visualization shows:")
        print("- Blue dots: Customers (numbered)")
        print("- Red square: Depot")
        print("- Colored lines: Vehicle routes")
        print("\nYou can find the image in your project folder.")
        
    except Exception as e:
        print("\n[ERROR] An error occurred during visualization:")
        print(f"{str(e)}\n")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error message above and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
