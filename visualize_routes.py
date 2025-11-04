import numpy as np
import matplotlib.pyplot as plt
from env.mvrp_env import MVRPSTWEnv
from stable_baselines3 import PPO

def plot_routes(env=None, model_path=None):
    # Load environment if not provided
    if env is None:
        env = MVRPSTWEnv(
            num_customers=100,
            num_vehicles=5,
            vehicle_capacity=200,
            max_time=1236
        )
    
    # Load model if provided
    if model_path:
        model = PPO.load(model_path)
    
    # Reset environment
    obs, _ = env.reset()
    done = False
    
    # Run simulation
    while not done:
        if model_path:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()  # Random action
            
        obs, reward, done, _, _ = env.step(action)
    
    # Plot routes
    plt.figure(figsize=(12, 8))
    
    # Plot depot (first customer is typically the depot in VRP)
    depot = env.customer_locations[0]
    plt.scatter([depot[0]], [depot[1]], c='red', s=200, marker='s', label='Depot')
    
    # Plot customers
    for i, (x, y) in enumerate(env.customer_locations):
        plt.scatter(x, y, c='blue', s=50, alpha=0.6)
        plt.text(x, y, str(i), fontsize=8)
    
    # Plot routes
    colors = ['green', 'purple', 'orange', 'brown', 'pink']
    for v in range(env.num_vehicles):
        if v < len(env.routes) and len(env.routes[v]) > 0:
            route = [depot] + [env.customer_locations[i+1] for i in env.routes[v]] + [depot]
            route = np.array(route)
            plt.plot(route[:, 0], route[:, 1], 'o-', color=colors[v % len(colors)], 
                    linewidth=2, markersize=6, label=f'Vehicle {v+1}')
    
    plt.title('Vehicle Routes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.savefig('route_visualization.png')
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # 1. Random policy
    print("Visualizing random policy...")
    plot_routes(None)
    
    # 2. Trained model (uncomment after training)
    # print("Visualizing trained model...")
    # plot_routes("checkpoints/best_model")
