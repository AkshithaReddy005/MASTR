"""
Evaluation metrics for MVRPSTW
Includes cost calculation, constraint violations, and comparison utilities
"""
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_route_distance(route: List[int], locations: np.ndarray, depot: np.ndarray) -> float:
    """
    Calculate total distance for a single route
    
    Args:
        route: List of customer indices
        locations: Customer locations [N, 2]
        depot: Depot location [2]
    
    Returns:
        total_distance: Total route distance
    """
    if len(route) == 0:
        return 0.0
    
    total_distance = 0.0
    current_pos = depot
    
    for customer_idx in route:
        customer_pos = locations[customer_idx]
        distance = np.linalg.norm(customer_pos - current_pos)
        total_distance += distance
        current_pos = customer_pos
    
    # Return to depot
    total_distance += np.linalg.norm(current_pos - depot)
    
    return total_distance


def calculate_time_window_penalties(
    route: List[int],
    locations: np.ndarray,
    depot: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    penalties_early: np.ndarray,
    penalties_late: np.ndarray
) -> Tuple[float, int, int]:
    """
    Calculate time window penalties for a route
    
    Returns:
        total_penalty: Total time window penalty
        num_early: Number of early arrivals
        num_late: Number of late arrivals
    """
    if len(route) == 0:
        return 0.0, 0, 0
    
    total_penalty = 0.0
    num_early = 0
    num_late = 0
    current_time = 0.0
    current_pos = depot
    
    for customer_idx in route:
        customer_pos = locations[customer_idx]
        travel_time = np.linalg.norm(customer_pos - current_pos)
        arrival_time = current_time + travel_time
        
        # Check time window
        if arrival_time < start_times[customer_idx]:
            # Early arrival
            penalty = penalties_early[customer_idx] * (start_times[customer_idx] - arrival_time)
            total_penalty += penalty
            num_early += 1
        elif arrival_time > end_times[customer_idx]:
            # Late arrival
            penalty = penalties_late[customer_idx] * (arrival_time - end_times[customer_idx])
            total_penalty += penalty
            num_late += 1
        
        # Update state
        current_time = arrival_time + 10.0  # Service time
        current_pos = customer_pos
    
    return total_penalty, num_early, num_late


def calculate_capacity_violations(
    route: List[int],
    demands: np.ndarray,
    vehicle_capacity: float
) -> Tuple[float, bool]:
    """
    Check capacity constraint violations
    
    Returns:
        total_demand: Total demand on route
        is_violated: Whether capacity is violated
    """
    total_demand = sum(demands[i] for i in route)
    is_violated = total_demand > vehicle_capacity
    return total_demand, is_violated


def evaluate_solution(
    routes: List[List[int]],
    locations: np.ndarray,
    depot: np.ndarray,
    demands: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    penalties_early: np.ndarray,
    penalties_late: np.ndarray,
    vehicle_capacity: float
) -> Dict:
    """
    Comprehensive solution evaluation
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    total_distance = 0.0
    total_penalty = 0.0
    total_early = 0
    total_late = 0
    capacity_violations = 0
    
    route_metrics = []
    
    for route in routes:
        # Distance
        distance = calculate_route_distance(route, locations, depot)
        total_distance += distance
        
        # Time window penalties
        penalty, num_early, num_late = calculate_time_window_penalties(
            route, locations, depot, start_times, end_times,
            penalties_early, penalties_late
        )
        total_penalty += penalty
        total_early += num_early
        total_late += num_late
        
        # Capacity
        demand, violated = calculate_capacity_violations(route, demands, vehicle_capacity)
        if violated:
            capacity_violations += 1
        
        route_metrics.append({
            'distance': distance,
            'penalty': penalty,
            'demand': demand,
            'num_customers': len(route),
            'capacity_violated': violated
        })
    
    # Total cost
    total_cost = total_distance + total_penalty
    
    # Check if all customers are served
    all_customers = set()
    for route in routes:
        all_customers.update(route)
    num_served = len(all_customers)
    
    metrics = {
        'total_cost': total_cost,
        'total_distance': total_distance,
        'total_penalty': total_penalty,
        'num_early_arrivals': total_early,
        'num_late_arrivals': total_late,
        'capacity_violations': capacity_violations,
        'num_customers_served': num_served,
        'num_routes': len(routes),
        'route_metrics': route_metrics
    }
    
    return metrics


def compare_solutions(solution1_metrics: Dict, solution2_metrics: Dict, names: Tuple[str, str] = ("Solution 1", "Solution 2")):
    """
    Compare two solutions and print comparison
    """
    print(f"\n{'='*60}")
    print(f"Solution Comparison: {names[0]} vs {names[1]}")
    print(f"{'='*60}")
    
    metrics_to_compare = [
        ('total_cost', 'Total Cost', '{:.2f}'),
        ('total_distance', 'Total Distance', '{:.2f}'),
        ('total_penalty', 'Time Window Penalty', '{:.2f}'),
        ('num_early_arrivals', 'Early Arrivals', '{}'),
        ('num_late_arrivals', 'Late Arrivals', '{}'),
        ('capacity_violations', 'Capacity Violations', '{}'),
        ('num_customers_served', 'Customers Served', '{}'),
    ]
    
    for key, label, fmt in metrics_to_compare:
        val1 = solution1_metrics[key]
        val2 = solution2_metrics[key]
        diff = val2 - val1
        diff_pct = (diff / val1 * 100) if val1 != 0 else 0
        
        print(f"{label:.<30} {fmt.format(val1):>10} | {fmt.format(val2):>10} | {diff:>+10.2f} ({diff_pct:>+6.1f}%)")
    
    print(f"{'='*60}\n")


def plot_routes(
    routes: List[List[int]],
    locations: np.ndarray,
    depot: np.ndarray,
    title: str = "Vehicle Routes",
    save_path: str = None
):
    """
    Visualize routes on a 2D plot
    """
    plt.figure(figsize=(10, 10))
    
    # Plot depot
    plt.scatter(depot[0], depot[1], c='red', s=300, marker='s', label='Depot', zorder=3)
    
    # Plot customers
    plt.scatter(locations[:, 0], locations[:, 1], c='gray', s=100, alpha=0.5, label='Customers', zorder=2)
    
    # Plot routes
    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
    
    for i, route in enumerate(routes):
        if len(route) == 0:
            continue
        
        # Build route path
        path = [depot]
        for customer_idx in route:
            path.append(locations[customer_idx])
        path.append(depot)
        
        path = np.array(path)
        
        # Plot route
        plt.plot(path[:, 0], path[:, 1], c=colors[i], linewidth=2, alpha=0.7, label=f'Vehicle {i+1}', zorder=1)
        plt.scatter(path[1:-1, 0], path[1:-1, 1], c=[colors[i]], s=150, edgecolors='black', linewidths=1.5, zorder=2)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(log_file: str, save_path: str = None):
    """
    Plot training curves from tensorboard logs
    """
    # This would typically use tensorboard data
    # For now, placeholder for structure
    pass


def print_solution_summary(metrics: Dict):
    """
    Print formatted solution summary
    """
    print("\n" + "="*60)
    print("SOLUTION SUMMARY")
    print("="*60)
    print(f"Total Cost:              {metrics['total_cost']:.2f}")
    print(f"  - Distance:            {metrics['total_distance']:.2f}")
    print(f"  - Time Penalties:      {metrics['total_penalty']:.2f}")
    print(f"\nConstraint Violations:")
    print(f"  - Early Arrivals:      {metrics['num_early_arrivals']}")
    print(f"  - Late Arrivals:       {metrics['num_late_arrivals']}")
    print(f"  - Capacity Violations: {metrics['capacity_violations']}")
    print(f"\nSolution Stats:")
    print(f"  - Customers Served:    {metrics['num_customers_served']}")
    print(f"  - Number of Routes:    {metrics['num_routes']}")
    print("\nRoute Details:")
    for i, route_metric in enumerate(metrics['route_metrics']):
        print(f"  Vehicle {i+1}:")
        print(f"    Customers: {route_metric['num_customers']}")
        print(f"    Distance:  {route_metric['distance']:.2f}")
        print(f"    Demand:    {route_metric['demand']:.2f}")
        print(f"    Penalty:   {route_metric['penalty']:.2f}")
        if route_metric['capacity_violated']:
            print(f"    ⚠ CAPACITY VIOLATED")
    print("="*60 + "\n")
