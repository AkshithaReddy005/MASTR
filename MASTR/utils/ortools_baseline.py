"""
OR-Tools baseline solver for MVRPSTW
Classical optimization approach for comparison
"""
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from typing import List, Dict, Tuple


class ORToolsVRPSolver:
    """
    Google OR-Tools solver for MVRPSTW
    Provides classical optimization baseline
    """
    
    def __init__(
        self,
        locations: np.ndarray,
        depot: np.ndarray,
        demands: np.ndarray,
        start_times: np.ndarray,
        end_times: np.ndarray,
        num_vehicles: int,
        vehicle_capacity: float,
        time_limit_seconds: int = 30
    ):
        self.locations = np.vstack([depot, locations])  # Depot is index 0
        self.demands = np.concatenate([[0], demands])  # Depot has 0 demand
        self.start_times = np.concatenate([[0], start_times])
        self.end_times = np.concatenate([[1e6], end_times])  # Depot has large time window
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = int(vehicle_capacity)
        self.time_limit_seconds = time_limit_seconds
        
        self.num_locations = len(self.locations)
    
    def _create_distance_matrix(self) -> np.ndarray:
        """Create distance matrix between all locations"""
        num_locs = len(self.locations)
        distance_matrix = np.zeros((num_locs, num_locs))
        
        for i in range(num_locs):
            for j in range(num_locs):
                if i != j:
                    distance_matrix[i, j] = np.linalg.norm(
                        self.locations[i] - self.locations[j]
                    )
        
        # Scale to integers for OR-Tools (multiply by 100 for precision)
        distance_matrix = (distance_matrix * 100).astype(int)
        
        return distance_matrix
    
    def _distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.distance_matrix[from_node][to_node]
    
    def _demand_callback(self, from_index):
        """Returns the demand of the node."""
        from_node = self.manager.IndexToNode(from_index)
        return int(self.demands[from_node])
    
    def _time_callback(self, from_index, to_index):
        """Returns the travel time between nodes (same as distance)."""
        return self._distance_callback(from_index, to_index)
    
    def solve(self) -> Tuple[List[List[int]], float, Dict]:
        """
        Solve MVRPSTW using OR-Tools
        
        Returns:
            routes: List of routes (customer indices, excluding depot)
            total_cost: Total solution cost
            info: Additional solver information
        """
        # Create distance matrix
        self.distance_matrix = self._create_distance_matrix()
        
        # Create routing index manager
        self.manager = pywrapcp.RoutingIndexManager(
            self.num_locations,
            self.num_vehicles,
            0  # Depot index
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(self.manager)
        
        # Create and register distance callback
        transit_callback_index = routing.RegisterTransitCallback(self._distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add capacity constraint
        demand_callback_index = routing.RegisterUnaryTransitCallback(self._demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.vehicle_capacity] * self.num_vehicles,  # vehicle capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Add time window constraints
        time_callback_index = routing.RegisterTransitCallback(self._time_callback)
        
        # Scale time windows to match distance scaling
        time_dimension_name = 'Time'
        routing.AddDimension(
            time_callback_index,
            int(1e6),  # allow waiting time
            int(1e6),  # maximum time per vehicle
            False,  # Don't force start cumul to zero
            time_dimension_name
        )
        time_dimension = routing.GetDimensionOrDie(time_dimension_name)
        
        # Add time window constraints for each location
        for location_idx in range(1, self.num_locations):  # Skip depot
            index = self.manager.NodeToIndex(location_idx)
            # Soft time windows: we'll add penalties instead of hard constraints
            # For now, use wide windows
            time_dimension.CumulVar(index).SetRange(0, int(1e6))
        
        # Set depot time window
        depot_idx = self.manager.NodeToIndex(0)
        time_dimension.CumulVar(depot_idx).SetRange(0, int(1e6))
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = self.time_limit_seconds
        
        # Solve
        solution = routing.SolveWithParameters(search_parameters)
        
        if solution:
            routes, total_cost = self._extract_solution(routing, solution)
            
            info = {
                'status': 'solved',
                'objective_value': solution.ObjectiveValue() / 100.0,  # Unscale
                'num_routes': len(routes)
            }
            
            return routes, total_cost, info
        else:
            return [], float('inf'), {'status': 'no_solution'}
    
    def _extract_solution(self, routing, solution) -> Tuple[List[List[int]], float]:
        """Extract routes from OR-Tools solution"""
        routes = []
        total_distance = 0
        
        for vehicle_id in range(self.num_vehicles):
            route = []
            index = routing.Start(vehicle_id)
            route_distance = 0
            
            while not routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                if node_index != 0:  # Skip depot
                    route.append(node_index - 1)  # Adjust for depot offset
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            
            if route:  # Only add non-empty routes
                routes.append(route)
            total_distance += route_distance
        
        total_cost = total_distance / 100.0  # Unscale
        
        return routes, total_cost


def solve_with_ortools(
    locations: np.ndarray,
    depot: np.ndarray,
    demands: np.ndarray,
    start_times: np.ndarray,
    end_times: np.ndarray,
    num_vehicles: int,
    vehicle_capacity: float,
    time_limit: int = 30
) -> Tuple[List[List[int]], float, Dict]:
    """
    Convenience function to solve MVRPSTW with OR-Tools
    """
    solver = ORToolsVRPSolver(
        locations=locations,
        depot=depot,
        demands=demands,
        start_times=start_times,
        end_times=end_times,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        time_limit_seconds=time_limit
    )
    
    return solver.solve()
