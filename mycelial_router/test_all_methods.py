from mycelial_router.core import MycelialRouter
from mycelial_router.astar import AStarPathfinder
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import time

def analyze_path(path: List[Tuple[int, int]], grid: np.ndarray) -> Dict:
    """Analyze a path's properties."""
    if not path:
        return {
            'total_cost': float('inf'),
            'total_nutrients': 0,
            'path_length': 0,
            'efficiency': 0
        }
        
    total_cost = sum(grid[node].traversal_cost for node in path)
    total_nutrients = sum(grid[node].nutrient_value for node in path)
    path_length = len(path)
    
    return {
        'total_cost': total_cost,
        'total_nutrients': total_nutrients,
        'path_length': path_length,
        'efficiency': total_nutrients / total_cost if total_cost > 0 else 0
    }

def compute_redundancy(graph: nx.Graph, path: List[Tuple[int, int]]) -> int:
    """Compute the number of alternative paths."""
    if not path:
        return 0
        
    start, end = path[0], path[-1]
    
    # Remove the path edges temporarily
    edges_to_remove = list(zip(path[:-1], path[1:]))
    graph.remove_edges_from(edges_to_remove)
    
    # Count alternative paths
    try:
        redundancy = len(list(nx.all_shortest_paths(
            graph, start, end, weight='weight')))
    except nx.NetworkXNoPath:
        redundancy = 0
    
    # Restore the edges
    graph.add_edges_from(edges_to_remove)
    
    return redundancy

def test_all_methods():
    # Initialize the router with default parameters
    router = MycelialRouter(
        grid_size=20,
        nutrient_density=0.3,
        obstacle_density=0.2
    )
    
    # Define start and goal (closer together)
    start = (7, 7)   # Center
    goal = (10, 10)  # Closer to start
    
    # Ensure start and goal are not obstacles
    router.grid[start].is_obstacle = False
    router.grid[goal].is_obstacle = False
    
    # Add start and goal to graph if not already present
    if start not in router.graph:
        router.graph.add_node(start, weight=router.grid[start].traversal_cost)
    if goal not in router.graph:
        router.graph.add_node(goal, weight=router.grid[goal].traversal_cost)
    
    # Add edges for start and goal
    for node in [start, goal]:
        i, j = node
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if (0 <= ni < router.grid_size[0] and 
                0 <= nj < router.grid_size[1] and 
                not router.grid[ni, nj].is_obstacle):
                neighbor = (ni, nj)
                if neighbor not in router.graph:
                    router.graph.add_node(neighbor, weight=router.grid[neighbor].traversal_cost)
                router.graph.add_edge(node, neighbor, 
                                    weight=(router.grid[node].traversal_cost + 
                                          router.grid[neighbor].traversal_cost)/2)
    
    # Initialize A* pathfinder
    astar = AStarPathfinder(router.grid, router.grid_size)
    
    # Find A* path
    print("\nFinding A* path...")
    astar_start_time = time.time()
    astar_path, astar_cost = astar.find_path(start, goal)
    astar_time = time.time() - astar_start_time
    
    # Find Dijkstra path
    print("\nFinding Dijkstra path...")
    dijkstra_start_time = time.time()
    try:
        dijkstra_path = nx.shortest_path(router.graph, start, goal, weight='weight')
        dijkstra_cost = nx.shortest_path_length(router.graph, start, goal, weight='weight')
    except nx.NetworkXNoPath:
        dijkstra_path = []
        dijkstra_cost = float('inf')
    dijkstra_time = time.time() - dijkstra_start_time
    
    # Train RL agent
    print("\nTraining RL agent...")
    router.start_growth(start)
    rl_metrics_history = []
    
    n_episodes = 15  # More episodes
    steps_per_episode = 100
    
    # Create figure for RL visualization
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('RL Agent Growth and Path Finding', fontsize=14)
    
    # Create a progress bar using text
    progress_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    status_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        # Reset growth but keep learned Q-values
        router.mycelium = nx.Graph()
        router.mycelium.add_node(start)
        router.grid[start].visited = True
        router.active_tips = [start]
        
        episode_metrics = []
        reached_goal = False
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Plot grid properties
        nutrient_values = np.array([[router.grid[i, j].nutrient_value for j in range(router.grid_size[1])] 
                                  for i in range(router.grid_size[0])])
        traversal_costs = np.array([[router.grid[i, j].traversal_cost for j in range(router.grid_size[1])] 
                                  for i in range(router.grid_size[0])])
        
        # Plot nutrient distribution with fixed scale
        nutrient_map = ax1.imshow(nutrient_values, cmap='YlGn', vmin=0, vmax=1)
        ax1.set_title('Nutrient Distribution\nGreen = High Nutrients', fontsize=10)
        plt.colorbar(nutrient_map, ax=ax1)
        
        # Plot traversal costs with fixed scale
        cost_map = ax2.imshow(traversal_costs, cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_title('Traversal Costs\nRed = High Cost', fontsize=10)
        plt.colorbar(cost_map, ax=ax2)
        
        # Mark start and goal
        ax1.plot(start[1], start[0], 'bo', markersize=10, label='Start')
        ax1.plot(goal[1], goal[0], 'r*', markersize=10, label='Goal')
        ax2.plot(start[1], start[0], 'bo', markersize=10, label='Start')
        ax2.plot(goal[1], goal[0], 'r*', markersize=10, label='Goal')
        
        # Initialize legend handles
        growth_line = plt.Line2D([], [], color='blue', alpha=0.5, label='Growth')
        path_line = plt.Line2D([], [], color='red', linewidth=2, label='Found Path')
        ax1.add_artist(ax1.legend(handles=[
            plt.Line2D([], [], color='blue', marker='o', label='Start'),
            plt.Line2D([], [], color='red', marker='*', label='Goal'),
            growth_line,
            path_line
        ], loc='upper right'))
        ax2.add_artist(ax2.legend(handles=[
            plt.Line2D([], [], color='blue', marker='o', label='Start'),
            plt.Line2D([], [], color='red', marker='*', label='Goal'),
            growth_line,
            path_line
        ], loc='upper right'))
        
        for step in range(steps_per_episode):
            router.grow_step()
            
            # Calculate progress metrics
            distance_to_goal = np.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
            closest_tip = min(router.active_tips, 
                            key=lambda tip: np.sqrt((goal[0] - tip[0])**2 + (goal[1] - tip[1])**2))
            current_distance = np.sqrt((goal[0] - closest_tip[0])**2 + (goal[1] - closest_tip[1])**2)
            progress = max(0, min(100, 100 * (1 - current_distance/distance_to_goal)))
            
            # Update progress and status text
            progress_text.set_text(f'Episode {episode + 1}/{n_episodes} - Step {step + 1}/{steps_per_episode}')
            status_text.set_text(f'Progress: {progress:.1f}% to goal - Active Tips: {len(router.active_tips)}')
            
            # Update mycelium visualization
            if len(router.mycelium.edges()) > 0:
                edge_coords = np.array([(n1, n2) for n1, n2 in router.mycelium.edges()])
                for edge in edge_coords:
                    ax1.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'b-', alpha=0.5)
                    ax2.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'b-', alpha=0.5)
            
            # Check if goal is reachable
            if goal in router.mycelium.nodes():
                try:
                    rl_path = nx.shortest_path(router.mycelium, start, goal)
                    metrics = analyze_path(rl_path, router.grid)
                    metrics['path'] = rl_path
                    episode_metrics.append(metrics)
                    reached_goal = True
                    
                    # Plot the found path
                    path_coords = np.array([(n1, n2) for n1, n2 in zip(rl_path[:-1], rl_path[1:])])
                    for edge in path_coords:
                        ax1.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'r-', linewidth=2)
                        ax2.plot([edge[0][1], edge[1][1]], [edge[0][0], edge[1][0]], 'r-', linewidth=2)
                    
                    status_text.set_text(f'GOAL REACHED! Efficiency: {metrics["efficiency"]}')
                    print(f"Step {step + 1}: Found path to goal! Efficiency: {metrics['efficiency']}")
                    break
                except nx.NetworkXNoPath:
                    episode_metrics.append({
                        'total_cost': float('inf'),
                        'total_nutrients': 0,
                        'path_length': 0,
                        'efficiency': 0
                    })
            else:
                episode_metrics.append({
                    'total_cost': float('inf'),
                    'total_nutrients': 0,
                    'path_length': 0,
                    'efficiency': 0
                })
            
            if not reached_goal and (step + 1) % 10 == 0:
                print(f"Step {step + 1}: Still exploring... Active tips: {len(router.active_tips)}")
            
            # Add grid lines for better visibility
            ax1.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax2.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Update plot
            plt.pause(0.01)
        
        if not reached_goal:
            status_text.set_text("Did not reach goal this episode")
            print("Did not reach goal this episode")
        
        rl_metrics_history.append(episode_metrics)
        plt.pause(1)  # Pause to show final state
    
    plt.ioff()  # Turn off interactive mode
    
    # Get best RL path metrics
    all_metrics = [m for episode in rl_metrics_history for m in episode if m['efficiency'] > 0]
    if not all_metrics:
        print("\nRL agent never found a path to the goal!")
        return
        
    best_rl_metrics = max(all_metrics, key=lambda x: x['efficiency'])
    
    # Analyze paths
    astar_metrics = analyze_path(astar_path, router.grid)
    dijkstra_metrics = analyze_path(dijkstra_path, router.grid)
    
    # Compute redundancies
    astar_redundancy = compute_redundancy(router.graph, astar_path)
    dijkstra_redundancy = compute_redundancy(router.graph, dijkstra_path)
    rl_redundancy = compute_redundancy(router.mycelium, best_rl_metrics['path'])
    
    # Calculate total computation time for RL
    rl_time = sum(episode['time'] for episode in rl_metrics_history if episode and 'time' in episode)
    
    # Print comparison
    print("\nComparison Results:")
    print("==================")
    print("A* Algorithm:")
    print(f"Path length: {astar_metrics['path_length']}")
    print(f"Total cost: {astar_metrics['total_cost']:.2f}")
    print(f"Total nutrients: {astar_metrics['total_nutrients']:.2f}")
    print(f"Efficiency: {astar_metrics['efficiency']:.2f}")
    print(f"Redundancy: {astar_redundancy}")
    print(f"Computation time: {astar_time:.4f}s")
    
    print("\nDijkstra's Algorithm:")
    print(f"Path length: {dijkstra_metrics['path_length']}")
    print(f"Total cost: {dijkstra_metrics['total_cost']:.2f}")
    print(f"Total nutrients: {dijkstra_metrics['total_nutrients']:.2f}")
    print(f"Efficiency: {dijkstra_metrics['efficiency']:.2f}")
    print(f"Redundancy: {dijkstra_redundancy}")
    print(f"Computation time: {dijkstra_time:.4f}s")
    
    print("\nRL Algorithm (Best Path):")
    print(f"Path length: {best_rl_metrics['path_length']}")
    print(f"Total cost: {best_rl_metrics['total_cost']:.2f}")
    print(f"Total nutrients: {best_rl_metrics['total_nutrients']:.2f}")
    print(f"Efficiency: {best_rl_metrics['efficiency']:.2f}")
    print(f"Redundancy: {rl_redundancy}")
    print(f"Computation time: {rl_time:.4f}s")
    
    # Plot learning progress
    plt.figure(figsize=(15, 10))
    
    # Plot efficiency over episodes
    plt.subplot(2, 2, 1)
    max_efficiencies = [max((m['efficiency'] for m in episode), default=0) 
                       for episode in rl_metrics_history]
    plt.plot(max_efficiencies, 'b-', label='RL Efficiency')
    plt.axhline(y=astar_metrics['efficiency'], color='r', linestyle='--', 
                label='A* Efficiency')
    plt.axhline(y=dijkstra_metrics['efficiency'], color='g', linestyle='--', 
                label='Dijkstra Efficiency')
    plt.title('Maximum Efficiency per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency (Nutrients/Cost)')
    plt.legend()
    plt.grid(True)
    
    # Plot path lengths over episodes
    plt.subplot(2, 2, 2)
    path_lengths = [max((m['path_length'] for m in episode if m['efficiency'] > 0), default=0) 
                   for episode in rl_metrics_history]
    plt.plot(path_lengths, 'b-', label='RL Path Length')
    plt.axhline(y=astar_metrics['path_length'], color='r', linestyle='--', 
                label='A* Path Length')
    plt.axhline(y=dijkstra_metrics['path_length'], color='g', linestyle='--', 
                label='Dijkstra Path Length')
    plt.title('Path Length per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Path Length')
    plt.legend()
    plt.grid(True)
    
    # Plot total nutrients over episodes
    plt.subplot(2, 2, 3)
    nutrients = [max((m['total_nutrients'] for m in episode if m['efficiency'] > 0), default=0) 
                for episode in rl_metrics_history]
    plt.plot(nutrients, 'b-', label='RL Nutrients')
    plt.axhline(y=astar_metrics['total_nutrients'], color='r', linestyle='--', 
                label='A* Nutrients')
    plt.axhline(y=dijkstra_metrics['total_nutrients'], color='g', linestyle='--', 
                label='Dijkstra Nutrients')
    plt.title('Total Nutrients per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Nutrients')
    plt.legend()
    plt.grid(True)
    
    # Plot total cost over episodes
    plt.subplot(2, 2, 4)
    costs = [min((m['total_cost'] for m in episode if m['efficiency'] > 0), default=float('inf')) 
            for episode in rl_metrics_history]
    plt.plot(costs, 'b-', label='RL Cost')
    plt.axhline(y=astar_metrics['total_cost'], color='r', linestyle='--', 
                label='A* Cost')
    plt.axhline(y=dijkstra_metrics['total_cost'], color='g', linestyle='--', 
                label='Dijkstra Cost')
    plt.title('Total Cost per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_all_methods() 