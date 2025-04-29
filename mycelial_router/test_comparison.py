import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from mycelial_router.core import MycelialRouter
from mycelial_router.astar import AStarPathfinder

def analyze_path(router: MycelialRouter, path: List[Tuple[int, int]]) -> Dict:
    """Analyze a path's properties."""
    if not path:
        return {
            'length': float('inf'),
            'total_cost': float('inf'),
            'total_nutrients': 0,
            'efficiency': 0
        }
        
    total_cost = sum(router.grid[node[0], node[1]].traversal_cost for node in path)
    total_nutrients = sum(router.grid[node[0], node[1]].nutrient_value for node in path)
    efficiency = total_nutrients / total_cost if total_cost > 0 else 0
    
    return {
        'length': len(path),
        'total_cost': total_cost,
        'total_nutrients': total_nutrients,
        'efficiency': efficiency
    }

def test_all_algorithms(grid_size: Tuple[int, int] = (15, 15),
                       start: Tuple[int, int] = (0, 0),
                       goal: Tuple[int, int] = (14, 14),
                       num_episodes: int = 15,
                       show_plots: bool = True):
    """Compare mycelial growth with A* and Dijkstra's algorithm."""
    print("\nComparing pathfinding algorithms...")
    
    # Initialize router
    router = MycelialRouter(
        grid_size=grid_size,
        nutrient_range=(0.1, 1.0),
        cost_range=(0.1, 1.0),
        obstacle_prob=0.1,
        failure_prob=0.05,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2
    )
    
    # Initialize visualization
    if show_plots:
        plt.ion()  # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
    # Track metrics for each algorithm
    metrics = {
        'astar': None,
        'dijkstra': None,
        'rl': []  # List to track RL performance over episodes
    }
    
    # 1. Test A* algorithm
    print("\nTesting A* algorithm...")
    astar = AStarPathfinder(router.grid, grid_size)
    start_time = time.time()
    astar_path, astar_cost = astar.find_path(start, goal)
    astar_time = time.time() - start_time
    metrics['astar'] = analyze_path(router, astar_path)
    metrics['astar']['time'] = astar_time
    print(f"A* path found in {astar_time:.4f} seconds")
    
    # 2. Test Dijkstra's algorithm
    print("\nTesting Dijkstra's algorithm...")
    start_time = time.time()
    try:
        dijkstra_path = nx.shortest_path(router.graph, start, goal, weight='weight')
        dijkstra_time = time.time() - start_time
        metrics['dijkstra'] = analyze_path(router, dijkstra_path)
        metrics['dijkstra']['time'] = dijkstra_time
        print(f"Dijkstra path found in {dijkstra_time:.4f} seconds")
    except nx.NetworkXNoPath:
        print("No path found using Dijkstra's algorithm")
        metrics['dijkstra'] = {
            'length': float('inf'),
            'total_cost': float('inf'),
            'total_nutrients': 0,
            'efficiency': 0,
            'time': time.time() - start_time
        }
    
    # 3. Test RL-based mycelial growth
    print("\nTesting RL-based mycelial growth...")
    best_rl_metrics = None
    episode_metrics = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset mycelium for new episode
        router.mycelium = nx.Graph()
        router.active_tips = []
        router.start_growth([start])
        
        start_time = time.time()
        current_node = start
        path = [start]
        step = 0
        max_steps = grid_size[0] * grid_size[1]  # Maximum steps before giving up
        
        while current_node != goal and step < max_steps:
            # Get valid actions and choose next node
            valid_actions = router.rl_agent.get_valid_actions(current_node)
            action = router.rl_agent.choose_action(current_node, valid_actions)
            
            # Apply action
            next_node = (current_node[0] + action[0], current_node[1] + action[1])
            
            # Update visualization
            if show_plots and step % 5 == 0:
                ax1.clear()
                ax2.clear()
                
                # Plot nutrient distribution
                nutrient_map = np.array([[router.grid[i, j].nutrient_value 
                                        for j in range(grid_size[1])]
                                       for i in range(grid_size[0])])
                ax1.imshow(nutrient_map, cmap='YlOrRd')
                ax1.plot([n[1] for n in path], [n[0] for n in path], 'b-', label='Path')
                ax1.plot(start[1], start[0], 'go', label='Start')
                ax1.plot(goal[1], goal[0], 'ro', label='Goal')
                ax1.set_title('Nutrient Distribution and Path')
                ax1.legend()
                
                # Plot traversal costs
                cost_map = np.array([[router.grid[i, j].traversal_cost 
                                    for j in range(grid_size[1])]
                                   for i in range(grid_size[0])])
                ax2.imshow(cost_map, cmap='viridis')
                ax2.plot([n[1] for n in path], [n[0] for n in path], 'r-', label='Path')
                ax2.plot(start[1], start[0], 'go', label='Start')
                ax2.plot(goal[1], goal[0], 'ro', label='Goal')
                ax2.set_title('Traversal Costs and Path')
                ax2.legend()
                
                plt.draw()
                plt.pause(0.1)
            
            # Check if next node is valid
            if next_node in router._get_neighbors(current_node):
                # Update path and current node
                path.append(next_node)
                current_node = next_node
                
                # Compute reward and update Q-values
                reward = router._compute_reward(next_node)
                router.rl_agent.update_q_value(current_node, action, reward, next_node)
            
            step += 1
        
        # Analyze episode results
        episode_time = time.time() - start_time
        if current_node == goal:
            print(f"Goal reached in {step} steps!")
            episode_metric = analyze_path(router, path)
            episode_metric['time'] = episode_time
            episode_metric['steps'] = step
            episode_metrics.append(episode_metric)
            
            # Update best metrics
            if (best_rl_metrics is None or 
                episode_metric['efficiency'] > best_rl_metrics['efficiency']):
                best_rl_metrics = episode_metric
                best_rl_metrics['path'] = path
        else:
            print(f"Failed to reach goal in episode {episode + 1}")
            episode_metrics.append({
                'length': float('inf'),
                'total_cost': float('inf'),
                'total_nutrients': 0,
                'efficiency': 0,
                'time': episode_time,
                'steps': step
            })
    
    # Plot comparison results
    if show_plots:
        plt.ioff()  # Disable interactive mode
        plt.figure(figsize=(15, 10))
        
        # Plot path lengths
        plt.subplot(2, 2, 1)
        algorithms = ['A*', 'Dijkstra', 'RL (Best)']
        lengths = [metrics['astar']['length'], 
                  metrics['dijkstra']['length'],
                  best_rl_metrics['length']]
        plt.bar(algorithms, lengths)
        plt.title('Path Length Comparison')
        plt.ylabel('Number of Steps')
        
        # Plot total costs
        plt.subplot(2, 2, 2)
        costs = [metrics['astar']['total_cost'],
                metrics['dijkstra']['total_cost'],
                best_rl_metrics['total_cost']]
        plt.bar(algorithms, costs)
        plt.title('Total Cost Comparison')
        plt.ylabel('Cost')
        
        # Plot efficiency
        plt.subplot(2, 2, 3)
        efficiency = [metrics['astar']['efficiency'],
                     metrics['dijkstra']['efficiency'],
                     best_rl_metrics['efficiency']]
        plt.bar(algorithms, efficiency)
        plt.title('Efficiency Comparison')
        plt.ylabel('Nutrients/Cost')
        
        # Plot RL learning progress
        plt.subplot(2, 2, 4)
        episodes = range(1, num_episodes + 1)
        efficiencies = [m['efficiency'] for m in episode_metrics]
        plt.plot(episodes, efficiencies, 'b-')
        plt.title('RL Learning Progress')
        plt.xlabel('Episode')
        plt.ylabel('Efficiency')
        
        plt.tight_layout()
        plt.show()
    
    # Print final comparison
    print("\nFinal Comparison:")
    print("\nA* Algorithm:")
    print(f"Path length: {metrics['astar']['length']}")
    print(f"Total cost: {metrics['astar']['total_cost']:.2f}")
    print(f"Total nutrients: {metrics['astar']['total_nutrients']:.2f}")
    print(f"Efficiency: {metrics['astar']['efficiency']:.2f}")
    print(f"Computation time: {metrics['astar']['time']:.4f}s")
    
    print("\nDijkstra's Algorithm:")
    print(f"Path length: {metrics['dijkstra']['length']}")
    print(f"Total cost: {metrics['dijkstra']['total_cost']:.2f}")
    print(f"Total nutrients: {metrics['dijkstra']['total_nutrients']:.2f}")
    print(f"Efficiency: {metrics['dijkstra']['efficiency']:.2f}")
    print(f"Computation time: {metrics['dijkstra']['time']:.4f}s")
    
    print("\nRL-based Mycelial Growth (Best Episode):")
    print(f"Path length: {best_rl_metrics['length']}")
    print(f"Total cost: {best_rl_metrics['total_cost']:.2f}")
    print(f"Total nutrients: {best_rl_metrics['total_nutrients']:.2f}")
    print(f"Efficiency: {best_rl_metrics['efficiency']:.2f}")
    print(f"Computation time: {best_rl_metrics['time']:.4f}s")

if __name__ == "__main__":
    test_all_algorithms() 