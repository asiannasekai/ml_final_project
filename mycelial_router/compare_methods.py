from mycelial_router.core import MycelialRouter
from mycelial_router.astar import AStarPathfinder
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import time

def analyze_path(path: List[Tuple[int, int]], grid: np.ndarray) -> Dict:
    """Analyze a path's properties."""
    total_cost = sum(grid[node].traversal_cost for node in path)
    total_nutrients = sum(grid[node].nutrient_value for node in path)
    path_length = len(path)
    
    return {
        'total_cost': total_cost,
        'total_nutrients': total_nutrients,
        'path_length': path_length,
        'efficiency': total_nutrients / total_cost if total_cost > 0 else 0
    }

def compare_methods():
    # Create environment
    grid_size = (15, 15)  # Smaller grid
    router = MycelialRouter(
        grid_size=grid_size,
        learning_rate=0.3,
        discount_factor=0.95,
        exploration_rate=0.5,  # Higher exploration
        obstacle_prob=0.05,
        failure_prob=0.02,
        nutrient_range=(0.5, 1.0)
    )
    
    # Define start and goal (closer together)
    start = (7, 7)   # Center
    goal = (10, 10)  # Closer to start
    
    # Initialize A* pathfinder
    astar = AStarPathfinder(router.grid, grid_size)
    
    # Find A* path
    print("\nFinding A* path...")
    astar_start_time = time.time()
    astar_path, astar_cost = astar.find_path(start, goal)
    astar_time = time.time() - astar_start_time
    
    if not astar_path:
        print("A* could not find a path to the goal!")
        return
        
    # Analyze A* path
    astar_metrics = analyze_path(astar_path, router.grid)
    
    # Train RL agent
    print("\nTraining RL agent...")
    router.start_growth(start)
    rl_metrics_history = []
    
    n_episodes = 15  # More episodes
    steps_per_episode = 100  # More steps
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        # Reset growth but keep learned Q-values
        router.mycelium = nx.Graph()
        router.mycelium.add_node(start)
        router.grid[start].visited = True
        router.active_tips = [start]
        
        episode_metrics = []
        reached_goal = False
        
        for step in range(steps_per_episode):
            router.grow_step()
            
            # Check if goal is reachable
            if goal in router.mycelium.nodes():
                try:
                    rl_path = nx.shortest_path(router.mycelium, start, goal)
                    metrics = analyze_path(rl_path, router.grid)
                    episode_metrics.append(metrics)
                    reached_goal = True
                    print(f"Step {step + 1}: Found path to goal! Efficiency: {metrics['efficiency']:.2f}")
                    break  # Stop episode once goal is reached
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
        
        if not reached_goal:
            print("Did not reach goal this episode")
        
        rl_metrics_history.append(episode_metrics)
    
    # Get best RL path metrics
    all_metrics = [m for episode in rl_metrics_history for m in episode if m['efficiency'] > 0]
    if not all_metrics:
        print("\nRL agent never found a path to the goal!")
        return
        
    best_rl_metrics = max(all_metrics, key=lambda x: x['efficiency'])
    
    # Print comparison
    print("\nComparison Results:")
    print("==================")
    print("A* Algorithm:")
    print(f"Path length: {astar_metrics['path_length']}")
    print(f"Total cost: {astar_metrics['total_cost']:.2f}")
    print(f"Total nutrients: {astar_metrics['total_nutrients']:.2f}")
    print(f"Efficiency: {astar_metrics['efficiency']:.2f}")
    print(f"Computation time: {astar_time:.4f}s")
    
    print("\nRL Algorithm (Best Path):")
    print(f"Path length: {best_rl_metrics['path_length']}")
    print(f"Total cost: {best_rl_metrics['total_cost']:.2f}")
    print(f"Total nutrients: {best_rl_metrics['total_nutrients']:.2f}")
    print(f"Efficiency: {best_rl_metrics['efficiency']:.2f}")
    
    # Plot learning progress
    plt.figure(figsize=(15, 10))
    
    # Plot efficiency over episodes
    plt.subplot(2, 2, 1)
    max_efficiencies = [max((m['efficiency'] for m in episode), default=0) 
                       for episode in rl_metrics_history]
    plt.plot(max_efficiencies)
    plt.axhline(y=astar_metrics['efficiency'], color='r', linestyle='--', 
                label='A* Efficiency')
    plt.title('Maximum Efficiency per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Efficiency (Nutrients/Cost)')
    plt.legend()
    plt.grid(True)
    
    # Plot path lengths
    plt.subplot(2, 2, 2)
    path_lengths = [max((m['path_length'] for m in episode if m['efficiency'] > 0), default=0) 
                   for episode in rl_metrics_history]
    plt.plot(path_lengths)
    plt.axhline(y=astar_metrics['path_length'], color='r', linestyle='--', 
                label='A* Path Length')
    plt.title('Path Length per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Path Length')
    plt.legend()
    plt.grid(True)
    
    # Plot total nutrients
    plt.subplot(2, 2, 3)
    nutrients = [max((m['total_nutrients'] for m in episode if m['efficiency'] > 0), default=0) 
                for episode in rl_metrics_history]
    plt.plot(nutrients)
    plt.axhline(y=astar_metrics['total_nutrients'], color='r', linestyle='--', 
                label='A* Nutrients')
    plt.title('Total Nutrients per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Nutrients')
    plt.legend()
    plt.grid(True)
    
    # Plot total cost
    plt.subplot(2, 2, 4)
    costs = [min((m['total_cost'] for m in episode if m['efficiency'] > 0), default=float('inf')) 
            for episode in rl_metrics_history]
    plt.plot(costs)
    plt.axhline(y=astar_metrics['total_cost'], color='r', linestyle='--', 
                label='A* Cost')
    plt.title('Total Cost per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_methods() 