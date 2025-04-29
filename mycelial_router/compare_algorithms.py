import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
from mycelial_router.core import MycelialRouter
from mycelial_router.enhanced_mycelium import EnhancedMycelium
from mycelial_router.rl_agent import MyceliumRLAgent
from tqdm import tqdm
import random

def run_comparison():
    """Run a comparison of different pathfinding algorithms."""
    # Initialize metrics
    metrics = {
        'A*': {'success_rate': [], 'path_lengths': [], 'times': [], 'dynamic_success': []},
        'Dijkstra': {'success_rate': [], 'path_lengths': [], 'times': [], 'dynamic_success': []},
        'RL': {'success_rate': [], 'path_lengths': [], 'times': [], 'dynamic_success': []},
        'Enhanced': {'success_rate': [], 'path_lengths': [], 'times': [], 'goal_proximity': [], 'dynamic_success': []}
    }
    
    num_tests = 20
    grid_size = 20
    
    for test in range(num_tests):
        print(f"\nRunning test {test + 1}/{num_tests}")
        
        # Create test environment
        env = MycelialRouter(grid_size=grid_size, obstacle_density=0.2)
        
        # Generate random start and goal positions that are not obstacles
        while True:
            start = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            goal = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            if (not env.grid[start].is_obstacle and 
                not env.grid[goal].is_obstacle and 
                start != goal):
                break
        
        # Test A*
        start_time = time.time()
        try:
            path = nx.astar_path(env.graph, start, goal, heuristic=lambda u, v: abs(u[0]-v[0]) + abs(u[1]-v[1]))
            metrics['A*']['success_rate'].append(1)
            metrics['A*']['path_lengths'].append(len(path))
        except nx.NetworkXNoPath:
            metrics['A*']['success_rate'].append(0)
            metrics['A*']['path_lengths'].append(0)
        metrics['A*']['times'].append(time.time() - start_time)
        
        # Test Dijkstra
        start_time = time.time()
        try:
            path = nx.dijkstra_path(env.graph, start, goal)
            metrics['Dijkstra']['success_rate'].append(1)
            metrics['Dijkstra']['path_lengths'].append(len(path))
        except nx.NetworkXNoPath:
            metrics['Dijkstra']['success_rate'].append(0)
            metrics['Dijkstra']['path_lengths'].append(0)
        metrics['Dijkstra']['times'].append(time.time() - start_time)
        
        # Test RL
        start_time = time.time()
        agent = MyceliumRLAgent(grid_size=grid_size, start=start, goal=goal)
        current = start
        path_length = 0
        success = False
        
        for _ in range(100):  # Limit steps
            # Get valid actions for current position
            valid_actions = []
            for action in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
                next_pos = (current[0] + action[0], current[1] + action[1])
                if (0 <= next_pos[0] < grid_size and 
                    0 <= next_pos[1] < grid_size and 
                    not env.grid[next_pos].is_obstacle):
                    valid_actions.append(action)
            
            if not valid_actions:
                break
                
            action = agent.choose_action(current, valid_actions)
            next_pos = (current[0] + action[0], current[1] + action[1])
            current = next_pos
            path_length += 1
            
            if current == goal:
                success = True
                break
        
        metrics['RL']['success_rate'].append(1 if success else 0)
        metrics['RL']['path_lengths'].append(path_length if success else 0)
        metrics['RL']['times'].append(time.time() - start_time)
        
        # Test Enhanced Mycelium
        start_time = time.time()
        mycelium = EnhancedMycelium(
            grid_size=grid_size,
            nutrient_density=0.3,
            obstacle_density=0.2,
            goal_direction_weight=0.6,
            exploration_weight=0.2,
            nutrient_weight=0.2
        )
        mycelium.start_growth(start, goal)
        
        # Run growth for a limited number of steps
        max_steps = 50
        for _ in range(max_steps):
            mycelium.grow_step()
            if mycelium.goal_reached:
                break
        
        # Calculate success and metrics
        success = mycelium.goal_reached
        metrics['Enhanced']['success_rate'].append(1 if success else 0)
        
        if success:
            # Find path length if goal was reached
            try:
                if goal in mycelium.mycelium.nodes():
                    path = nx.shortest_path(mycelium.mycelium, start, goal)
                    metrics['Enhanced']['path_lengths'].append(len(path))
                else:
                    metrics['Enhanced']['path_lengths'].append(0)
            except nx.NetworkXNoPath:
                metrics['Enhanced']['path_lengths'].append(0)
        else:
            # Calculate proximity to goal
            min_dist = float('inf')
            for node in mycelium.mycelium.nodes():
                dist = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
                min_dist = min(min_dist, dist)
            metrics['Enhanced']['goal_proximity'].append(min_dist)
            metrics['Enhanced']['path_lengths'].append(0)
        
        metrics['Enhanced']['times'].append(time.time() - start_time)
        
        # Test dynamic environment adaptation
        print("\nTesting dynamic environment adaptation...")
        # Create a new environment with moving obstacles
        dynamic_env = MycelialRouter(grid_size=grid_size, obstacle_density=0.2)
        
        # Initialize graph for dynamic environment
        dynamic_env.graph = nx.Graph()
        for i in range(grid_size):
            for j in range(grid_size):
                if not dynamic_env.grid[(i, j)].is_obstacle:
                    dynamic_env.graph.add_node((i, j))
                    # Add edges to previous nodes if they exist and aren't obstacles
                    if i > 0 and not dynamic_env.grid[(i-1, j)].is_obstacle:
                        dynamic_env.graph.add_edge((i, j), (i-1, j))
                    if j > 0 and not dynamic_env.grid[(i, j-1)].is_obstacle:
                        dynamic_env.graph.add_edge((i, j), (i, j-1))
        
        # Ensure start and goal are valid in dynamic environment
        max_attempts = 10
        dynamic_goal = None
        for _ in range(max_attempts):
            temp_goal = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            if (temp_goal != start and 
                not dynamic_env.grid[temp_goal].is_obstacle and 
                temp_goal in dynamic_env.graph):
                dynamic_goal = temp_goal
                break
        
        if dynamic_goal is None:
            print("Could not find valid dynamic goal, skipping dynamic test")
            for algo in metrics:
                metrics[algo]['dynamic_success'].append(0)
            continue
        
        # Test RL in dynamic environment
        dynamic_agent = MyceliumRLAgent(grid_size=grid_size, start=start, goal=dynamic_goal)
        current = start
        success = False
        steps = 0
        max_steps = 50  # Reduced from 100 to make test faster
        
        for _ in range(max_steps):
            # Move some obstacles randomly
            if steps % 5 == 0:  # Change environment more frequently but with fewer changes
                for _ in range(2):  # Move 2 obstacles
                    old_pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
                    new_pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
                    if (dynamic_env.grid[old_pos].is_obstacle and 
                        not dynamic_env.grid[new_pos].is_obstacle and
                        new_pos != current and new_pos != dynamic_goal):
                        # Update grid
                        dynamic_env.grid[old_pos].is_obstacle = False
                        dynamic_env.grid[new_pos].is_obstacle = True
                        
                        # Update graph structure
                        if old_pos in dynamic_env.graph:
                            dynamic_env.graph.remove_node(old_pos)
                        if new_pos not in dynamic_env.graph:
                            dynamic_env.graph.add_node(new_pos)
                            # Add edges to non-obstacle neighbors
                            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                ni, nj = new_pos[0] + di, new_pos[1] + dj
                                if (0 <= ni < grid_size and 0 <= nj < grid_size and 
                                    not dynamic_env.grid[(ni, nj)].is_obstacle):
                                    dynamic_env.graph.add_edge(new_pos, (ni, nj))
            
            # Get valid actions
            valid_actions = []
            for action in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + action[0], current[1] + action[1])
                if (0 <= next_pos[0] < grid_size and 
                    0 <= next_pos[1] < grid_size and 
                    not dynamic_env.grid[next_pos].is_obstacle):
                    valid_actions.append(action)
            
            if not valid_actions:
                break
            
            action = dynamic_agent.choose_action(current, valid_actions)
            next_pos = (current[0] + action[0], current[1] + action[1])
            current = next_pos
            steps += 1
            
            if current == dynamic_goal:
                success = True
                break
        
        metrics['RL']['dynamic_success'].append(1 if success else 0)
        
        # Test other algorithms in dynamic environment
        for algo in ['A*', 'Dijkstra', 'Enhanced']:
            if algo == 'Enhanced':
                dynamic_mycelium = EnhancedMycelium(
                    grid_size=grid_size,
                    nutrient_density=0.3,
                    obstacle_density=0.2,
                    goal_direction_weight=0.6,
                    exploration_weight=0.2,
                    nutrient_weight=0.2
                )
                dynamic_mycelium.start_growth(start, dynamic_goal)
                for _ in range(30):  # Reduced from 50
                    dynamic_mycelium.grow_step()
                    if dynamic_mycelium.goal_reached:
                        break
                metrics[algo]['dynamic_success'].append(1 if dynamic_mycelium.goal_reached else 0)
            else:
                try:
                    if algo == 'A*':
                        path = nx.astar_path(dynamic_env.graph, start, dynamic_goal)
                    else:
                        path = nx.dijkstra_path(dynamic_env.graph, start, dynamic_goal)
                    metrics[algo]['dynamic_success'].append(1)
                except nx.NetworkXNoPath:
                    metrics[algo]['dynamic_success'].append(0)
    
    # Print results
    print("\nAlgorithm Comparison Results:")
    print("=" * 50)
    
    for algo in metrics:
        success_rate = np.mean(metrics[algo]['success_rate']) * 100
        avg_path_length = np.mean([l for l in metrics[algo]['path_lengths'] if l > 0] or [0])
        avg_time = np.mean(metrics[algo]['times'])
        dynamic_success = np.mean(metrics[algo]['dynamic_success']) * 100
        
        print(f"\n{algo}:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Path Length: {avg_path_length:.1f}")
        print(f"Average Time: {avg_time:.3f} seconds")
        print(f"Dynamic Environment Success Rate: {dynamic_success:.1f}%")
        
        if algo == 'Enhanced':
            avg_proximity = np.mean(metrics[algo]['goal_proximity'])
            print(f"Average Distance to Goal (when not reached): {avg_proximity:.1f} steps")
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Success Rate
    plt.subplot(131)
    success_rates = [np.mean(metrics[algo]['success_rate']) * 100 for algo in metrics]
    plt.bar(metrics.keys(), success_rates)
    plt.title('Success Rate')
    plt.ylabel('Percentage')
    
    # Path Length
    plt.subplot(132)
    path_lengths = [np.mean([l for l in metrics[algo]['path_lengths'] if l > 0] or [0]) for algo in metrics]
    plt.bar(metrics.keys(), path_lengths)
    plt.title('Average Path Length')
    plt.ylabel('Steps')
    
    # Dynamic Success Rate
    plt.subplot(133)
    dynamic_success = [np.mean(metrics[algo]['dynamic_success']) * 100 for algo in metrics]
    plt.bar(metrics.keys(), dynamic_success)
    plt.title('Dynamic Environment Success Rate')
    plt.ylabel('Percentage')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.close()

if __name__ == "__main__":
    run_comparison() 