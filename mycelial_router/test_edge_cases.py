from mycelial_router.core import MycelialRouter
from mycelial_router.astar import AStarPathfinder
import networkx as nx
import numpy as np
import time

def test_edge_cases():
    print("\nTesting Edge Cases...")
    
    # Test Case 1: Dense Obstacles
    print("\n1. Testing with Dense Obstacles (80% obstacle density)")
    router = MycelialRouter(
        grid_size=20,
        nutrient_density=0.3,
        obstacle_density=0.8  # Very high obstacle density
    )
    start = (0, 0)
    goal = (19, 19)
    test_pathfinding(router, start, goal, "Dense Obstacles")

    # Test Case 2: No Obstacles
    print("\n2. Testing with No Obstacles")
    router = MycelialRouter(
        grid_size=20,
        nutrient_density=0.3,
        obstacle_density=0.0  # No obstacles
    )
    test_pathfinding(router, start, goal, "No Obstacles")

    # Test Case 3: Start/Goal Adjacent to Obstacles
    print("\n3. Testing with Start/Goal Adjacent to Obstacles")
    router = MycelialRouter(
        grid_size=20,
        nutrient_density=0.3,
        obstacle_density=0.2
    )
    # Force obstacles around start and goal
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        x, y = 5 + dx, 5 + dy
        if 0 <= x < 20 and 0 <= y < 20:
            router.grid[x, y].is_obstacle = True
    start = (5, 5)
    goal = (15, 15)
    test_pathfinding(router, start, goal, "Adjacent Obstacles")

    # Test Case 4: Narrow Corridor
    print("\n4. Testing with Narrow Corridor")
    router = MycelialRouter(
        grid_size=20,
        nutrient_density=0.3,
        obstacle_density=0.0
    )
    # Create narrow corridor
    for i in range(20):
        for j in range(20):
            if i != 10 and not (i == start[0] and j == start[1]) and not (i == goal[0] and j == goal[1]):
                router.grid[i, j].is_obstacle = True
    start = (10, 0)
    goal = (10, 19)
    test_pathfinding(router, start, goal, "Narrow Corridor")

    # Test Case 5: Disconnected Regions
    print("\n5. Testing with Disconnected Regions")
    router = MycelialRouter(
        grid_size=20,
        nutrient_density=0.3,
        obstacle_density=0.0
    )
    # Create wall in the middle
    for i in range(20):
        router.grid[i, 10].is_obstacle = True
    start = (0, 0)
    goal = (19, 19)
    test_pathfinding(router, start, goal, "Disconnected Regions")

def test_pathfinding(router, start, goal, case_name):
    # Initialize A* pathfinder
    astar = AStarPathfinder(router.grid, router.grid_size)
    
    # Test A*
    astar_start_time = time.time()
    try:
        astar_path, astar_cost = astar.find_path(start, goal)
        astar_success = len(astar_path) > 0
    except Exception as e:
        print(f"A* failed with error: {str(e)}")
        astar_success = False
    astar_time = time.time() - astar_start_time

    # Test Dijkstra
    dijkstra_start_time = time.time()
    try:
        dijkstra_path = nx.shortest_path(router.graph, start, goal, weight='weight')
        dijkstra_success = len(dijkstra_path) > 0
    except (nx.NetworkXNoPath, Exception) as e:
        print(f"Dijkstra failed with error: {str(e)}")
        dijkstra_success = False
    dijkstra_time = time.time() - dijkstra_start_time

    # Test RL
    print(f"\nTesting {case_name}:")
    print(f"A* {'succeeded' if astar_success else 'failed'} in {astar_time:.4f}s")
    print(f"Dijkstra {'succeeded' if dijkstra_success else 'failed'} in {dijkstra_time:.4f}s")
    
    # Train RL for a few episodes
    router.start_growth(start)
    rl_success = False
    rl_start_time = time.time()
    
    for episode in range(5):  # Limited episodes for testing
        router.mycelium = nx.Graph()
        router.mycelium.add_node(start)
        router.grid[start].visited = True
        router.active_tips = [start]
        
        for step in range(50):  # Limited steps per episode
            router.grow_step()
            if goal in router.mycelium.nodes():
                try:
                    path = nx.shortest_path(router.mycelium, start, goal)
                    rl_success = True
                    break
                except nx.NetworkXNoPath:
                    continue
        if rl_success:
            break
    
    rl_time = time.time() - rl_start_time
    print(f"RL {'succeeded' if rl_success else 'failed'} in {rl_time:.4f}s")

if __name__ == "__main__":
    test_edge_cases() 