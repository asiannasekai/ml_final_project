import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import imageio
from enhanced_mycelium import EnhancedMycelium
from tqdm import tqdm

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def test_enhanced_growth():
    """Test the enhanced mycelium growth with thickness and resource allocation."""
    # Create directories for saving visualizations
    ensure_dir('visualizations')
    ensure_dir('visualizations/frames')
    
    # Initialize enhanced mycelium
    mycelium = EnhancedMycelium(grid_size=20, nutrient_density=0.3, obstacle_density=0.2)
    
    # Start growth from center
    center = (10, 10)
    mycelium.start_growth(center)
    
    # Grow for 50 steps and save frames
    frames = []
    total_steps = 50
    save_interval = 5  # Save every 5 steps to reduce number of frames
    
    print("Generating growth visualization...")
    for step in tqdm(range(total_steps)):
        mycelium.grow_step()
        
        # Only save frames at intervals
        if step % save_interval == 0:
            # Create and save frame
            plt.figure(figsize=(15, 6))
            
            # Plot grid
            plt.subplot(121)
            grid_data = np.zeros(mycelium.grid_size)
            for i in range(mycelium.grid_size[0]):
                for j in range(mycelium.grid_size[1]):
                    state = mycelium.grid[i, j]
                    if state.is_obstacle:
                        grid_data[i, j] = -1
                    else:
                        grid_data[i, j] = state.nutrient_value
            
            plt.imshow(grid_data, cmap='YlOrRd')
            plt.title(f'Nutrient Distribution (Step {step})')
            
            # Plot mycelium
            plt.subplot(122)
            if mycelium.mycelium.number_of_nodes() > 0:
                pos = {node: (node[1], -node[0]) for node in mycelium.mycelium.nodes()}
                
                # Draw nodes with resource levels
                node_colors = [mycelium.grid[node].resource_level for node in mycelium.mycelium.nodes()]
                nx.draw_networkx_nodes(mycelium.mycelium, pos, node_color=node_colors,
                                     cmap='YlGn', node_size=200, alpha=0.8)
                
                # Draw edges with thickness
                edge_widths = [mycelium.mycelium.edges[edge].get('thickness', mycelium.min_thickness) * 3
                              for edge in mycelium.mycelium.edges()]
                nx.draw_networkx_edges(mycelium.mycelium, pos, width=edge_widths,
                                     edge_color='darkgreen', alpha=0.8)
                
                plt.title(f'Enhanced Mycelium Network (Step {step})')
            
            # Save frame
            frame_path = f'visualizations/frames/step_{step:03d}.png'
            plt.savefig(frame_path, dpi=100)  # Reduced DPI for faster processing
            plt.close()
            frames.append(frame_path)
    
    print("Creating GIF...")
    # Create GIF with progress bar
    images = []
    for frame in tqdm(frames):
        images.append(imageio.v2.imread(frame))
    imageio.mimsave('visualizations/growth_progression.gif', images, duration=0.3)
    
    # Clean up individual frames
    print("Cleaning up temporary files...")
    for frame in frames:
        os.remove(frame)
    
    # Print statistics
    print("\nEnhanced Mycelium Statistics:")
    print(f"Number of nodes: {mycelium.mycelium.number_of_nodes()}")
    print(f"Number of edges: {mycelium.mycelium.number_of_edges()}")
    print(f"Number of fusion events: {len(mycelium.fusion_events)}")
    
    # Calculate average thickness
    thicknesses = [mycelium.mycelium.edges[edge].get('thickness', mycelium.min_thickness)
                  for edge in mycelium.mycelium.edges()]
    print(f"Average hyphal thickness: {np.mean(thicknesses):.3f}")
    
    # Calculate resource distribution
    resource_levels = [mycelium.grid[node].resource_level for node in mycelium.mycelium.nodes()]
    print(f"Average resource level: {np.mean(resource_levels):.3f}")
    print(f"Resource level standard deviation: {np.std(resource_levels):.3f}")

def test_resource_allocation():
    """Test resource allocation and pruning."""
    mycelium = EnhancedMycelium(grid_size=15, nutrient_density=0.4, obstacle_density=0.1)
    
    # Start from multiple points
    start_points = [(7, 7), (7, 8), (8, 7), (8, 8)]
    mycelium.start_growth(start_points)
    
    # Grow and visualize at different stages
    stages = [10, 20, 30, 40, 50]
    for stage in stages:
        for _ in range(10):
            mycelium.grow_step()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        grid_data = np.zeros(mycelium.grid_size)
        for i in range(mycelium.grid_size[0]):
            for j in range(mycelium.grid_size[1]):
                state = mycelium.grid[i, j]
                if state.is_obstacle:
                    grid_data[i, j] = -1
                else:
                    grid_data[i, j] = state.resource_level
        
        plt.imshow(grid_data, cmap='YlGn')
        plt.title(f'Resource Distribution (Step {stage*10})')
        
        plt.subplot(122)
        pos = {node: (node[1], -node[0]) for node in mycelium.mycelium.nodes()}
        edge_widths = [mycelium.mycelium.edges[edge].get('thickness', mycelium.min_thickness) * 3
                      for edge in mycelium.mycelium.edges()]
        nx.draw_networkx_edges(mycelium.mycelium, pos, width=edge_widths,
                             edge_color='darkgreen', alpha=0.8)
        plt.title(f'Hyphal Thickness (Step {stage*10})')
        
        plt.show()

def test_pruning():
    """Test the pruning mechanism."""
    mycelium = EnhancedMycelium(grid_size=20, nutrient_density=0.2, obstacle_density=0.3)
    
    # Start from center
    mycelium.start_growth((10, 10))
    
    # Grow and track network size
    steps = 100
    node_counts = []
    edge_counts = []
    
    for step in range(steps):
        mycelium.grow_step()
        if step % 5 == 0:  # Track every 5 steps
            node_counts.append(mycelium.mycelium.number_of_nodes())
            edge_counts.append(mycelium.mycelium.number_of_edges())
    
    # Plot growth and pruning
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, steps, 5), node_counts, label='Nodes')
    plt.plot(range(0, steps, 5), edge_counts, label='Edges')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.title('Network Size Over Time')
    plt.legend()
    plt.show()

def test_goal_directed_growth():
    """Test the goal-directed growth capabilities of the enhanced mycelium."""
    print("\nTesting goal-directed growth...")
    
    # Initialize enhanced mycelium with higher goal direction weight
    mycelium = EnhancedMycelium(
        grid_size=20,
        nutrient_density=0.3,
        obstacle_density=0.2,
        goal_direction_weight=0.7,  # High weight for goal-directed behavior
        exploration_weight=0.2,     # Lower weight for exploration
        nutrient_weight=0.1         # Lower weight for nutrient following
    )
    
    # Set start and goal positions
    start = (2, 2)
    goal = (18, 18)
    
    # Start growth from start position with goal
    mycelium.start_growth(start, goal)
    
    # Grow until goal is reached or max steps
    max_steps = 100
    frames = []
    ensure_dir('visualizations/goal_directed')
    
    print("Growing towards goal...")
    for step in tqdm(range(max_steps)):
        mycelium.grow_step()
        
        # Save visualization every 5 steps
        if step % 5 == 0:
            plt.figure(figsize=(10, 10))
            
            # Plot the network
            pos = {node: (node[1], -node[0]) for node in mycelium.mycelium.nodes()}
            
            # Draw nodes
            node_colors = [mycelium.grid[node].resource_level for node in mycelium.mycelium.nodes()]
            nx.draw_networkx_nodes(mycelium.mycelium, pos, node_color=node_colors,
                                 cmap='YlGn', node_size=100, alpha=0.8)
            
            # Draw edges
            edge_widths = [mycelium.mycelium.edges[edge].get('thickness', mycelium.min_thickness) * 2
                          for edge in mycelium.mycelium.edges()]
            nx.draw_networkx_edges(mycelium.mycelium, pos, width=edge_widths,
                                 edge_color='darkgreen', alpha=0.8)
            
            # Highlight start and goal
            plt.plot(start[1], -start[0], 'go', markersize=15, label='Start')
            plt.plot(goal[1], -goal[0], 'ro', markersize=15, label='Goal')
            
            plt.title(f'Goal-Directed Growth (Step {step})')
            plt.legend()
            
            # Save frame
            frame_path = f'visualizations/goal_directed/step_{step:03d}.png'
            plt.savefig(frame_path)
            plt.close()
            frames.append(frame_path)
            
            # Check if goal is reached
            if mycelium.goal_reached:
                print(f"\nGoal reached at step {step}!")
                break
    
    # Create GIF
    print("Creating goal-directed growth GIF...")
    images = []
    for frame in tqdm(frames):
        images.append(imageio.v2.imread(frame))
    imageio.mimsave('visualizations/goal_directed_growth.gif', images, duration=0.3)
    
    # Clean up frames
    for frame in frames:
        os.remove(frame)
    
    # Print statistics
    print("\nGoal-Directed Growth Statistics:")
    print(f"Steps taken: {step}")
    print(f"Number of nodes: {mycelium.mycelium.number_of_nodes()}")
    print(f"Number of edges: {mycelium.mycelium.number_of_edges()}")
    print(f"Number of fusion events: {len(mycelium.fusion_events)}")
    
    # Calculate path efficiency
    if mycelium.goal_reached:
        shortest_path = nx.shortest_path(mycelium.mycelium, start, goal)
        actual_path_length = len(shortest_path) - 1
        manhattan_distance = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        efficiency = manhattan_distance / actual_path_length
        print(f"Path efficiency: {efficiency:.3f}")

if __name__ == "__main__":
    print("Testing enhanced mycelium growth...")
    test_enhanced_growth()
    
    print("\nTesting resource allocation...")
    test_resource_allocation()
    
    print("\nTesting pruning mechanism...")
    test_pruning()
    
    print("\nTesting goal-directed growth...")
    test_goal_directed_growth() 