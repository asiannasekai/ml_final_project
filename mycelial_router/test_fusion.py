from mycelial_router.core import MycelialRouter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def test_hyphal_fusion(visualize_every: int = 5, show_plots: bool = True):
    """
    Test mycelial growth with hyphal fusion.
    
    Args:
        visualize_every: Number of steps between visualizations
        show_plots: Whether to show plots during growth
    """
    print("\nTesting mycelial growth with hyphal fusion...")
    
    # Create router with fusion enabled
    router = MycelialRouter(
        grid_size=(15, 15),
        nutrient_range=(0.1, 1.0),
        cost_range=(0.1, 1.0),
        obstacle_prob=0.1,
        failure_prob=0.05,
        fusion_threshold=2.0  # Allow fusion within 2 units
    )
    
    # Start growth from multiple points
    start_nodes = [(5, 5), (10, 10), (5, 10), (10, 5)]
    router.start_growth(start_nodes)
    
    # Visualize initial state
    if show_plots:
        print("Initial state:")
        router.visualize()
    
    # Track metrics over time
    metrics_history = {
        'nodes': [],
        'edges': [],
        'fusions': [],
        'active_tips': []
    }
    
    # Grow for 20 steps
    for i in range(20):
        print(f"\nStep {i+1}:")
        router.grow_step()
        
        # Store metrics
        metrics_history['nodes'].append(router.mycelium.number_of_nodes())
        metrics_history['edges'].append(router.mycelium.number_of_edges())
        metrics_history['fusions'].append(len(router.fusion_events))
        metrics_history['active_tips'].append(len(router.active_tips))
        
        # Print fusion events if any
        if router.fusion_events:
            print(f"Fusion events in step {i+1}: {len(router.fusion_events)}")
        
        # Visualize periodically
        if show_plots and (i + 1) % visualize_every == 0:
            router.visualize()
    
    # Analyze the network
    print("\nNetwork Analysis:")
    print(f"Total nodes: {router.mycelium.number_of_nodes()}")
    print(f"Total edges: {router.mycelium.number_of_edges()}")
    print(f"Fusion events: {len(router.fusion_events)}")
    
    # Calculate network metrics
    if router.mycelium.number_of_nodes() > 1:
        # Average path length
        avg_path_length = nx.average_shortest_path_length(router.mycelium)
        print(f"Average path length: {avg_path_length:.2f}")
        
        # Network diameter
        diameter = nx.diameter(router.mycelium)
        print(f"Network diameter: {diameter}")
        
        # Clustering coefficient
        clustering = nx.average_clustering(router.mycelium)
        print(f"Average clustering coefficient: {clustering:.2f}")
        
        # Network density
        density = nx.density(router.mycelium)
        print(f"Network density: {density:.2f}")
    
    # Plot metrics over time
    if show_plots:
        plt.figure(figsize=(15, 10))
        
        # Plot network size metrics
        plt.subplot(2, 2, 1)
        plt.plot(metrics_history['nodes'], 'b-', label='Nodes')
        plt.plot(metrics_history['edges'], 'g-', label='Edges')
        plt.title('Network Size Over Time')
        plt.xlabel('Step')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        
        # Plot fusion events
        plt.subplot(2, 2, 2)
        plt.plot(metrics_history['fusions'], 'r-')
        plt.title('Cumulative Fusion Events')
        plt.xlabel('Step')
        plt.ylabel('Number of Fusions')
        plt.grid(True)
        
        # Plot active tips
        plt.subplot(2, 2, 3)
        plt.plot(metrics_history['active_tips'], 'm-')
        plt.title('Active Tips Over Time')
        plt.xlabel('Step')
        plt.ylabel('Number of Active Tips')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Run with visualization every 5 steps
    test_hyphal_fusion(visualize_every=5, show_plots=True) 