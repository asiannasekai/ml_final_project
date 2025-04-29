from mycelial_router.core import MycelialRouter
import matplotlib.pyplot as plt
import numpy as np

def test_basic_growth():
    """Test basic mycelial growth on a small grid."""
    print("\nTesting basic mycelial growth...")
    
    # Create router with small grid
    router = MycelialRouter(
        grid_size=(10, 10),
        nutrient_range=(0.1, 1.0),
        cost_range=(0.1, 1.0),
        obstacle_prob=0.1,
        failure_prob=0.05
    )
    
    # Start growth from center
    start_node = (5, 5)
    router.start_growth(start_node)
    
    # Visualize initial state
    print("Initial state:")
    router.visualize()
    
    # Grow for 10 steps
    for i in range(10):
        print(f"\nStep {i+1}:")
        router.grow_step()
        router.visualize()
    
    # Compare with Dijkstra
    end_node = (0, 0)  # Top-left corner
    comparison = router.compare_with_dijkstra(start_node, end_node)
    
    print("\nPath Comparison:")
    print(f"Mycelial path cost: {comparison['mycelial_cost']:.2f}")
    print(f"Dijkstra path cost: {comparison['dijkstra_cost']:.2f}")
    print(f"Mycelial redundancy: {comparison['mycelial_redundancy']}")
    print(f"Dijkstra redundancy: {comparison['dijkstra_redundancy']}")

def test_failure_tolerance():
    """Test mycelial growth with node failures."""
    print("\nTesting failure tolerance...")
    
    # Create router with higher failure probability
    router = MycelialRouter(
        grid_size=(15, 15),
        nutrient_range=(0.1, 1.0),
        cost_range=(0.1, 1.0),
        obstacle_prob=0.1,
        failure_prob=0.2  # Higher failure probability
    )
    
    # Start growth from center
    start_node = (7, 7)
    router.start_growth(start_node)
    
    # Visualize initial state
    print("Initial state:")
    router.visualize()
    
    # Grow for 15 steps
    for i in range(15):
        print(f"\nStep {i+1}:")
        router.grow_step()
        router.visualize()
    
    # Compare with Dijkstra
    end_node = (0, 0)
    comparison = router.compare_with_dijkstra(start_node, end_node)
    
    print("\nPath Comparison (with failures):")
    print(f"Mycelial path cost: {comparison['mycelial_cost']:.2f}")
    print(f"Dijkstra path cost: {comparison['dijkstra_cost']:.2f}")
    print(f"Mycelial redundancy: {comparison['mycelial_redundancy']}")
    print(f"Dijkstra redundancy: {comparison['dijkstra_redundancy']}")

if __name__ == "__main__":
    # Run tests
    test_basic_growth()
    test_failure_tolerance() 