from mycelial_router.core import MycelialRouter
import time

def test_visualization():
    print("Starting test visualization...")
    
    # Create a small router instance
    print("Creating router instance...")
    router = MycelialRouter(grid_size=(5, 5))
    
    # Start growth from the center
    print("Starting growth from center...")
    start_node = (2, 2)
    router.start_growth(start_node)
    
    # Grow for a few steps
    print("Growing mycelium...")
    start_time = time.time()
    for i in range(5):
        print(f"Growth step {i+1}/5")
        router.grow_step()
    growth_time = time.time() - start_time
    
    print(f"Growth completed in {growth_time:.3f} seconds")
    
    # Visualize
    print("Starting visualization...")
    start_time = time.time()
    router.visualize(show_grid=True, show_mycelium=True)
    viz_time = time.time() - start_time
    print(f"Visualization completed in {viz_time:.3f} seconds")

if __name__ == "__main__":
    print("Starting main execution...")
    test_visualization()
    print("Test completed.") 