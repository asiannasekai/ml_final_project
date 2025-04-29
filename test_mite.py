import numpy as np
from mite import MITE
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_swiss_roll

def generate_torus(n_points=1000, R=2, r=1):
    """Generate points on a torus."""
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    x = (R + r*np.cos(theta)) * np.cos(phi)
    y = (R + r*np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return np.column_stack((x, y, z))

def test_circle():
    """Test MITE on a circle."""
    print("\n" + "="*50)
    print("Testing on circle (expected Betti numbers: b0=1, b1=1)")
    print("="*50)
    
    points, _ = make_circles(n_samples=1000, noise=0.05, factor=0.5)
    mite = MITE(
        n_seeds=5,
        k_neighbors=15,
        branch_prob=0.1,
        fusion_threshold=0.1,
        max_steps=500,  # Reduced for testing
        verbose=True,
        check_interval=100
    )
    mite.fit(points)
    b0, b1 = mite.get_betti_numbers()
    print(f"\nFinal Betti numbers: b0={b0}, b1={b1}")
    mite.visualize(points)

def test_torus():
    """Test MITE on a torus."""
    print("\n" + "="*50)
    print("Testing on torus (expected Betti numbers: b0=1, b1=2)")
    print("="*50)
    
    points = generate_torus(n_points=2000)
    mite = MITE(
        n_seeds=10,
        k_neighbors=20,
        branch_prob=0.1,
        fusion_threshold=0.2,
        max_steps=500,  # Reduced for testing
        verbose=True,
        check_interval=100
    )
    mite.fit(points)
    b0, b1 = mite.get_betti_numbers()
    print(f"\nFinal Betti numbers: b0={b0}, b1={b1}")
    
    # 3D visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', alpha=0.3)
    plt.title("MITE on Torus")
    plt.show()

def test_swiss_roll():
    """Test MITE on Swiss roll."""
    print("\n" + "="*50)
    print("Testing on Swiss roll (expected Betti numbers: b0=1, b1=1)")
    print("="*50)
    
    points, _ = make_swiss_roll(n_samples=2000, noise=0.1)
    mite = MITE(
        n_seeds=10,
        k_neighbors=20,
        branch_prob=0.1,
        fusion_threshold=0.2,
        max_steps=500,  # Reduced for testing
        verbose=True,
        check_interval=100
    )
    mite.fit(points)
    b0, b1 = mite.get_betti_numbers()
    print(f"\nFinal Betti numbers: b0={b0}, b1={b1}")
    
    # 3D visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', alpha=0.3)
    plt.title("MITE on Swiss Roll")
    plt.show()

if __name__ == "__main__":
    # Test one shape at a time for better visualization
    test_circle()
    # Uncomment to test other shapes
    # test_torus()
    # test_swiss_roll() 