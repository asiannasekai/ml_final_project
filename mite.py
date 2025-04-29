import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import time

class MITE:
    def __init__(
        self,
        n_seeds: int = 5,
        k_neighbors: int = 10,
        branch_prob: float = 0.1,
        max_steps: int = 1000,
        fusion_threshold: float = 0.1,
        verbose: bool = True,
        check_interval: int = 100
    ):
        """
        Initialize the Mycelium-Inspired Topology Explorer.
        
        Args:
            n_seeds: Number of initial seed points
            k_neighbors: Number of nearest neighbors to consider
            branch_prob: Probability of branching at each step
            max_steps: Maximum number of growth steps
            fusion_threshold: Distance threshold for fusion events
            verbose: Whether to print progress information
            check_interval: Number of steps between progress checks
        """
        self.n_seeds = n_seeds
        self.k_neighbors = k_neighbors
        self.branch_prob = branch_prob
        self.max_steps = max_steps
        self.fusion_threshold = fusion_threshold
        self.verbose = verbose
        self.check_interval = check_interval
        self.graph = nx.Graph()
        self.fusion_events = []
        self.node_counter = 0
        self.step_count = 0
        self.start_time = None
        
    def _print_progress(self, active_tips: List[int]) -> None:
        """Print progress information."""
        if not self.verbose:
            return
            
        elapsed = time.time() - self.start_time
        print(f"\rStep {self.step_count}/{self.max_steps} | "
              f"Active tips: {len(active_tips)} | "
              f"Nodes: {self.graph.number_of_nodes()} | "
              f"Edges: {self.graph.number_of_edges()} | "
              f"Fusions: {len(self.fusion_events)} | "
              f"Time: {elapsed:.1f}s", end="")
        
    def _visualize_progress(self, points: np.ndarray) -> None:
        """Visualize current state of the graph."""
        if not self.verbose:
            return
            
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.3)
        
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, node_size=20, node_color='red', 
                edge_color='blue', width=1.5)
        
        for src, dst, _ in self.fusion_events:
            plt.plot([pos[src][0], pos[dst][0]], 
                    [pos[src][1], pos[dst][1]], 
                    'g-', linewidth=2)
        
        plt.title(f"MITE Progress (Step {self.step_count})")
        plt.show()
        
    def _compute_local_density(self, points: np.ndarray) -> np.ndarray:
        """Compute local density for each point using k-nearest neighbors."""
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(points)
        distances, _ = nbrs.kneighbors(points)
        return 1 / np.mean(distances, axis=1)
    
    def _select_seeds(self, points: np.ndarray) -> np.ndarray:
        """Select initial seed points based on local density."""
        densities = self._compute_local_density(points)
        seed_indices = np.random.choice(
            len(points),
            size=self.n_seeds,
            p=densities/np.sum(densities),
            replace=False
        )
        return points[seed_indices]
    
    def _grow_step(self, points: np.ndarray, active_tips: List[int]) -> List[int]:
        """Perform one growth step for all active tips."""
        new_tips = []
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(points)
        
        for tip in active_tips:
            # Get the current tip's position
            tip_pos = self.graph.nodes[tip]['pos']
            
            # Find neighbors
            distances, indices = nbrs.kneighbors([tip_pos])
            neighbors = points[indices[0]]
            
            # Compute local densities
            neighbor_densities = self._compute_local_density(neighbors)
            
            # Select next point to grow towards
            probs = neighbor_densities / np.sum(neighbor_densities)
            next_idx = np.random.choice(len(neighbors), p=probs)
            next_point = neighbors[next_idx]
            
            # Add edge to graph with new node
            next_node = self.node_counter
            self.node_counter += 1
            self.graph.add_node(next_node, pos=next_point)
            self.graph.add_edge(tip, next_node)
            
            # Check for fusion
            for existing_node in self.graph.nodes():
                if existing_node != tip and existing_node != next_node:
                    existing_pos = self.graph.nodes[existing_node]['pos']
                    dist = np.linalg.norm(existing_pos - next_point)
                    if dist < self.fusion_threshold:
                        self.graph.add_edge(next_node, existing_node)
                        self.fusion_events.append((next_node, existing_node, dist))
            
            new_tips.append(next_node)
            
            # Branching
            if np.random.random() < self.branch_prob:
                branch_idx = np.random.choice(len(neighbors), p=probs)
                branch_point = neighbors[branch_idx]
                branch_node = self.node_counter
                self.node_counter += 1
                self.graph.add_node(branch_node, pos=branch_point)
                self.graph.add_edge(tip, branch_node)
                new_tips.append(branch_node)
        
        return new_tips
    
    def fit(self, points: np.ndarray) -> None:
        """
        Fit the MITE model to the given point cloud.
        
        Args:
            points: Input point cloud as numpy array of shape (n_points, n_dimensions)
        """
        self.start_time = time.time()
        
        # Initialize with seed points
        seeds = self._select_seeds(points)
        for i, seed in enumerate(seeds):
            self.graph.add_node(i, pos=seed)
            self.node_counter = max(self.node_counter, i + 1)
        
        active_tips = list(range(len(seeds)))
        
        if self.verbose:
            print("\nInitial state:")
            self._print_progress(active_tips)
            self._visualize_progress(points)
        
        # Growth process
        for self.step_count in range(1, self.max_steps + 1):
            if not active_tips:
                break
                
            active_tips = self._grow_step(points, active_tips)
            
            # Check progress at intervals
            if self.verbose and (self.step_count % self.check_interval == 0):
                self._print_progress(active_tips)
                self._visualize_progress(points)
        
        if self.verbose:
            print("\n\nFinal state:")
            self._print_progress(active_tips)
            print("\n")
    
    def get_betti_numbers(self) -> Tuple[int, int]:
        """
        Compute Betti numbers from the grown graph.
        
        Returns:
            Tuple of (b0, b1) where:
            - b0: Number of connected components
            - b1: Number of loops
        """
        b0 = nx.number_connected_components(self.graph)
        b1 = len(self.fusion_events)  # Simplified version
        return b0, b1
    
    def visualize(self, points: np.ndarray) -> None:
        """Visualize the grown graph and point cloud."""
        plt.figure(figsize=(10, 10))
        
        # Plot point cloud
        plt.scatter(points[:, 0], points[:, 1], c='gray', alpha=0.3)
        
        # Plot graph
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, node_size=20, node_color='red', 
                edge_color='blue', width=1.5)
        
        # Highlight fusion events
        for src, dst, _ in self.fusion_events:
            plt.plot([pos[src][0], pos[dst][0]], 
                    [pos[src][1], pos[dst][1]], 
                    'g-', linewidth=2)
        
        plt.title("MITE Topology Exploration")
        plt.show() 