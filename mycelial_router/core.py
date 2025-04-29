import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mycelial_router.rl_agent import MyceliumRLAgent
import random

@dataclass
class NodeState:
    """State of a node in the network."""
    position: Tuple[int, int]  # Position in the grid
    is_obstacle: bool     # Whether the node is an obstacle
    is_failed: bool      # Whether the node has failed
    nutrient_value: float  # Amount of nutrients at this node
    traversal_cost: float  # Cost to traverse this node
    visited: bool = False  # Whether the node has been visited by the mycelium

class MycelialRouter:
    def __init__(self, grid_size=20, nutrient_density=0.3, obstacle_density=0.2):
        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size)  # Convert to tuple if int
        else:
            self.grid_size = grid_size
        self.nutrient_density = nutrient_density
        self.obstacle_density = obstacle_density
        self.graph = nx.Graph()  # Initialize graph first
        self.grid = self._initialize_grid()
        self.rl_agent = None  # Initialize as None, will be set when needed
        
        # Track mycelial growth
        self.mycelium = nx.Graph()
        self.active_tips = []
        self.fusion_events = []  # Track fusion events
    
    def _initialize_grid(self) -> np.ndarray:
        """Initialize the grid with random nutrient values and obstacles."""
        rows, cols = self.grid_size
        grid = np.empty((rows, cols), dtype=object)
        
        for i in range(rows):
            for j in range(cols):
                # Randomly determine if this is an obstacle
                is_obstacle = np.random.random() < self.obstacle_density
                
                # Randomly determine if this node has failed
                is_failed = np.random.random() < 0.05
                
                # Generate random nutrient value and traversal cost
                nutrient = np.random.uniform(0.1, 1.0)
                cost = np.random.uniform(0.1, 1.0)
                
                # Create node state
                grid[i, j] = NodeState(
                    position=(i, j),
                    is_obstacle=is_obstacle,
                    is_failed=is_failed,
                    nutrient_value=nutrient,
                    traversal_cost=cost,
                    visited=False  # Explicitly set visited to False
                )
                
                # Add node to graph if not an obstacle
                if not is_obstacle:
                    self.graph.add_node((i, j), weight=cost)
                    
                    # Add edges to adjacent nodes (4-connectivity)
                    if i > 0 and not grid[i-1, j].is_obstacle:
                        self.graph.add_edge((i, j), (i-1, j), weight=(cost + grid[i-1, j].traversal_cost)/2)
                    if j > 0 and not grid[i, j-1].is_obstacle:
                        self.graph.add_edge((i, j), (i, j-1), weight=(cost + grid[i, j-1].traversal_cost)/2)
        
        return grid
    
    def _create_graph(self) -> nx.Graph:
        """Create a network graph from the grid."""
        G = nx.Graph()
        rows, cols = self.grid_size
        
        # Add nodes
        for i in range(rows):
            for j in range(cols):
                node_id = (i, j)
                G.add_node(node_id, state=self.grid[i, j])
        
        # Add edges (8-connected grid)
        for i in range(rows):
            for j in range(cols):
                current = (i, j)
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbor = (ni, nj)
                            # Add edge with weight based on traversal costs
                            weight = (self.grid[i, j].traversal_cost + 
                                    self.grid[ni, nj].traversal_cost) / 2
                            G.add_edge(current, neighbor, weight=weight)
        
        return G
    
    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors for a node."""
        i, j = node
        rows, cols = self.grid_size
        neighbors = []
        
        # Include diagonal movements
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor = (ni, nj)
                    state = self.grid[ni, nj]
                    if not state.is_obstacle and not state.is_failed:
                        neighbors.append(neighbor)
        
        return neighbors
    
    def _compute_reward(self, node: Tuple[int, int]) -> float:
        """Compute reward for moving to a node."""
        state = self.grid[node]
        
        # Large penalty for obstacles and failed nodes
        if state.is_obstacle:
            return -1.0  # Reduced from -2.0
        if state.is_failed:
            return -0.8  # Reduced from -1.5
            
        # Base reward from nutrients and cost
        nutrient_reward = state.nutrient_value
        cost_penalty = state.traversal_cost * 0.3  # Reduced from 0.5
        
        # Bonus for exploring new areas
        exploration_bonus = 0.3 if not state.visited else 0.0  # Increased from 0.2
        
        # Combine rewards
        total_reward = nutrient_reward - cost_penalty + exploration_bonus
        
        return total_reward
    
    def _check_fusion(self, new_node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Check for potential fusion events with the new node."""
        fusion_events = []
        
        # Calculate distance to all other nodes
        for existing_node in self.mycelium.nodes():
            if existing_node != new_node:
                # Calculate Euclidean distance
                distance = np.sqrt((new_node[0] - existing_node[0])**2 + 
                                 (new_node[1] - existing_node[1])**2)
                
                # If within fusion threshold and not already connected
                if (distance <= 2.0 and 
                    not self.mycelium.has_edge(new_node, existing_node)):
                    fusion_events.append((new_node, existing_node))
        
        return fusion_events

    def grow_step(self) -> None:
        """Perform one step of mycelial growth with hyphal fusion."""
        if not self.active_tips:
            print("No active tips to grow from!")
            return
            
        print(f"Growing from {len(self.active_tips)} active tips")
        new_active_tips = []
        fusion_events = 0
        
        # Track visited positions to prevent oscillation
        visited_positions = set()
        
        for tip in self.active_tips:
            # Get valid neighbors for this tip
            neighbors = self._get_neighbors(tip)
            if not neighbors:
                print(f"No valid neighbors for tip {tip}")
                continue
                
            print(f"Tip {tip} has {len(neighbors)} valid neighbors")
            
            # Sort neighbors by nutrient value (descending)
            neighbors.sort(key=lambda n: self.grid[n].nutrient_value, reverse=True)
            
            # Always grow to the best nutrient neighbor
            next_node = neighbors[0]
            
            # Check if we've been here before
            if next_node in visited_positions:
                # If we've been here, try the next best neighbor
                for node in neighbors[1:]:
                    if node not in visited_positions:
                        next_node = node
                        break
                else:
                    # If all neighbors have been visited, choose randomly
                    next_node = random.choice(neighbors)
            
            visited_positions.add(next_node)
            
            # Add the new edge
            self.mycelium.add_edge(tip, next_node)
            print(f"Added new tip at {next_node}")
            new_active_tips.append(next_node)
            
            # Check for fusion with existing network
            for existing_node in self.mycelium.nodes():
                if existing_node != tip and existing_node != next_node:
                    if self._should_fuse(next_node, existing_node):
                        self.mycelium.add_edge(next_node, existing_node)
                        print(f"Fusion event: {next_node} fused with {existing_node}")
                        fusion_events += 1
            
            # Increase branching probability to maintain more active tips
            if random.random() < 0.4:  # Increased from 0.3
                # Choose a different neighbor for branching
                branch_candidates = [n for n in neighbors if n != next_node]
                if branch_candidates:
                    branch_node = random.choice(branch_candidates)
                    if branch_node not in visited_positions:
                        self.mycelium.add_edge(tip, branch_node)
                        print(f"Added branch tip at {branch_node}")
                        new_active_tips.append(branch_node)
                        visited_positions.add(branch_node)
                        
                        # Check for fusion with the branch
                        for existing_node in self.mycelium.nodes():
                            if existing_node != tip and existing_node != branch_node:
                                if self._should_fuse(branch_node, existing_node):
                                    self.mycelium.add_edge(branch_node, existing_node)
                                    print(f"Fusion event: {branch_node} fused with {existing_node}")
                                    fusion_events += 1
        
        # Update active tips, ensuring we maintain at least 2 active tips
        self.active_tips = new_active_tips
        total_grid_size = self.grid_size[0] * self.grid_size[1]  # Calculate total grid size
        if len(self.active_tips) < 2 and len(visited_positions) < total_grid_size:
            # If we have too few active tips, add new ones from unvisited positions
            unvisited = [(i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
                        if (i, j) not in visited_positions]
            if unvisited:
                new_tips = random.sample(unvisited, min(2 - len(self.active_tips), len(unvisited)))
                for tip in new_tips:
                    # Connect to nearest existing node
                    nearest = min(self.mycelium.nodes(),
                                key=lambda n: abs(n[0] - tip[0]) + abs(n[1] - tip[1]))
                    self.mycelium.add_edge(nearest, tip)
                    self.active_tips.append(tip)
                    print(f"Added new tip at {tip} to maintain minimum active tips")
        
        print(f"New active tips: {len(self.active_tips)}")
        return fusion_events
    
    def start_growth(self, start_nodes):
        """Start mycelial growth from multiple nodes."""
        # Initialize mycelium if it doesn't exist
        if not hasattr(self, 'mycelium'):
            self.mycelium = nx.Graph()
            self.active_tips = []
            self.fusion_events = []
        
        # Convert single node to list if necessary
        if not isinstance(start_nodes, list):
            start_nodes = [start_nodes]
        
        # Add each start node
        for start_node in start_nodes:
            i, j = start_node
            self.mycelium.add_node(start_node)
            self.grid[i, j].visited = True
            self.active_tips.append(start_node)
    
    def visualize(self, show_grid: bool = True, show_mycelium: bool = True) -> None:
        """Visualize the current state of the grid and mycelium."""
        plt.figure(figsize=(15, 6))
        
        if show_grid:
            # Create nutrient and cost maps more efficiently
            nutrient_map = np.array([[self.grid[i, j].nutrient_value 
                                   for j in range(self.grid_size[1])] 
                                  for i in range(self.grid_size[0])])
            obstacle_map = np.array([[1 if self.grid[i, j].is_obstacle else 0 
                                   for j in range(self.grid_size[1])] 
                                  for i in range(self.grid_size[0])])
            failed_map = np.array([[1 if self.grid[i, j].is_failed else 0 
                                 for j in range(self.grid_size[1])] 
                                for i in range(self.grid_size[0])])
            
            # Plot grid
            plt.subplot(121)
            im = plt.imshow(nutrient_map, cmap='YlOrRd', interpolation='nearest')
            plt.colorbar(im, label='Nutrient Value', fraction=0.046, pad=0.04)
            
            # Overlay obstacles and failed nodes
            plt.imshow(obstacle_map, cmap='binary', alpha=0.3, interpolation='nearest')
            plt.imshow(failed_map, cmap='Reds', alpha=0.3, interpolation='nearest')
            
            # Add grid lines
            plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Customize ticks
            plt.xticks(np.arange(-0.5, self.grid_size[1], 1), [])
            plt.yticks(np.arange(-0.5, self.grid_size[0], 1), [])
            
            plt.title('Grid State\n(Yellow=Low Nutrients, Red=High Nutrients)\nGray=Obstacles, Red=Failed Nodes')
        
        if show_mycelium and self.mycelium.number_of_nodes() > 0:
            # Plot mycelium
            plt.subplot(122)
            
            # Create a grid layout for the nodes
            pos = {node: (node[1], -node[0]) for node in self.mycelium.nodes()}
            
            # Draw the network with different colors for different node types
            node_colors = ['red' if self.active_tips and node == self.active_tips[0] else 'green' 
                         for node in self.mycelium.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.mycelium, pos,
                node_color=node_colors,
                node_size=200,
                alpha=0.8
            )
            
            # Draw regular edges
            regular_edges = [edge for edge in self.mycelium.edges() 
                           if edge not in self.fusion_events and 
                           (edge[1], edge[0]) not in self.fusion_events]
            if regular_edges:
                nx.draw_networkx_edges(
                    self.mycelium, pos,
                    edgelist=regular_edges,
                    edge_color='darkgreen',
                    width=2,
                    alpha=0.8,
                    arrows=True,
                    arrowsize=10
                )
            
            # Draw fusion edges
            if self.fusion_events:
                nx.draw_networkx_edges(
                    self.mycelium, pos,
                    edgelist=self.fusion_events,
                    edge_color='purple',
                    width=3,
                    alpha=0.8,
                    style='dashed'
                )
            
            # Add node labels with coordinates
            labels = {node: f'({node[0]},{node[1]})' for node in self.mycelium.nodes()}
            nx.draw_networkx_labels(
                self.mycelium, pos,
                labels=labels,
                font_size=8,
                font_color='black'
            )
            
            # Add grid
            plt.grid(True, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Customize ticks
            plt.xticks(np.arange(-0.5, self.grid_size[1], 1), [])
            plt.yticks(np.arange(-0.5, self.grid_size[0], 1), [])
            
            plt.title('Mycelial Network\n(Green=Regular Nodes, Red=Active Tip)\nPurple Dashed=Fusion Edges')
        
        plt.tight_layout()
        plt.draw()  # Use draw() instead of show() for faster updates
        plt.pause(0.1)  # Small pause to allow the plot to update
    
    def compare_with_dijkstra(self, start: Tuple[int, int], 
                            end: Tuple[int, int]) -> Dict:
        """Compare mycelial path with Dijkstra's algorithm."""
        # Get mycelial path
        try:
            mycelial_path = nx.shortest_path(self.mycelium, start, end)
            mycelial_cost = sum(self.grid[node].traversal_cost 
                              for node in mycelial_path)
        except nx.NetworkXNoPath:
            mycelial_path = []
            mycelial_cost = float('inf')
        
        # Get Dijkstra's path
        try:
            dijkstra_path = nx.shortest_path(self.graph, start, end, 
                                           weight='weight')
            dijkstra_cost = nx.shortest_path_length(self.graph, start, end, 
                                                  weight='weight')
        except nx.NetworkXNoPath:
            dijkstra_path = []
            dijkstra_cost = float('inf')
        
        # Compute path redundancy
        mycelial_redundancy = self._compute_redundancy(mycelial_path)
        dijkstra_redundancy = self._compute_redundancy(dijkstra_path)
        
        return {
            'mycelial_path': mycelial_path,
            'mycelial_cost': mycelial_cost,
            'mycelial_redundancy': mycelial_redundancy,
            'dijkstra_path': dijkstra_path,
            'dijkstra_cost': dijkstra_cost,
            'dijkstra_redundancy': dijkstra_redundancy
        }
    
    def _compute_redundancy(self, path: List[Tuple[int, int]]) -> int:
        """Compute the number of alternative paths."""
        if not path:
            return 0
            
        start, end = path[0], path[-1]
        
        # Remove the path edges temporarily
        edges_to_remove = list(zip(path[:-1], path[1:]))
        self.graph.remove_edges_from(edges_to_remove)
        
        # Count alternative paths
        try:
            redundancy = len(list(nx.all_shortest_paths(
                self.graph, start, end, weight='weight')))
        except nx.NetworkXNoPath:
            redundancy = 0
        
        # Restore the edges
        self.graph.add_edges_from(edges_to_remove)
        
        return redundancy 

    def _should_fuse(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> bool:
        """
        Determine if two nodes should fuse based on multiple benefit factors.
        
        Args:
            node1: First node coordinates
            node2: Second node coordinates
            
        Returns:
            bool: True if fusion would be beneficial, False otherwise
        """
        # Don't fuse if nodes are already connected
        if self.mycelium.has_edge(node1, node2):
            return False
            
        # Calculate Manhattan distance between nodes
        distance = abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
        
        # Maximum allowed fusion distance
        max_fusion_distance = 3
        
        # If nodes are too far apart, don't fuse
        if distance > max_fusion_distance:
            return False
            
        # Calculate nutrient benefit
        node1_nutrients = self.grid[node1].nutrient_value
        node2_nutrients = self.grid[node2].nutrient_value
        avg_nutrients = (node1_nutrients + node2_nutrients) / 2
        
        # Calculate path efficiency benefit
        # Check if fusion would create a more efficient path to goal
        path_efficiency = 0.0
        if hasattr(self, 'goal') and self.goal is not None:
            try:
                # Get current path lengths
                path1_length = nx.shortest_path_length(self.mycelium, node1, self.goal)
                path2_length = nx.shortest_path_length(self.mycelium, node2, self.goal)
                
                # Calculate potential new path length after fusion
                new_path_length = min(path1_length, path2_length) + distance
                
                # Calculate efficiency improvement
                old_path_length = max(path1_length, path2_length)
                path_efficiency = (old_path_length - new_path_length) / old_path_length
            except nx.NetworkXNoPath:
                path_efficiency = 0.0
        
        # Calculate network benefit
        # Consider the current state of the network
        network_density = len(self.mycelium.edges()) / (self.grid_size[0] * self.grid_size[1])
        network_benefit = 1.0 - network_density  # Higher benefit when network is sparse
        
        # Calculate total benefit score
        distance_factor = 1.0 - (distance / max_fusion_distance)
        nutrient_factor = avg_nutrients
        path_factor = max(0, path_efficiency)  # Only consider positive improvements
        network_factor = network_benefit
        
        # Weight the factors
        benefit_score = (
            0.3 * distance_factor +    # Distance is important but not critical
            0.3 * nutrient_factor +    # Nutrient collection is important
            0.2 * path_factor +        # Path efficiency is valuable
            0.2 * network_factor       # Network state matters
        )
        
        # Only fuse if benefit score exceeds threshold
        fusion_threshold = 0.6  # High threshold to ensure significant benefit
        
        return benefit_score > fusion_threshold

    def find_path_rl(self, start, goal, max_steps=1000):
        if self.rl_agent is None:
            self.rl_agent = MyceliumRLAgent(start, goal)