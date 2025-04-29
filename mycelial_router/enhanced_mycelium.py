import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
import random

@dataclass
class EnhancedNodeState:
    """Enhanced state of a node in the network."""
    position: Tuple[int, int]  # Position in the grid
    is_obstacle: bool     # Whether the node is an obstacle
    is_failed: bool      # Whether the node has failed
    nutrient_value: float  # Amount of nutrients at this node
    traversal_cost: float  # Cost to traverse this node
    visited: bool = False  # Whether the node has been visited
    resource_level: float = 0.0  # Current resource level
    max_resources: float = 1.0  # Maximum resource capacity
    success_memory: float = 0.0  # Memory of successful paths

class EnhancedMycelium:
    def __init__(self, grid_size=20, nutrient_density=0.3, obstacle_density=0.2, 
                 goal_direction_weight=0.4, exploration_weight=0.3, nutrient_weight=0.3):
        """
        Initialize enhanced mycelium with thickness and resource allocation.
        
        Args:
            grid_size: Size of the grid (rows, columns)
            nutrient_density: Density of nutrients in the grid
            obstacle_density: Density of obstacles in the grid
            goal_direction_weight: Weight for growing towards goal (0-1)
            exploration_weight: Weight for exploring new areas (0-1)
            nutrient_weight: Weight for following nutrients (0-1)
        """
        if isinstance(grid_size, int):
            self.grid_size = (grid_size, grid_size)
        else:
            self.grid_size = grid_size
        self.nutrient_density = nutrient_density
        self.obstacle_density = obstacle_density
        self.graph = nx.Graph()
        self.grid = self._initialize_grid()
        
        # Track mycelial growth
        self.mycelium = nx.Graph()
        self.active_tips = []
        self.fusion_events = []
        
        # Resource management
        self.resource_threshold = 0.1
        self.pruning_interval = 10
        self.step_count = 0
        
        # Hyphal thickness parameters
        self.min_thickness = 0.1
        self.max_thickness = 1.0
        self.thickness_growth_rate = 0.05
        self.thickness_decay_rate = 0.02
        
        # Dynamic weights
        self.base_goal_weight = goal_direction_weight
        self.base_exploration_weight = exploration_weight
        self.base_nutrient_weight = nutrient_weight
        self.current_goal_weight = goal_direction_weight
        self.current_exploration_weight = exploration_weight
        self.current_nutrient_weight = nutrient_weight
        
        # Success tracking
        self.successful_paths = []
        self.path_success_threshold = 3
        self.weight_adjustment_rate = 0.1
        
        # Goal-directed growth parameters
        self.goal = None
        self.goal_reached = False
        
        # Chemical signaling parameters
        self.pheromone_grid = np.zeros(self.grid_size)
        self.pheromone_decay_rate = 0.05
        self.pheromone_diffusion_rate = 0.1
        self.chemical_gradient_weight = 0.3
        
        # Adaptive branching parameters
        self.base_branching_rate = 0.3
        self.stress_response_factor = 0.2
        self.min_branch_angle = 30
        self.max_branch_angle = 90
        self.branch_memory = defaultdict(lambda: {'attempts': 0, 'successes': 0})
    
    def _initialize_grid(self) -> np.ndarray:
        """Initialize the grid with enhanced node states."""
        rows, cols = self.grid_size
        grid = np.empty((rows, cols), dtype=object)
        
        for i in range(rows):
            for j in range(cols):
                is_obstacle = np.random.random() < self.obstacle_density
                is_failed = np.random.random() < 0.05
                nutrient = np.random.uniform(0.1, 1.0)
                cost = np.random.uniform(0.1, 1.0)
                
                grid[i, j] = EnhancedNodeState(
                    position=(i, j),
                    is_obstacle=is_obstacle,
                    is_failed=is_failed,
                    nutrient_value=nutrient,
                    traversal_cost=cost,
                    visited=False,
                    resource_level=0.0,
                    max_resources=1.0
                )
                
                if not is_obstacle:
                    self.graph.add_node((i, j), weight=cost)
                    if i > 0 and not grid[i-1, j].is_obstacle:
                        self.graph.add_edge((i, j), (i-1, j), 
                                          weight=(cost + grid[i-1, j].traversal_cost)/2)
                    if j > 0 and not grid[i, j-1].is_obstacle:
                        self.graph.add_edge((i, j), (i, j-1), 
                                          weight=(cost + grid[i, j-1].traversal_cost)/2)
        
        return grid
    
    def _update_hyphal_thickness(self, edge=None):
        """Update the thickness of hyphae based on resource flow.
        
        Args:
            edge: Optional; if provided, updates only this edge, otherwise updates all edges
        """
        if edge is not None:
            if edge not in self.mycelium.edges():
                return
                
            # Get current thickness or initialize
            current_thickness = self.mycelium.edges[edge].get('thickness', self.min_thickness)
            
            # Calculate resource flow through the edge
            node1, node2 = edge
            resource_diff = abs(self.grid[node1].resource_level - self.grid[node2].resource_level)
            
            # Update thickness based on resource flow
            if resource_diff > 0.1:  # Significant resource flow
                new_thickness = min(self.max_thickness, 
                                  current_thickness + self.thickness_growth_rate)
            else:
                new_thickness = max(self.min_thickness, 
                                  current_thickness - self.thickness_decay_rate)
            
            self.mycelium.edges[edge]['thickness'] = new_thickness
        else:
            # Update all edges
            for edge in self.mycelium.edges():
                self._update_hyphal_thickness(edge)
    
    def _distribute_resources(self) -> None:
        """Distribute resources through the mycelium network."""
        # First, collect resources from nutrient-rich nodes
        for node in self.mycelium.nodes():
            state = self.grid[node]
            if not state.is_obstacle and not state.is_failed:
                # Absorb nutrients from the environment
                absorption_rate = 0.1
                new_resources = min(state.max_resources - state.resource_level,
                                  state.nutrient_value * absorption_rate)
                state.resource_level += new_resources
        
        # Then, distribute resources through the network
        for _ in range(3):  # Multiple passes for better distribution
            for edge in self.mycelium.edges():
                node1, node2 = edge
                state1 = self.grid[node1]
                state2 = self.grid[node2]
                
                # Calculate flow based on thickness and resource difference
                thickness = self.mycelium.edges[edge].get('thickness', self.min_thickness)
                flow = thickness * (state1.resource_level - state2.resource_level) * 0.1
                
                # Update resource levels
                state1.resource_level -= flow
                state2.resource_level += flow
    
    def _prune_inefficient_hyphae(self) -> None:
        """Remove inefficient hyphae based on resource levels and thickness."""
        edges_to_remove = []
        
        for edge in self.mycelium.edges():
            node1, node2 = edge
            state1 = self.grid[node1]
            state2 = self.grid[node2]
            thickness = self.mycelium.edges[edge].get('thickness', self.min_thickness)
            
            # Calculate efficiency score
            resource_avg = (state1.resource_level + state2.resource_level) / 2
            efficiency = resource_avg * thickness
            
            # Mark for removal if inefficient
            if efficiency < self.resource_threshold:
                edges_to_remove.append(edge)
        
        # Remove marked edges
        for edge in edges_to_remove:
            self.mycelium.remove_edge(*edge)
            
            # If a node becomes disconnected, remove it
            for node in edge:
                if self.mycelium.degree(node) == 0:
                    self.mycelium.remove_node(node)
                    if node in self.active_tips:
                        self.active_tips.remove(node)
    
    def start_growth(self, start_nodes, goal=None):
        """Start enhanced mycelial growth from given nodes."""
        if not isinstance(start_nodes, list):
            start_nodes = [start_nodes]
        
        self.mycelium = nx.Graph()
        self.active_tips = []
        self.fusion_events = []
        self.step_count = 0
        self.goal = goal
        self.goal_reached = False
        
        for start_node in start_nodes:
            i, j = start_node
            self.mycelium.add_node(start_node)
            self.grid[i, j].visited = True
            self.grid[i, j].resource_level = 1.0
            self.active_tips.append(start_node)
    
    def _update_chemical_gradients(self):
        """Update chemical gradients in the environment."""
        if self.goal is None:
            return
            
        # Create goal-directed chemical gradient
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if not self.grid[i, j].is_obstacle:
                    distance = abs(i - self.goal[0]) + abs(j - self.goal[1])
                    self.pheromone_grid[i, j] = max(0, 1 - (distance / (sum(self.grid_size) / 2)))
        
        # Diffuse pheromones
        new_grid = self.pheromone_grid.copy()
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if not self.grid[i, j].is_obstacle:
                    neighbors = self._get_neighbors((i, j))
                    for ni, nj in neighbors:
                        new_grid[i, j] += (self.pheromone_grid[ni, nj] - self.pheromone_grid[i, j]) * self.pheromone_diffusion_rate
        
        # Apply decay
        self.pheromone_grid = new_grid * (1 - self.pheromone_decay_rate)

    def _calculate_branching_probability(self, tip, direction):
        """Calculate adaptive branching probability based on environmental factors."""
        if tip not in self.branch_memory:
            self.branch_memory[tip] = {'successes': 0, 'attempts': 0}
        
        # Get local conditions
        local_nutrients = self.grid[tip].nutrient_value
        local_pheromone = self.pheromone_grid[tip]
        stress_level = 1 - local_nutrients  # Higher stress when fewer nutrients
        
        # Calculate success rate of previous branches
        success_rate = (self.branch_memory[tip]['successes'] / 
                       max(1, self.branch_memory[tip]['attempts']))
        
        # Adjust branching rate based on conditions
        prob = self.base_branching_rate
        prob += self.stress_response_factor * stress_level  # More branches under stress
        prob += 0.2 * success_rate  # More likely to branch if successful before
        prob += 0.2 * local_pheromone  # More branches in high-pheromone areas
        
        return min(0.8, max(0.2, prob))  # Keep probability in reasonable range

    def _adjust_weights(self):
        """Dynamically adjust weights based on success and distance to goal."""
        if not self.goal or not self.active_tips:
            return

        # Calculate average distance to goal
        avg_distance = np.mean([
            abs(tip[0] - self.goal[0]) + abs(tip[1] - self.goal[1])
            for tip in self.active_tips
        ])
        
        # Normalize distance to [0, 1]
        max_possible_distance = self.grid_size[0] + self.grid_size[1]
        normalized_distance = avg_distance / max_possible_distance
        
        # Adjust weights based on distance and success
        if len(self.successful_paths) >= self.path_success_threshold:
            # If we have successful paths, focus more on goal direction
            self.current_goal_weight = min(0.8, self.base_goal_weight + self.weight_adjustment_rate)
            self.current_exploration_weight = max(0.1, self.base_exploration_weight - self.weight_adjustment_rate)
        elif normalized_distance > 0.7:
            # If far from goal, increase exploration
            self.current_exploration_weight = min(0.8, self.base_exploration_weight + self.weight_adjustment_rate)
            self.current_goal_weight = max(0.1, self.base_goal_weight - self.weight_adjustment_rate)
        else:
            # Gradually return to base weights
            self.current_goal_weight = self.base_goal_weight
            self.current_exploration_weight = self.base_exploration_weight
        
        # Ensure weights sum to 1
        total = self.current_goal_weight + self.current_exploration_weight + self.current_nutrient_weight
        self.current_goal_weight /= total
        self.current_exploration_weight /= total
        self.current_nutrient_weight /= total

    def _calculate_growth_score(self, current_node, candidate_node):
        """Calculate growth score with dynamic weights and success memory."""
        score = 0.0
        
        # Goal direction score
        if self.goal is not None:
            current_dist = abs(current_node[0] - self.goal[0]) + abs(current_node[1] - self.goal[1])
            candidate_dist = abs(candidate_node[0] - self.goal[0]) + abs(candidate_node[1] - self.goal[1])
            direction_score = (current_dist - candidate_dist) / max(current_dist, 1)
            score += self.current_goal_weight * direction_score
        
        # Chemical gradient and success memory score
        chemical_score = self.pheromone_grid[candidate_node]
        success_memory = self.grid[candidate_node].success_memory
        combined_score = 0.7 * chemical_score + 0.3 * success_memory
        score += self.chemical_gradient_weight * combined_score
        
        # Exploration score with dynamic weight
        if not self.grid[candidate_node].visited:
            score += self.current_exploration_weight
        
        # Nutrient score
        nutrient_score = self.grid[candidate_node].nutrient_value
        score += self.current_nutrient_weight * nutrient_score
        
        return score

    def _update_success_memory(self, path):
        """Update success memory for nodes in successful paths."""
        decay_rate = 0.95  # Memory decay rate
        
        # Decay existing memory
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                self.grid[i, j].success_memory *= decay_rate
        
        # Strengthen memory along successful path
        for node in path:
            self.grid[node].success_memory = min(1.0, self.grid[node].success_memory + 0.2)
        
        # Add path to successful paths
        self.successful_paths.append(path)
        if len(self.successful_paths) > self.path_success_threshold:
            self.successful_paths.pop(0)

    def _handle_stress_response(self, node: Tuple[int, int]) -> None:
        """Handle stress response at a given node."""
        state = self.grid[node]
        
        # Calculate local stress factors
        nutrient_stress = 1 - state.nutrient_value
        resource_stress = 1 - state.resource_level
        crowding_stress = len(self._get_neighbors(node)) / 8  # Normalize by max possible neighbors
        
        # Combined stress level
        stress_level = (nutrient_stress + resource_stress + crowding_stress) / 3
        
        if stress_level > 0.7:  # High stress
            # Increase resource allocation to stressed areas
            state.resource_level = min(1.0, state.resource_level + 0.1)
            
            # Strengthen existing connections
            for neighbor in self.mycelium.neighbors(node):
                if self.mycelium.has_edge(node, neighbor):
                    current_thickness = self.mycelium.edges[node, neighbor].get('thickness', self.min_thickness)
                    self.mycelium.edges[node, neighbor]['thickness'] = min(
                        self.max_thickness,
                        current_thickness + self.thickness_growth_rate * 2
                    )
        
        # Update pheromone levels based on stress
        self.pheromone_grid[node] = max(0.1, self.pheromone_grid[node] - stress_level * 0.1)

    def grow_step(self) -> None:
        """Perform one step of enhanced mycelial growth with biological components."""
        if not self.active_tips:
            return

        new_tips = []
        current_path = []
        
        for tip in self.active_tips:
            # Skip if tip is no longer active
            if not self.mycelium.has_node(tip):
                continue
                
            neighbors = self._get_neighbors(tip)
            if not neighbors:
                continue
                
            # Calculate growth scores for all neighbors
            neighbor_scores = [(n, self._calculate_growth_score(tip, n)) for n in neighbors]
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select best neighbor for primary growth
            if neighbor_scores:
                best_neighbor = neighbor_scores[0][0]
                current_path.append(best_neighbor)
                
                # Add new growth
                if not self.mycelium.has_node(best_neighbor):
                    self.mycelium.add_node(best_neighbor, 
                                         thickness=self.min_thickness,
                                         age=0)
                    self.grid[best_neighbor].visited = True
                    
                if not self.mycelium.has_edge(tip, best_neighbor):
                    self.mycelium.add_edge(tip, best_neighbor)
                    
                new_tips.append(best_neighbor)
                
                # Check for goal reaching
                if self.goal and best_neighbor == self.goal:
                    self.goal_reached = True
                    self._update_success_memory(current_path)
                    
                # Consider branching with adaptive probability
                if len(neighbor_scores) > 1:
                    # Calculate direction to second best neighbor
                    direction = (neighbor_scores[1][0][0] - tip[0], neighbor_scores[1][0][1] - tip[1])
                    branch_prob = self._calculate_branching_probability(tip, direction)
                    
                    if random.random() < branch_prob:
                        # Select second best neighbor for branching
                        branch_neighbor = neighbor_scores[1][0]
                        if not self.mycelium.has_node(branch_neighbor):
                            self.mycelium.add_node(branch_neighbor,
                                                 thickness=self.min_thickness,
                                                 age=0)
                            self.grid[branch_neighbor].visited = True
                            
                        if not self.mycelium.has_edge(tip, branch_neighbor):
                            self.mycelium.add_edge(tip, branch_neighbor)
                            
                        new_tips.append(branch_neighbor)
            
            # Update chemical gradients and stress response
            self._update_chemical_gradients()
            self._handle_stress_response(tip)
            
        # Update active tips
        self.active_tips = new_tips
        
        # Periodic maintenance
        self.step_count += 1
        if self.step_count % self.pruning_interval == 0:
            self._prune_inefficient_hyphae()
        
        # Adjust weights based on current state
        self._adjust_weights()
        
        # If we've made progress, update success memory
        if current_path:
            self._update_success_memory(current_path)
            
        # Resource distribution
        self._distribute_resources()
        
        # Update thickness based on resource flow
        self._update_hyphal_thickness()

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors for a node."""
        i, j = node
        rows, cols = self.grid_size
        neighbors = []
        
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
    
    def _should_fuse(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> bool:
        """Determine if two nodes should fuse."""
        if self.mycelium.has_edge(node1, node2):
            return False
            
        distance = abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
        if distance > 3:
            return False
            
        # Consider resource levels in fusion decision
        state1 = self.grid[node1]
        state2 = self.grid[node2]
        resource_benefit = (state1.resource_level + state2.resource_level) / 2
        
        # Calculate fusion score
        distance_factor = 1.0 - (distance / 3)
        resource_factor = resource_benefit
        network_factor = 1.0 - (len(self.mycelium.edges()) / (self.grid_size[0] * self.grid_size[1]))
        
        fusion_score = 0.4 * distance_factor + 0.4 * resource_factor + 0.2 * network_factor
        return fusion_score > 0.6
    
    def visualize(self, show_grid: bool = True, show_mycelium: bool = True) -> None:
        """Visualize the enhanced mycelium network."""
        plt.figure(figsize=(15, 6))
        
        if show_grid:
            plt.subplot(121)
            grid_data = np.zeros(self.grid_size)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    state = self.grid[i, j]
                    if state.is_obstacle:
                        grid_data[i, j] = -1
                    else:
                        grid_data[i, j] = state.nutrient_value
            
            plt.imshow(grid_data, cmap='YlOrRd')
            plt.title('Nutrient Distribution')
        
        if show_mycelium and self.mycelium.number_of_nodes() > 0:
            plt.subplot(122)
            pos = {node: (node[1], -node[0]) for node in self.mycelium.nodes()}
            
            # Draw nodes with resource levels
            node_colors = [self.grid[node].resource_level for node in self.mycelium.nodes()]
            nx.draw_networkx_nodes(self.mycelium, pos, node_color=node_colors,
                                 cmap='YlGn', node_size=200, alpha=0.8)
            
            # Draw edges with thickness
            edge_widths = [self.mycelium.edges[edge].get('thickness', self.min_thickness) * 3
                          for edge in self.mycelium.edges()]
            nx.draw_networkx_edges(self.mycelium, pos, width=edge_widths,
                                 edge_color='darkgreen', alpha=0.8)
            
            plt.title('Enhanced Mycelium Network')
        
        plt.show() 