import numpy as np
from typing import List, Tuple, Dict
import heapq

class AStarPathfinder:
    def __init__(self, grid: np.ndarray, grid_size: Tuple[int, int]):
        """
        Initialize A* pathfinder.
        
        Args:
            grid: Grid of NodeState objects
            grid_size: Size of the grid (rows, columns)
        """
        self.grid = grid
        self.grid_size = grid_size
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors for a position."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < self.grid_size[0] and 
                0 <= new_pos[1] < self.grid_size[1]):
                state = self.grid[new_pos]
                if not state.is_obstacle and not state.is_failed:
                    neighbors.append(new_pos)
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        Find path from start to goal using A*.
        
        Returns:
            Tuple of (path, total_cost)
        """
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
            
            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + self.grid[next_pos].traversal_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Check if goal was reached
        if goal not in came_from:
            return [], float('inf')
            
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path, cost_so_far.get(goal, float('inf')) 