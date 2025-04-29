import numpy as np
from typing import Dict, Tuple, List
import random
from collections import defaultdict

class MyceliumRLAgent:
    def __init__(self, grid_size: Tuple[int, int], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize the reinforcement learning agent for mycelium growth.
        
        Args:
            grid_size: Size of the grid (rows, columns)
            start: Starting position
            goal: Goal position
        """
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.learning_rate = 0.3  # Higher learning rate
        self.discount_factor = 0.95  # Higher discount
        self.exploration_rate = 0.1  # Lower initial exploration
        self.exploration_decay = 0.995  # Decay exploration over time
        self.min_exploration = 0.01  # Minimum exploration rate
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0.1))  # Initialize with small positive value
        
        # Possible actions (including diagonals)
        self.actions = [
            (-1, 0),   # up
            (1, 0),    # down
            (0, -1),   # left
            (0, 1),    # right
            (-1, -1),  # up-left
            (-1, 1),   # up-right
            (1, -1),   # down-left
            (1, 1),    # down-right
        ]
    
    def get_state_key(self, position: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert position to state key including goal direction."""
        goal_dx = self.goal[0] - position[0]
        goal_dy = self.goal[1] - position[1]
        return (position[0], position[1], 
                1 if goal_dx > 0 else (-1 if goal_dx < 0 else 0),
                1 if goal_dy > 0 else (-1 if goal_dy < 0 else 0))
    
    def get_valid_actions(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get list of valid actions from current position."""
        valid_actions = []
        for action in self.actions:
            new_pos = (position[0] + action[0], position[1] + action[1])
            if (0 <= new_pos[0] < self.grid_size[0] and 
                0 <= new_pos[1] < self.grid_size[1]):
                valid_actions.append(action)
        return valid_actions
    
    def get_q_value(self, state: Tuple[int, int], action: Tuple[int, int]) -> float:
        """Get Q-value for state-action pair."""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0.0
        return self.q_table[state_key][action]
    
    def choose_action(self, state: Tuple[int, int], valid_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Choose action using epsilon-greedy policy with goal-directed bias."""
        state_key = self.get_state_key(state)
        
        if random.random() < self.exploration_rate:
            # During exploration, bias towards goal direction
            goal_dx = self.goal[0] - state[0]
            goal_dy = self.goal[1] - state[1]
            
            # Sort actions by how well they align with goal direction
            scored_actions = []
            for action in valid_actions:
                # Calculate how well this action aligns with goal direction
                alignment = (
                    (1 if (goal_dx > 0 and action[0] > 0) or (goal_dx < 0 and action[0] < 0) else 0) +
                    (1 if (goal_dy > 0 and action[1] > 0) or (goal_dy < 0 and action[1] < 0) else 0)
                )
                scored_actions.append((action, alignment))
            
            # Choose randomly but favor actions aligned with goal
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            cutoff = max(1, len(scored_actions) // 2)  # Take top half of actions
            return random.choice(scored_actions[:cutoff])[0]
        
        # Get Q-values for valid actions
        q_values = {action: self.q_table[state_key][action] for action in valid_actions}
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state: Tuple[int, int], action: Tuple[int, int], reward: float, next_state: Tuple[int, int], done: bool):
        """Update Q-value using Q-learning with goal-based reward shaping."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Add goal-based reward shaping
        current_dist = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
        next_dist = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
        reward += 0.1 * (current_dist - next_dist)  # Small bonus for getting closer to goal
        
        if done:
            target = reward
        else:
            next_actions = self.get_valid_actions(next_state)
            if not next_actions:
                target = reward
            else:
                next_q_values = [self.q_table[next_state_key][a] for a in next_actions]
                target = reward + self.discount_factor * max(next_q_values)
        
        # Update Q-value
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = (1 - self.learning_rate) * current_q + self.learning_rate * target
        
        # Decay exploration rate
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)
    
    def get_policy(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """Get the current policy (best action for each state)."""
        policy = {}
        for state_key in self.q_table:
            if self.q_table[state_key]:
                best_action = max(
                    self.q_table[state_key].items(),
                    key=lambda x: x[1]
                )[0]
                policy[state_key] = best_action
        return policy 