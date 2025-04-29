from mycelial_router.core import MycelialRouter
import time
import matplotlib.pyplot as plt
import networkx as nx

def test_rl_growth():
    # Create a router instance with RL parameters
    router = MycelialRouter(
        grid_size=(10, 10),
        learning_rate=0.3,
        discount_factor=0.95,
        exploration_rate=0.3,
        obstacle_prob=0.05,
        failure_prob=0.02,
        nutrient_range=(0.5, 1.0)
    )
    
    # Start growth from the center
    start_node = (5, 5)
    router.start_growth(start_node)
    
    # Track rewards over time
    rewards = []
    
    # Grow for multiple episodes
    n_episodes = 10
    steps_per_episode = 30
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        episode_rewards = []
        
        # Reset growth but keep learned Q-values
        router.mycelium = nx.Graph()
        router.mycelium.add_node(start_node)
        router.grid[start_node].visited = True
        router.active_tips = [start_node]
        
        for step in range(steps_per_episode):
            # Get current state for visualization
            current_reward = sum(
                router._compute_reward(node)
                for node in router.mycelium.nodes()
            )
            episode_rewards.append(current_reward)
            
            # Grow one step
            router.grow_step()
            
            print(f"Step {step + 1}/{steps_per_episode} - Current reward: {current_reward:.2f}")
        
        rewards.append(episode_rewards)
        
        # Visualize after each episode
        plt.figure(figsize=(15, 6))
        
        # Plot grid and mycelium
        router.visualize(show_grid=True, show_mycelium=True)
        
        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards)
        plt.title(f'Rewards during Episode {episode + 1}')
        plt.xlabel('Step')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.show()
    
    # Plot learning progress
    plt.figure(figsize=(10, 5))
    for i, episode_rewards in enumerate(rewards):
        plt.plot(episode_rewards, label=f'Episode {i+1}')
    plt.title('Learning Progress')
    plt.xlabel('Step')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_rl_growth() 