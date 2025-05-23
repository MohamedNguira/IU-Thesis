# run_experiments.py
import numpy as np
import matplotlib.pyplot as plt
from environments.baird import BairdCounterexample
from agents.q_learning import QLearning

def run_baird_experiment(num_episodes=1000, episode_length=100):
    """Run corrected Baird's counterexample experiment"""
    env = BairdCounterexample(episode_length=episode_length)
    agent = QLearning(num_features=env.num_states, num_actions=env.num_actions,
                     alpha=0.01, gamma=0.99)
    
    behavior_policy = env.get_behavior_policy()
    weight_history = []
    max_weight_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = np.random.choice(env.num_actions, p=behavior_policy[env.current_state])
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        
        # Record weights after each episode
        current_weights = agent.get_weights()
        weight_history.append(current_weights.copy())
        max_weight_history.append(np.max(np.abs(current_weights)))
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Max weight: {max_weight_history[-1]:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot maximum weight magnitude
    plt.subplot(1, 2, 1)
    plt.plot(max_weight_history)
    plt.title("Maximum Weight Magnitude")
    plt.xlabel("Episode")
    plt.ylabel("Max |weight|")
    
    # Plot final weights
    plt.subplot(1, 2, 2)
    final_weights = weight_history[-1]
    plt.imshow(final_weights, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title("Final Weight Matrix")
    plt.xlabel("Action")
    plt.ylabel("State Feature")
    plt.xticks([0, 1], ["Solid", "Dashed"])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Running corrected Baird's counterexample...")
    run_baird_experiment(num_episodes=2000, episode_length=100)