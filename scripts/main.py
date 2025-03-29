import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from hall_problem import HallProblem
from star_problem import StarProblem
from q_learning import QLearningAgent

def test_environment(env_class, env_params={}, episodes=1000):
    """Test an environment with Q-learning"""
    print(f"\nTesting {env_class.__name__}...")
    env = env_class(**env_params)
    agent = QLearningAgent(env)
    print("HII")
    rewards, steps = agent.learn(episodes)
    print("HI")
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    plt.tight_layout()
    plt.show()
    
    # Show learned policy
    print("\nLearned Policy:")
    policy = agent.get_optimal_policy()
    for state, action in sorted(policy.items()):
        print(f"State {state}: Best action = {action}")
    
    # Show Q-table
    agent.visualize_q_table()
    
    return agent

if __name__ == "__main__":
    # Test improved Star Problem
    print("Testing improved Star Problem...")
    star_env = StarProblem(max_steps=20)  # Reasonable step limit
    star_agent = QLearningAgent(star_env, epsilon=0.3, epsilon_decay=0.998)
    
    rewards, steps = star_agent.learn(episodes=1000)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    plt.tight_layout()
    plt.show()
    
    # Show learned policy
    print("\nLearned Policy:")
    policy = star_agent.get_optimal_policy()
    for state, action in sorted(policy.items()):
        print(f"State {state}: Best action = {action}")