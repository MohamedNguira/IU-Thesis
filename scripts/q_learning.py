from environment import MDPEnvironment
import numpy as np

class QLearningAgent:
    """Improved Q-learning agent with epsilon decay"""
    
    def __init__(self, env: MDPEnvironment, alpha=0.1, gamma=0.9, epsilon=0.1, 
                 epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table
        self.q_table = {}
        for state in env.get_states():
            self.q_table[state] = {}
            for action in env.get_actions(state):
                self.q_table[state][action] = 0.0
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.env.get_actions(state))
        else:
            # Exploit: best action
            return max(self.q_table[state].items(), key=lambda x: x[1])[0]
    
    def decay_epsilon(self):
        """Decay the exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def learn(self, episodes=1000):
        """Run Q-learning algorithm with epsilon decay"""
        rewards = []
        steps = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Q-learning update
                best_next_action = max(self.q_table[next_state].items(), key=lambda x: x[1])[0]
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error
                
                state = next_state
                total_reward += reward
                step += 1
            
            # Decay epsilon after each episode
            self.decay_epsilon()
            
            rewards.append(total_reward)
            steps.append(step)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode + 1}: Avg reward (last 100) = {avg_reward:.1f}, " +
                      f"Steps = {step}, Epsilon = {self.epsilon:.3f}")
        
        return rewards, steps
