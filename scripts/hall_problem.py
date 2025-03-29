from environment import MDPEnvironment

class HallProblem(MDPEnvironment):
    """
    The Hall Problem MDP:
    - States: Positions in a hallway (0 to N-1)
    - Actions: Move left (0) or right (1)
    - Transitions: Deterministic except at ends
    - Rewards: +1 for reaching right end, -1 for left end
    - Terminal state: Both ends are terminal
    """
    
    def __init__(self, length=5):
        self.length = length
        self.state_space = list(range(length))
        self.current_state = length // 2  # Start in middle
        
    def get_states(self):
        return self.state_space
    
    def get_actions(self, state):
        return [0, 1]  # 0=left, 1=right
    
    def get_reward(self, state, action, next_state):
        if next_state == self.length - 1:  # Right end
            return 1
        elif next_state == 0:  # Left end
            return -1
        return 0
    
    def get_transition_prob(self, state, action, next_state):
        # Deterministic transitions
        if action == 0:  # Left
            expected = max(0, state - 1)
        else:  # Right
            expected = min(self.length - 1, state + 1)
        
        return 1.0 if next_state == expected else 0.0
    
    def is_terminal(self, state):
        return state == 0 or state == self.length - 1
    
    def reset(self):
        self.current_state = self.length // 2
        return self.current_state
    
    def step(self, action):
        old_state = self.current_state
        
        if action == 0:  # Left
            self.current_state = max(0, self.current_state - 1)
        else:  # Right
            self.current_state = min(self.length - 1, self.current_state + 1)
        
        reward = self.get_reward(old_state, action, self.current_state)
        done = self.is_terminal(self.current_state)
        
        return self.current_state, reward, done, {}
    
    def render(self):
        hallway = ['-'] * self.length
        hallway[self.current_state] = 'o'
        print('[' + ' '.join(hallway) + ']')
