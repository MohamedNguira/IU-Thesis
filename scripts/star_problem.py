from environment import MDPEnvironment

class StarProblem(MDPEnvironment):
    """
    Improved Star Problem MDP:
    - States: Center (C) and peripheral states (P1, P2, P3, P4)
    - Actions: At center: choose any peripheral (1-4), at peripheral: only return (0)
    - Transitions: Deterministic
    - Rewards: +10 for reaching P1, +2 for reaching P2, -2 for P3, -10 for P4
    - Small negative reward (-0.1) for each transition to encourage efficiency
    - Episode terminates after max_steps to prevent infinite loops
    """
    
    def __init__(self, max_steps=100):
        self.state_space = ['C', 'P1', 'P2', 'P3', 'P4']
        self.current_state = 'C'
        self.max_steps = max_steps
        self.steps = 0
        
    def get_states(self):
        return self.state_space
    
    def get_actions(self, state):
        if state == 'C':
            return [1, 2, 3, 4]  # Choose which peripheral to go to
        else:
            return [0]  # Only action is to return to center
    
    def get_reward(self, state, action, next_state):
        # Primary rewards for reaching peripheral states
        if next_state == 'P1':
            return 10
        elif next_state == 'P2':
            return 2
        elif next_state == 'P3':
            return -2
        elif next_state == 'P4':
            return -10
        
        # Small negative reward for each transition to encourage efficiency
        return -0.1
    
    def get_transition_prob(self, state, action, next_state):
        # Deterministic transitions
        if state == 'C' and action in [1, 2, 3, 4] and next_state == f'P{action}':
            return 1.0
        elif state.startswith('P') and action == 0 and next_state == 'C':
            return 1.0
        return 0.0
    
    def is_terminal(self, state):
        # Episode terminates after max_steps to prevent infinite loops
        return self.steps >= self.max_steps
    
    def reset(self):
        self.current_state = 'C'
        self.steps = 0
        return self.current_state
    
    def step(self, action):
        old_state = self.current_state
        
        if old_state == 'C':
            if action in [1, 2, 3, 4]:
                self.current_state = f'P{action}'
        else:  # In a peripheral state
            if action == 0:
                self.current_state = 'C'
        
        self.steps += 1
        reward = self.get_reward(old_state, action, self.current_state)
        done = self.is_terminal(self.current_state)
        
        return self.current_state, reward, done, {}
    
    def render(self):
        print(f"Current state: {self.current_state} (Step {self.steps}/{self.max_steps})")
