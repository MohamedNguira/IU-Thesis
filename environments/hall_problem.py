import gym
import gym.utils.seeding

class HallProblem(gym.Env):
    """Hall Problem MDP with 6 states (linear chain)."""
    
    action_space = gym.spaces.Discrete(2)  # 0: backward, 1: forward
    observation_space = gym.spaces.Discrete(6)  # States 0 to 5
    reward_range = (0, 1)  # Reward only in State 5

    def __init__(self):
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)
        
        reward = 0.0
        done = False
        current_state = self.state  # Track current state
        
        if action == 1:  # Move forward
            next_state = min(current_state + 1, 5)
        else:  # Action 0: Move backward
            next_state = max(current_state - 1, 0)
        
        if next_state == 5:  # Goal state
            reward = 1.0
            
        self.state = next_state
        return next_state, 0.0, done, {}

    def reset(self):
        self.state = 0  # Always start at State 0
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

# Register the environment
gym.envs.registration.register(
    id='HallProblem-v0',
    entry_point=lambda: HallProblem(),
    nondeterministic=False,  # Deterministic transitions
)