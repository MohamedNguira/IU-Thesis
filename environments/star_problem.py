import gym
import gym.utils.seeding

class StarProblem(gym.Env):
    """Star Problem MDP with 6 states (center + 5 peripherals)."""
    
    action_space = gym.spaces.Discrete(2)  # 0: random peripheral, 1: back to center
    observation_space = gym.spaces.Discrete(6)  # State 4 is rewarding state, state 5 is the center
    reward_range = (0, 1)  # Reward only in State 5

    def __init__(self):
        self.seed()

    def step(self, action):
        assert self.action_space.contains(action)
        
        reward = 0.0
        done = False
        
        if action == 0:  # Move to random peripheral (States 1-5)
            next_state = self.np_random.integers(0, 5)
            if next_state == 4:  # Only State 4 gives reward
                reward = 1.0
        else:  # Action 1: Return to center (State 0)
            next_state = 5
            
        return next_state, 0.0, done, {}

    def reset(self):
        return self.np_random.integers(self.observation_space.n)  # Start anywhere

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

# Register the environment
gym.envs.registration.register(
    id='StarProblem-v0',
    entry_point=lambda: StarProblem(),
    nondeterministic=True,
)