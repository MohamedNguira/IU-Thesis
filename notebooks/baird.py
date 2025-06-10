import gymnasium as gym
from typing import Optional
import numpy as np 
class BairdsCounterexample(gym.Env):

    def __init__(self):
        super(BairdsCounterexample, self).__init__()
        # Dashed line action = 0, solid line action = 1.
        self.action_space = gym.spaces.Discrete(2)
        self.steps = 0
        # Upper states = 0-5, lower state = 6.
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64)
        self.np_random = None
        
        
    def step(self, action):
        assert self.action_space.contains(action)

        state = None
        if action == 0:  # Dashed line action
            state = self.np_random.integers(6)
        if action == 1:  # Solid line action
            state = 6
        self.steps += 1
        obs = np.zeros((7,))
        obs[state] = 1.0

        Terminated = self.steps > 100
        reward = 0.0 if state == 6 else 0.0
        return obs, reward, Terminated, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        state = self.np_random.integers(7)
        obs = np.zeros((7,), dtype=np.float64)
        obs[state] = 1.0
        return obs, {}

    def render(self, mode='human'):
        return
    def close(self):
        return
    
gym.envs.registration.register(
    id='BairdsCounterexample-v0',
    entry_point=lambda: BairdsCounterexample(),
    nondeterministic=True,
    
)
