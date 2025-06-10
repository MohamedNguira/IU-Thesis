import gymnasium as gym
from typing import Optional
import numpy as np 

# Constants
RIGHT = 0
SKIP = 1

class BoyanChain(gym.Env):
    def __init__(self, size = 14):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(size,), dtype=np.float64)
        self.size = size
        self.np_random = None
        self.state = size - 1

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.size - 1
        obs = np.zeros((self.size,), dtype=np.float64)
        obs[self.state] = 1.0
        return obs, {}
    
    def render(self, mode='human'):
        return
    def close(self):
        return

    def step(self, action):
        reward = -3
        terminal = False

        if action == SKIP and self.state <= 2:
            feature = [0.0] * self.size
            feature[self.state] = 1.0
            feature = tuple(feature) 
            return (feature, -1000, True, False, {})

        if action == RIGHT:
            self.state = self.state - 1
        elif action == SKIP:
            self.state = self.state - 2

        if (self.state == 1):
            reward = -2
        if (self.state == 0):
            reward = 0
            terminal = True

        feature = [0.0] * self.size
        feature[self.state] = 1.0
        feature = tuple(feature) 

        return (feature,reward, terminal, False, {})

gym.envs.registration.register(
    id='BoyanChain-v0',
    entry_point=lambda: BoyanChain(),
    nondeterministic=True,
    
)
