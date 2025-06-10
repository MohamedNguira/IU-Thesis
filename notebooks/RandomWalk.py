import gymnasium as gym
from typing import Optional
import numpy as np 

# Constants
LEFT = 0
RIGHT = 1

class RandomWalk(gym.Env):
    def __init__(self, size=5):
        self.size = size
        self.state = size // 2 + 1
        self.np_random = None
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(size + 2,), dtype=np.float64)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.size // 2 + 1
        obs = np.zeros((self.size + 2,), dtype=np.float64)
        obs[self.state] = 1.0
        return obs, {}
    def render(self, mode='human'):
        return
    def close(self):
        return
    
    def step(self, action):
        if action == LEFT:
            self.state = self.state - 1
        elif action == RIGHT:
            self.state = self.size + 1

        reward = 0
        terminal = False

        if self.state == 0:
            reward = -1
            terminal = True

        elif self.state == self.size + 1:
            reward = 1
            terminal = True
        
        feature = [0.0] * (self.size + 2)
        feature[self.state] = 1.0
        feature = tuple(feature) 

        return (feature,reward, terminal, False, {})

gym.envs.registration.register(
    id='RandomWalk-v0',
    entry_point=lambda: RandomWalk(),
    nondeterministic=True,
    
)
