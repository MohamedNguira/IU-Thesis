import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.registration import register

class CustomFrozenLake(FrozenLakeEnv):
    def __init__(self, **kwargs):
        kwargs={"is_slippery": True}
        super().__init__(**kwargs)

        # Replace Discrete(n) with Box(n,) to represent one-hot vector
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.nrow * self.ncol,), dtype=np.float64)

    def observation(self, state):
        """Convert integer state to one-hot encoded numpy array."""
        one_hot = np.zeros(self.nrow * self.ncol, dtype=np.float64)
        one_hot[state] = 1.0
        return one_hot

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        return self.observation(state), info

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        return self.observation(state), reward, terminated, truncated, info

register(
    id="FrozenLakeCustom-v0",
    entry_point=lambda: CustomFrozenLake(),
    max_episode_steps=500
)

