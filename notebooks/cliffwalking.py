import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from gymnasium.envs.registration import register

class CustomCliffWalking(CliffWalkingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_states = self.shape[0] * self.shape[1]
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.num_states,), dtype=np.float64)

    def observation(self, state):
        """Convert integer state to one-hot encoded numpy array."""
        one_hot = np.zeros(self.num_states, dtype=np.float64)
        one_hot[state] = 1.0
        return one_hot

    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        return self.observation(state), info

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        return self.observation(state), reward, terminated, truncated, info

register(
    id="CustomCliffWalking-v0",
    entry_point=lambda: CustomCliffWalking(),
    max_episode_steps=500
)

