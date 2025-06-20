import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.toy_text.blackjack import BlackjackEnv
from gymnasium.envs.registration import register

class CustomBlackjack(BlackjackEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # 32 for player sum, 11 for dealer card, 2 for usable ace
        self.observation_space = Box(low=0.0, high=1.0, shape=(45,), dtype=np.float64)

    def observation(self, obs):
        """Convert (player_sum, dealer_card, usable_ace) to one-hot vector."""
        player_sum, dealer_card, usable_ace = obs

        one_hot = np.zeros(45, dtype=np.float64)
        if 0 <= player_sum < 32:
            one_hot[player_sum] = 1.0
        if 0 <= dealer_card < 11:
            one_hot[32 + dealer_card] = 1.0
        if 0 <= usable_ace < 2:
            one_hot[32 + 11 + usable_ace] = 1.0

        return one_hot

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self.observation(obs), reward, terminated, truncated, info

register(
    id="CustomBlackjack-v0",
    entry_point= lambda: CustomBlackjack(),
    max_episode_steps=1000
)
