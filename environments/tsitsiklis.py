import gym
import gym.utils.seeding
import numpy as np

class TsitsiklisVanRoyCounterexample(gym.Env):
    """ 
    Tsitsiklis and Van Roy's Counterexample MDP.
    This demonstrates divergence of Q-learning with linear function approximation.
    """
    
    # Two actions: left (0) and right (1)
    action_space = gym.spaces.Discrete(2)
    # Three states: 0, 1, and 2
    observation_space = gym.spaces.Discrete(3)
    # Rewards are always 0
    reward_range = (0, 0)
    
    def __init__(self):
        self.seed()
        self.state = None
        
    def step(self, action):
        assert self.action_space.contains(action)
        
        # Transition dynamics:
        if self.state == 0:
            next_state = 1
        elif self.state == 1:
            next_state = self.state + action
        else:  # state == 2
            next_state = 2
            
        # All rewards are d0
        reward = 0.0
        done = (next_state == 2)
        
        self.state = next_state
        return next_state, reward, done, {}
    
    def reset(self):
        # Start in state 0
        self.state = 0
        return self.state
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

gym.envs.registration.register(
    id='TsitsiklisVanRoyCounterexample-v0',
    entry_point=lambda: TsitsiklisVanRoyCounterexample(),
    nondeterministic=True,
)