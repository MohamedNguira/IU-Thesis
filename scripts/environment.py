from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

class MDPEnvironment(ABC):
    """Abstract base class for MDP environments"""
    
    @abstractmethod
    def get_states(self) -> list:
        """Return list of all possible states"""
        pass
    
    @abstractmethod
    def get_actions(self, state: Any) -> list:
        """Return list of possible actions for a given state"""
        pass
    
    @abstractmethod
    def get_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """Return reward for state, action, next_state transition"""
        pass
    
    @abstractmethod
    def get_transition_prob(self, state: Any, action: Any, next_state: Any) -> float:
        """Return transition probability P(next_state | state, action)"""
        pass
    
    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        """Check if a state is terminal"""
        pass
    
    @abstractmethod
    def reset(self) -> Any:
        """Reset the environment to initial state"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Execute one action in the environment"""
        pass
    
    @abstractmethod
    def render(self):
        """Visualize the current state of the environment"""
        pass
