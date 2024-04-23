from abc import ABC, abstractmethod

class TradingPolicy(ABC):
    def __init__(self):
        self.env = None
        
    @abstractmethod
    def set_env(self, env):
        self.env = env

    def get_action(self, obs):
        # Implement your trading strategy here
        # Return 0 for hold, 1 for buy, 2 for sell
        return 0  # Replace with your trading logic
