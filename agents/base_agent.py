from abc import ABC, abstractmethod
from backend.core.env import ShadowHeistEnv

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, env: ShadowHeistEnv, time_limit_ms=None):
        """Return ((row, col), telemetry). Must not mutate the environment."""
        raise NotImplementedError
