from abc import ABC, abstractmethod
from backend.core.env import ChessEnv

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, env: ChessEnv, time_limit_ms: float | None = None) -> tuple[str,dict]:
        raise NotImplementedError
