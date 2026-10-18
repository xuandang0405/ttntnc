import random
from time import perf_counter
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def select_action(self, env, time_limit_ms=None):
        started = perf_counter()
        actions = env.legal_actions()
        if not actions: raise ValueError("No legal actions")
        action = self.rng.choice(actions)
        return action,{"chosen":list(action),"nodes":len(actions),
                       "latency_ms":round((perf_counter()-started)*1000,4),
                       "action_scores":{f"{r},{c}":1/len(actions) for r,c in actions},
                       "algorithm":"RandomAgent","score_kind":"probability"}
