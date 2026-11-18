import random
from time import perf_counter
from backend.agents.base_agent import BaseAgent
from backend.core.env import ChessEnv
from backend.core.eval_pst import evaluate

class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None, rng: random.Random | None = None):
        self.rng = rng if rng is not None else random.Random(seed)

    def select_action(self, env: ChessEnv, time_limit_ms: float | None = None) -> tuple[str,dict]:
        start = perf_counter()
        legal = env.legal_actions()
        if not legal:
            raise ValueError('Không có nước đi hợp lệ.')
        chosen = self.rng.choice(legal)
        return chosen, {'chosen':chosen,'algorithm':'Random','nodes':0,'latency_ms':round((perf_counter()-start)*1000,4),
                        'eval_score':evaluate(env.board)/100,'top_moves':{m:1/len(legal) for m in legal},'legal_count':len(legal),'selected_probability':1/len(legal)}
