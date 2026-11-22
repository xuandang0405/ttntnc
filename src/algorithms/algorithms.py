import os

import chess
import torch

from algorithms.algo_a2c import A2CAgent
from algorithms.algo_alphazero import AlphaZeroAgent
from algorithms.algo_dqn import DQNAgent
from algorithms.algo_ppo import PPOAgent

ALGORITHMS = {"A2C": A2CAgent, "PPO": PPOAgent, "DQN": DQNAgent, "AlphaZero": AlphaZeroAgent}
ALGORITHM_NAMES = list(ALGORITHMS)
_BY_KEY = {cls.algorithm: cls for cls in ALGORITHMS.values()}


def _color_name(color):
    return "White" if color == chess.WHITE else "Black"

 
def make_agent(algorithm, color, device, episodes_per_update=1):
    cls = ALGORITHMS.get(algorithm) if isinstance(algorithm, str) else None
    if cls is None and isinstance(algorithm, str):
        cls = _BY_KEY.get(algorithm.lower())
    if cls is None:
        expected = ", ".join(ALGORITHM_NAMES)
        raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {expected}")
    return cls(color, _color_name(color), device, episodes_per_update)

 
def _read(path):
    ck = torch.load(path, map_location="cpu")
    if not isinstance(ck, dict) or "net" not in ck:
        raise ValueError("not a model checkpoint from this program")
    return ck


def read_checkpoint_info(path):
    ck = _read(path)
    cls = _BY_KEY.get(ck.get("algorithm", "a2c"))
    if cls is None:
        raise ValueError(f"unknown algorithm {ck.get('algorithm')!r}")
    return {"label": cls.label, "key": cls.algorithm}


def load_agent(path, color, device, episodes_per_update=1):
    ck = _read(path)
    cls = _BY_KEY.get(ck.get("algorithm", "a2c"))
    if cls is None:
        raise ValueError(f"unknown algorithm {ck.get('algorithm')!r}")
    agent = cls(color, _color_name(color), device, episodes_per_update)
    agent.load_state(ck)
    agent.loaded_from = path
    return agent


def build_agent(cfg, color, device, episodes_per_update=1):
    if cfg.get("path"):
        return load_agent(cfg["path"], color, device, episodes_per_update)
    return make_agent(cfg["algorithm"], color, device, episodes_per_update)
