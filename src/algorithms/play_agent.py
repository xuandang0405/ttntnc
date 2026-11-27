from functools import partial

import torch

from algorithms.algo_alphazero import _evaluate, run_mcts, select_search_moves
from algorithms.algo_base import select_policy_moves
from algorithms.algo_config import AZ_PLAY_SIMULATIONS
from algorithms.legacy_model import (LEGACY_ACTION_SIZE, LegacyPolicyValueNet, legacy_encode,
                                     legacy_legal_moves)
from encoding import encode, legal_moves_map
from model import PolicyValueNet, load_policy_value_weights
from strategy import StrategyLearner, lookahead_move

LEGACY_AZ_PLAY_SIMULATIONS = 128


def is_legacy_checkpoint(checkpoint):
    return checkpoint.get("format") == 2 and "c1.weight" in checkpoint["net"]


class PlayAgent:
    """Run a checkpoint without creating a coach, optimizer or training buffers."""

    def __init__(self, checkpoint, agent_class, device):
        self.label = agent_class.label
        self.algorithm = agent_class.algorithm
        self.temperature = agent_class.play_temperature
        self.device = device
        self.search_evaluator = None
        self.simulations = AZ_PLAY_SIMULATIONS
        self.encode_position = encode
        self.map_moves = legal_moves_map
        if is_legacy_checkpoint(checkpoint):
            self.net = LegacyPolicyValueNet()
            self.net.load_state_dict(checkpoint["net"])
            self.encode_position = legacy_encode
            self.map_moves = legacy_legal_moves
            self.search_evaluator = partial(_evaluate, encode_position=legacy_encode,
                                            map_moves=legacy_legal_moves, action_size=LEGACY_ACTION_SIZE)
            self.simulations = LEGACY_AZ_PLAY_SIMULATIONS
            self.has_aux_weights = True
        else:
            self.net = PolicyValueNet()
            missing = load_policy_value_weights(self.net, checkpoint["net"])
            self.has_aux_weights = not missing
        self.net.to(device).eval()
        self.strategy = StrategyLearner(self)
        self.strategy.load(checkpoint.get("extra", {}).get("strategy"))

    def supports_lookahead(self):
        return self.algorithm != "alphazero" and self.has_aux_weights and self.strategy.ready

    def choose_move(self, board, greedy=False, lookahead=False):
        if lookahead and self.supports_lookahead():
            return lookahead_move(self, board, greedy,
                                  encode_position=self.encode_position, map_moves=self.map_moves)
        return self.select_moves([board], greedy)[0]

    @torch.inference_mode()
    def select_moves(self, boards, greedy=False):
        if not boards:
            return []
        if self.algorithm == "alphazero":
            results = run_mcts(self.net, boards, self.simulations, self.device, noise=False,
                               evaluate=self.search_evaluator, allow_python_fallback=True)
            return select_search_moves(results, greedy)
        maps = [self.map_moves(board, board.turn) for board in boards]
        states = torch.cat([self.encode_position(board, board.turn) for board in boards]).to(self.device)
        return select_policy_moves(self.net, states, maps, self.temperature, greedy)
