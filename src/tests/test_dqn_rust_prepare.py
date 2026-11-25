import unittest
from unittest.mock import patch

import chess
import numpy as np
import torch

try:
    import algorithms_rs
except ImportError:
    algorithms_rs = None

from algorithms.algo_dqn import DQNAgent
from encoding import ACTION_SIZE, INPUT_CHANNELS, encode, legal_moves_map


HAS_DQN_PREP = algorithms_rs is not None and hasattr(algorithms_rs, "prepare_dqn_batch")


@unittest.skipUnless(HAS_DQN_PREP, "Rust DQN preparation is not installed")
class RustDqnPreparationTests(unittest.TestCase):
    def make_boards(self):
        boards = [chess.Board()]
        moved = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"):
            moved.push_uci(uci)
        boards.append(moved)
        en_passant = chess.Board()
        en_passant.push_uci("e2e4")
        boards.append(en_passant)
        boards.append(chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"))
        repeated = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
            repeated.push_uci(uci)
        boards.append(repeated)
        return boards

    def test_batch_preparation_matches_python_for_both_perspectives(self):
        boards = self.make_boards()
        root_fens = [board.root().fen() for board in boards]
        move_lists = [[move.uci() for move in board.move_stack] for board in boards]
        for color in (chess.WHITE, chess.BLACK):
            states, legal, ucis = algorithms_rs.prepare_dqn_batch(
                root_fens, move_lists, color == chess.WHITE
            )
            self.assertEqual(states.shape, (len(boards), INPUT_CHANNELS * 64))
            self.assertEqual(len(ucis), len(boards))
            for i, board in enumerate(boards):
                np.testing.assert_array_equal(
                    states[i].reshape(INPUT_CHANNELS, 8, 8),
                    encode(board, color)[0].numpy(),
                )
                valid = legal[i][legal[i] >= 0]
                actual = {
                    int(action): chess.Move.from_uci(uci)
                    for action, uci in zip(valid, ucis[i])
                }
                self.assertEqual(actual, legal_moves_map(board, color))

    def test_native_action_uses_rust_preparation(self):
        class ZeroNet(torch.nn.Module):
            def forward(self, states):
                return (
                    torch.zeros((len(states), ACTION_SIZE), device=states.device),
                    torch.zeros(len(states), device=states.device),
                )

        class FirstLegalCore:
            def act(self, game_ids, states, q, legal):
                return np.asarray([row[row >= 0][0] for row in legal], dtype=np.int64)

        agent = DQNAgent.__new__(DQNAgent)
        agent.color = chess.WHITE
        agent.name = "White"
        agent.device = torch.device("cpu")
        agent.net = ZeroNet()
        agent.core = FirstLegalCore()
        agent.eps = {}
        agent.metrics = None
        agent.coach = None
        board = chess.Board()

        with patch("algorithms.algo_dqn.legal_moves_map", side_effect=AssertionError):
            move = agent.act_batch([0], [board])[0]
        self.assertIn(move, board.legal_moves)

    def test_python_fallback_remains_usable(self):
        class ZeroNet(torch.nn.Module):
            def forward(self, states):
                return (
                    torch.zeros((len(states), ACTION_SIZE), device=states.device),
                    torch.zeros(len(states), device=states.device),
                )

        agent = DQNAgent.__new__(DQNAgent)
        agent.color = chess.WHITE
        agent.name = "White"
        agent.device = torch.device("cpu")
        agent.net = ZeroNet()
        agent.core = None
        agent._act_steps = 0
        agent.eps = {}
        agent.metrics = None
        agent.coach = None
        board = chess.Board()

        move = agent.act_batch([0], [board])[0]
        self.assertIn(move, board.legal_moves)
        self.assertEqual(agent.act_steps, 1)


if __name__ == "__main__":
    unittest.main()
