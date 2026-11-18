import unittest

import chess
import numpy as np
import torch

from encoding import ACTION_SIZE, HISTORY_LENGTH, INPUT_CHANNELS, encode, legal_moves_map, move_to_index
from model import PolicyValueNet
from algorithms.algo_alphazero import run_mcts


class ZeroNet(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], ACTION_SIZE), device=x.device), torch.zeros(x.shape[0], device=x.device)


class EncodingTests(unittest.TestCase):
    def test_input_shape_and_history(self):
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        state = encode(board, board.turn)
        self.assertEqual(tuple(state.shape), (1, INPUT_CHANNELS, 8, 8))
        self.assertEqual(INPUT_CHANNELS, HISTORY_LENGTH * 12 + 8)
        self.assertGreater(float(state.sum()), 0.0)

    def test_all_promotion_choices_have_distinct_actions(self):
        board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        promotions = [m for m in board.legal_moves if m.from_square == chess.A7]
        self.assertEqual({m.promotion for m in promotions}, {
            chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
        })
        indices = [move_to_index(m, chess.WHITE) for m in promotions]
        self.assertEqual(len(indices), len(set(indices)))
        legal = legal_moves_map(board, chess.WHITE)
        self.assertTrue(all(i in legal for i in indices))

    def test_repetition_plane_uses_move_history(self):
        board = chess.Board()
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        for uci in cycle * 2:
            board.push_uci(uci)
        self.assertTrue(board.is_repetition(3))
        state = encode(board, board.turn)[0].numpy()
        repetition_plane = HISTORY_LENGTH * 12 + 6
        self.assertTrue(np.allclose(state[repetition_plane], 1.0))

    def test_residual_network_shapes(self):
        net = PolicyValueNet()
        x = torch.zeros(2, INPUT_CHANNELS, 8, 8)
        policy, value = net(x)
        opponent, phase = net.forward_aux(x)
        self.assertEqual(tuple(policy.shape), (2, ACTION_SIZE))
        self.assertEqual(tuple(opponent.shape), (2, ACTION_SIZE))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertEqual(tuple(phase.shape), (2, 3))

    def test_mcts_recognizes_terminal_repetition(self):
        board = chess.Board()
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        for uci in cycle * 4:
            board.push_uci(uci)
        self.assertIsNotNone(board.outcome(claim_draw=True))
        result = run_mcts(ZeroNet(), [board], 4, torch.device("cpu"), noise=False)
        self.assertEqual(result[0][0], [])


if __name__ == "__main__":
    unittest.main()
