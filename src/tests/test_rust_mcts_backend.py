import unittest
from unittest.mock import patch

import chess
import numpy as np
import torch

try:
    import algorithms_rs
except ImportError:
    algorithms_rs = None

from algorithms import algo_alphazero
from encoding import ACTION_SIZE


class ZeroNet(torch.nn.Module):
    def forward(self, states):
        return (
            torch.zeros((len(states), ACTION_SIZE), device=states.device),
            torch.zeros(len(states), device=states.device),
        )


class RustMctsBackendTests(unittest.TestCase):
    def setUp(self):
        self.boards = [chess.Board()]
        opening = chess.Board()
        for move in ("e2e4", "e7e5", "g1f3"):
            opening.push_uci(move)
        self.boards.append(opening)

    @unittest.skipIf(algorithms_rs is None, "Rust extension is not installed")
    def test_rust_and_python_searches_match(self):
        net = ZeroNet()
        with patch("algorithms.algo_alphazero.AZ_USE_RUST_MCTS", False):
            python_results = algo_alphazero.run_mcts(
                net, self.boards, 4, torch.device("cpu"), noise=False
            )
        with patch("algorithms.algo_alphazero.AZ_USE_RUST_MCTS", True):
            rust_results = algo_alphazero.run_mcts(
                net, self.boards, 4, torch.device("cpu"), noise=False
            )

        for expected, actual in zip(python_results, rust_results):
            self.assertEqual(expected[0], actual[0])
            np.testing.assert_array_equal(expected[1], actual[1])
            self.assertEqual(expected[2], actual[2])

    def test_requested_rust_backend_requires_extension(self):
        with patch("algorithms.algo_alphazero.AZ_USE_RUST_MCTS", True):
            with patch("algorithms.algo_alphazero.algorithms_rs", None):
                with self.assertRaisesRegex(RuntimeError, "extension is not installed"):
                    algo_alphazero.run_mcts(
                        ZeroNet(), self.boards, 1, torch.device("cpu"), noise=False
                    )


if __name__ == "__main__":
    unittest.main()
