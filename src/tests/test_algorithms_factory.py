import unittest
from unittest.mock import Mock, patch

import chess
import torch

from algorithms.algorithms import make_agent


class AlgorithmFactoryTests(unittest.TestCase):
    def test_make_agent_accepts_internal_key_and_display_name(self):
        agent_class = Mock(return_value=object())
        device = torch.device("cpu")
        with patch("algorithms.algorithms._BY_KEY", {"alphazero": agent_class}), \
                patch("algorithms.algorithms.ALGORITHMS", {"AlphaZero": agent_class}):
            for name in ("alphazero", "AlphaZero"):
                with self.subTest(algorithm=name):
                    self.assertIsNotNone(make_agent(name, chess.WHITE, device))

        self.assertEqual(agent_class.call_count, 2)
        agent_class.assert_called_with(chess.WHITE, "White", device, 1)

    def test_make_agent_rejects_unknown_algorithm(self):
        with self.assertRaisesRegex(ValueError, "unknown algorithm"):
            make_agent("not-an-algorithm", chess.WHITE, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
