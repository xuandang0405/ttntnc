"""Parity and behaviour tests for the native Rust search (algorithms_rs.BatchSearch)."""
import random
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
from encoding import ACTION_SIZE, INPUT_CHANNELS, encode, legal_moves_map

HAS_SEARCH = algorithms_rs is not None and hasattr(algorithms_rs, "BatchSearch")


class DeterministicNet(torch.nn.Module):
    """Priors depend only on the action index and values on the piece planes, so
    there are no tied PUCT scores and move-generation order cannot matter."""

    def forward(self, states):
        n = len(states)
        action = torch.arange(ACTION_SIZE, dtype=torch.float32, device=states.device)
        logits = (torch.sin(action * 0.7 + 1.3) * 2.0).expand(n, ACTION_SIZE).contiguous()
        own = states[:, :12].sum(dim=(1, 2, 3))
        values = torch.tanh(0.01 * (own - states[:, 12:24].sum(dim=(1, 2, 3))) + 0.003 * states[:, 24:36].sum(dim=(1, 2, 3)))
        return logits, values


def random_boards(count, seed):
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        b = chess.Board()
        for _ in range(rng.randint(0, 120)):
            if b.outcome(claim_draw=True) is not None:
                break
            b.push(rng.choice(list(b.legal_moves)))
        if b.outcome(claim_draw=True) is None:
            boards.append(b)
    return boards


class NativeEncoderCompatibilityTests(unittest.TestCase):
    def test_rejects_stale_native_encoder_channel_count(self):
        states = np.zeros((1, 44, 8, 8), dtype=np.float32)
        legal = np.zeros((1, 1), dtype=np.int32)

        with self.assertRaisesRegex(RuntimeError, f"expected \\(N, {INPUT_CHANNELS}, 8, 8\\)"):
            algo_alphazero._evaluate_encoded(
                None, states, legal, torch.device("cpu"), {}
            )


@unittest.skipIf(not HAS_SEARCH, "Rust BatchSearch extension is not installed")
class NativeSearchTests(unittest.TestCase):
    def test_encoding_legal_moves_and_terminals_match_python_chess(self):
        rng = random.Random(5)
        for _ in range(60):
            b = chess.Board()
            for _ in range(rng.randint(0, 200)):
                if b.outcome(claim_draw=False) is not None:
                    break
                b.push(rng.choice(list(b.legal_moves)))
                fen, ucis = b.root().fen(), [m.uci() for m in b.move_stack]
                planes, actions, _ = algorithms_rs.encode_position(fen, ucis)
                np.testing.assert_array_equal(planes, encode(b, b.turn)[0].numpy())
                self.assertEqual(sorted(actions), sorted(legal_moves_map(b, b.turn)))
                out = b.outcome(claim_draw=True)
                expected = None if out is None else (0.0 if out.winner is None else -1.0)
                self.assertEqual(algorithms_rs.terminal_status(fen, ucis), expected)

    def test_visit_counts_match_python_mcts(self):
        net = DeterministicNet().eval()
        boards = random_boards(10, seed=11)
        cpu = torch.device("cpu")
        with patch.object(algo_alphazero, "AZ_USE_RUST_SEARCH", False), \
             patch.object(algo_alphazero, "AZ_USE_RUST_MCTS", False):
            expected = algo_alphazero.run_mcts(net, boards, 48, cpu, noise=False)
        with patch.object(algo_alphazero, "AZ_USE_RUST_SEARCH", True):
            actual = algo_alphazero.run_mcts(net, boards, 48, cpu, noise=False)
        for board, (e_idx, e_visits, e_map), (a_idx, a_visits, a_map) in zip(boards, expected, actual):
            self.assertEqual(dict(zip(e_idx, e_visits)), dict(zip(a_idx, a_visits)), board.fen())
            self.assertEqual({i: m.uci() for i, m in e_map.items()}, {i: m.uci() for i, m in a_map.items()})

    def test_finds_forced_mate_in_one(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")     # Ra8#
        with patch.object(algo_alphazero, "AZ_USE_RUST_SEARCH", True):
            idxs, visits, moves = algo_alphazero.run_mcts(
                DeterministicNet().eval(), [board], 32, torch.device("cpu"), noise=False)[0]
        best = idxs[int(np.argmax(visits))]
        self.assertEqual(moves[best].uci(), "a1a8")

    def test_terminal_roots_return_empty_and_visits_sum_to_sims(self):
        mated = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")           # stalemate
        live = chess.Board()
        with patch.object(algo_alphazero, "AZ_USE_RUST_SEARCH", True):
            results = algo_alphazero.run_mcts(
                DeterministicNet().eval(), [mated, live], 20, torch.device("cpu"), noise=True)
        self.assertEqual(results[0][0], [])
        self.assertEqual(int(results[1][1].sum()), 20)
        self.assertEqual(sorted(results[1][0]), sorted(legal_moves_map(live, live.turn)))

    def test_protocol_errors(self):
        search = algorithms_rs.BatchSearch([chess.STARTING_FEN], [[]], 2, 1.5)
        with self.assertRaises(RuntimeError):
            search.submit(np.zeros((1, 1), np.float32), np.zeros(1, np.float32))
        search.next_batch()
        with self.assertRaises(RuntimeError):
            search.next_batch()


if __name__ == "__main__":
    unittest.main()
