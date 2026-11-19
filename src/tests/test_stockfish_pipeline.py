"""Tests for the Stockfish teacher and coach. Skipped when no Stockfish executable is found."""
import os
import random
import tempfile
import unittest

import chess
import numpy as np
import torch

from algorithms.algorithms import load_agent, make_agent
from config import SHAPING
from encoding import legal_moves_map, move_to_index
from stockfish_coach import StockfishCoach
from stockfish_engine import cp_to_value, find_stockfish, StockfishPool
from stockfish_teacher import TeacherData, _Rows, evaluate, label_position, merge, pretrain

try:
    find_stockfish()
    HAVE_STOCKFISH = True
except FileNotFoundError:
    HAVE_STOCKFISH = False

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"       # Ra8#
CPU = torch.device("cpu")


def labelled_positions(pool, count, seed=0):
    rng, rows = random.Random(seed), _Rows(4)
    while len(rows) < count:
        board = chess.Board()
        for _ in range(rng.randint(2, 40)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            rows.add(board, label_position(board, pool, 500, 4))
    return TeacherData(merge([rows.arrays()]))


class ValueCurveTests(unittest.TestCase):
    def test_cp_to_value_is_monotone_and_bounded(self):
        values = [cp_to_value(c) for c in (-10000, -300, 0, 300, 10000)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(cp_to_value(0), 0.0)
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in values))


@unittest.skipUnless(HAVE_STOCKFISH, "Stockfish executable not found (set STOCKFISH_PATH)")
class StockfishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = StockfishPool(engines=2)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def test_labels_are_legal_and_normalised(self):
        board = chess.Board()
        label = label_position(board, self.pool, 2000, 4)
        legal = legal_moves_map(board, board.turn)
        self.assertEqual(len(label["idx"]), 4)
        self.assertTrue(all(i in legal for i in label["idx"]))
        self.assertAlmostEqual(sum(label["p"]), 1.0, places=6)
        self.assertEqual(label["p"], sorted(label["p"], reverse=True))      # best line first

    def test_mate_in_one_is_the_top_label(self):
        board = chess.Board(MATE_IN_ONE)
        label = label_position(board, self.pool, 2000, 3)
        self.assertEqual(label["idx"][0], move_to_index(chess.Move.from_uci("a1a8"), chess.WHITE))
        self.assertGreater(label["value"], 0.99)
        self.assertGreater(label["p"][0], 0.95)

    def test_labels_are_from_the_side_to_move(self):
        # Same position with colours swapped must give the same evaluation for whoever moves.
        white = label_position(chess.Board(MATE_IN_ONE), self.pool, 1000, 1)
        black = label_position(chess.Board(chess.Board(MATE_IN_ONE).mirror().fen()), self.pool, 1000, 1)
        self.assertGreater(white["value"], 0.99)
        self.assertGreater(black["value"], 0.99)

    def test_pretraining_learns_for_every_algorithm(self):
        data = labelled_positions(self.pool, 96)
        for name in ("A2C", "PPO", "DQN", "AlphaZero"):
            with self.subTest(algorithm=name):
                torch.manual_seed(0)
                agent = make_agent(name, chess.WHITE, CPU)
                history = pretrain(agent, data, epochs=4, batch_size=32, lr=1e-3, holdout=0.25, log=lambda *_: None)
                self.assertEqual(len(history), 4)
                self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
                self.assertIn("top1", history[-1])
                if name == "DQN":
                    for a, b in zip(agent.net.parameters(), agent.target.parameters()):
                        self.assertTrue(torch.equal(a, b))                  # target network synced
                else:
                    self.assertIn("value_mae", history[-1])
                self.assertGreater(agent.strategy.train_steps, 0)

    def test_pretrained_checkpoint_loads_back(self):
        data = labelled_positions(self.pool, 40, seed=3)
        agent = make_agent("A2C", chess.WHITE, CPU)
        pretrain(agent, data, epochs=1, batch_size=32, holdout=0.2, log=lambda *_: None)
        with tempfile.TemporaryDirectory() as folder:
            path = agent.save(folder)
            restored = load_agent(path, chess.BLACK, CPU)
        for a, b in zip(agent.net.state_dict().values(), restored.net.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_evaluate_reports_agreement(self):
        data = labelled_positions(self.pool, 32, seed=4)
        agent = make_agent("A2C", chess.WHITE, CPU)
        metrics = evaluate(agent, data, np.arange(data.n))
        self.assertTrue(0.0 <= metrics["top1"] <= metrics["top3"] <= 1.0)


@unittest.skipUnless(HAVE_STOCKFISH, "Stockfish executable not found (set STOCKFISH_PATH)")
class CoachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = StockfishPool(engines=2)

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def test_potential_is_from_the_requested_colour_and_clipped(self):
        coach = StockfishCoach(self.pool, nodes=1000, clip=10.0)
        board = chess.Board(MATE_IN_ONE)
        self.assertEqual(coach.potential(board, chess.WHITE), 10.0)
        self.assertEqual(coach.potential(board, chess.BLACK), -10.0)

    def test_prefetch_fills_the_cache(self):
        coach = StockfishCoach(self.pool, nodes=1000)
        boards = [chess.Board(), chess.Board(MATE_IN_ONE)]
        coach.prefetch(boards)
        first = [coach.potential(b, b.turn) for b in boards]
        self.assertEqual(coach.misses, 2)
        again = [coach.potential(b, b.turn) for b in boards]
        self.assertEqual((coach.hits, first), (2, again))

    def test_agent_rewards_follow_the_evaluation(self):
        agent = make_agent("A2C", chess.WHITE, CPU)
        board = chess.Board(MATE_IN_ONE)
        self.assertEqual(agent.balance_of(board), 5)                           # material mode: an extra rook
        agent.coach = StockfishCoach(self.pool, nodes=1000)
        self.assertEqual(agent.balance_of(board), 10.0)                        # Stockfish mode
        # losing the advantage is penalised, and a finished game ends at potential 0
        rewards = agent._rewards([4.0, 1.0], final_balance=99, terminal={})
        self.assertAlmostEqual(rewards[0], SHAPING * (1.0 - 4.0))
        self.assertAlmostEqual(rewards[1], SHAPING * (0.0 - 1.0))
        agent.coach = None
        self.assertAlmostEqual(agent._rewards([4.0, 1.0], 99, {})[1], SHAPING * (99 - 1.0))

    def test_every_algorithm_collects_coach_potentials_during_play(self):
        coach = StockfishCoach(self.pool, nodes=500)
        for name in ("A2C", "PPO", "DQN", "AlphaZero"):
            with self.subTest(algorithm=name):
                agent = make_agent(name, chess.WHITE, CPU)
                agent.coach = coach
                board = chess.Board(MATE_IN_ONE)
                if name == "AlphaZero":
                    from unittest.mock import patch
                    with patch("algorithms.algo_alphazero.AZ_SIMULATIONS", 4):
                        agent.act_batch([0], [board])
                else:
                    agent.act_batch([0], [board])
                self.assertEqual(agent.eps[0]["bal"], [10.0])


if __name__ == "__main__":
    unittest.main()
