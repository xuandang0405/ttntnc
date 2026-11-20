import random
import unittest

import chess
import numpy as np

from encoding import legal_moves_map
from stockfish.stockfish_engine import Line
from stockfish.stockfish_teacher import (TeacherData, _Rows, _WORKER, _play_games, infer_game_ids,
                                         merge, position_keys)


def fake_label(board):
    idx = next(iter(legal_moves_map(board, board.turn)))
    return {"idx": [idx], "p": [1.0], "q": [0.0], "value": 0.0, "cp": 0}


def make_rows(games, plies, seed=0):
    """Random games from the start position; every game begins with the same position."""
    rng, rows = random.Random(seed), _Rows(4)
    for game in range(games):
        board = chess.Board()
        for _ in range(plies):
            if board.is_game_over():
                break
            rows.add(board, fake_label(board), game)
            board.push(rng.choice(list(board.legal_moves)))
    return rows.arrays()


class StubPool:
    def analyse(self, board, nodes, multipv):
        return [Line(next(iter(board.legal_moves)), 20, 0.04)]


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays = make_rows(games=40, plies=30)

    def test_no_game_is_on_both_sides(self):
        data = TeacherData(dict(self.arrays))
        train, test = data.split(0.25, seed=1)
        games = data.a["game"]
        self.assertFalse(set(games[train]) & set(games[test]))
        self.assertEqual(len(np.unique(games[test])), 10)                  # 25% of the 40 games
        self.assertEqual(data.game_source, "recorded")

    def test_old_datasets_without_game_ids_are_recovered_exactly(self):
        old = {k: v for k, v in self.arrays.items() if k != "game"}
        inferred = infer_game_ids(old["states"])
        recorded = self.arrays["game"]
        pairs = set(zip(recorded.tolist(), inferred.tolist()))
        self.assertEqual(len(pairs), len(np.unique(recorded)))             # one-to-one with the real games
        self.assertEqual(len(pairs), len(np.unique(inferred)))
        data = TeacherData(old)
        train, test = data.split(0.25, seed=1)
        self.assertEqual(data.game_source, "inferred")
        self.assertFalse(set(inferred[train]) & set(inferred[test]))

    def test_positions_seen_in_other_games_leave_the_training_set(self):
        data = TeacherData(dict(self.arrays))                              # all 40 games start from the same position
        train, test = data.split(0.25, seed=2)
        keys = data.position_keys
        self.assertFalse(set(keys[train]) & set(keys[test]))
        self.assertGreater(data.split_info["removed"], 0)
        self.assertLessEqual(len(train) + data.split_info["removed"] + len(test), data.n)

    def test_split_is_reproducible_and_depends_on_the_seed(self):
        data = TeacherData(dict(self.arrays))
        a1, b1 = data.split(0.25, seed=5)
        a2, b2 = data.split(0.25, seed=5)
        _, b3 = data.split(0.25, seed=6)
        np.testing.assert_array_equal(b1, b2)
        self.assertFalse(np.array_equal(b1, b3))

    def test_too_little_data_means_no_holdout(self):
        one_game = TeacherData(make_rows(games=1, plies=20))
        train, test = one_game.split(0.25)
        self.assertEqual((len(train), len(test)), (one_game.n, 0))
        train, test = TeacherData(dict(self.arrays)).split(0.0)
        self.assertEqual(len(test), 0)

    def test_position_keys_tell_positions_apart(self):
        keys = position_keys(self.arrays["states"])
        again = position_keys(self.arrays["states"])
        np.testing.assert_array_equal(keys, again)
        first_rows = np.flatnonzero(np.diff(self.arrays["game"], prepend=-1))
        self.assertEqual(len(set(keys[first_rows])), 1)                    # same start position -> same key
        self.assertGreater(len(set(keys)), 100)


class GenerationTests(unittest.TestCase):
    def test_workers_number_their_games_without_overlap(self):
        _WORKER["pool"] = StubPool()
        try:
            first = _play_games((1, 3, 100, 4, 8, 10))                    # games 10, 11, 12
            second = _play_games((2, 2, 100, 4, 8, 13))                   # games 13, 14
        finally:
            _WORKER.clear()
        ids_a, ids_b = set(first["game"].tolist()), set(second["game"].tolist())
        self.assertTrue(ids_a <= {10, 11, 12} and ids_b <= {13, 14})
        self.assertTrue(ids_a and ids_b and not ids_a & ids_b)
        merged = merge([first, second])
        self.assertIn("game", merged)
        self.assertEqual(len(merged["game"]), len(merged["value"]))


if __name__ == "__main__":
    unittest.main()
