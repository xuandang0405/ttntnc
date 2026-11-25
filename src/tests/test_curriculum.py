import random
import unittest

import chess

from curriculum import new_start


class CurriculumTests(unittest.TestCase):
    def test_match_number_alternates_starting_color_for_all_position_kinds(self):
        for kind in ("opening", "middlegame", "endgame"):
            for match_number, expected_color in ((1, chess.WHITE), (2, chess.BLACK)):
                with self.subTest(kind=kind, match_number=match_number):
                    _, board = new_start(
                        kind=kind, rng=random.Random(match_number),
                        match_number=match_number)
                    self.assertEqual(board.turn, expected_color)
                    self.assertTrue(board.is_valid())
                    self.assertIsNone(board.outcome(claim_draw=True))


if __name__ == "__main__":
    unittest.main()
