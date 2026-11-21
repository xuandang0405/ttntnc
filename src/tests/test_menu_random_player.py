import chess
import unittest

from menu_scene import MenuScene
from players import RandomPlayer


class MenuRandomPlayerTests(unittest.TestCase):
    def test_random_opponent_selection_starts_a_game(self):
        menu = MenuScene.__new__(MenuScene)
        menu.mode = 1
        menu.sel = {
            "Bạn chơi": 0,
            "Kiểu AI Encoding": 0,
            "Đoán nước đi": 0,
        }
        menu.paths = {}
        menu.fonts = None
        menu.error = ""

        menu.start()

        self.assertIsInstance(menu.next.match.black, RandomPlayer)
        move = menu.next.match.black.choose_move(chess.Board())
        self.assertIn(move, chess.Board().legal_moves)


if __name__ == "__main__":
    unittest.main()
