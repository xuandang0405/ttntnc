import chess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pygame

from menu_scene import MenuScene
from model_presets import find_model_presets
from players import RandomPlayer


class MenuRandomPlayerTests(unittest.TestCase):
    def test_random_opponent_selection_starts_a_game(self):
        menu = MenuScene(None)

        menu.start()

        self.assertIsInstance(menu.next.match.black, RandomPlayer)
        move = menu.next.match.black.choose_move(chess.Board())
        self.assertIn(move, chess.Board().legal_moves)

    def test_model_picker_keeps_the_run_when_human_switches_color(self):
        with TemporaryDirectory() as work:
            directory = Path(work) / "exp_20261126_000717"
            directory.mkdir()
            for color in ("white", "black"):
                (directory / f"{color}_dqn.pt").touch()
            presets = find_model_presets(Path(work))

            menu = MenuScene(None)
            opponent_button = next(button for _, name, button in menu.row_buttons() if name == "Đối thủ")
            with patch("menu_scene.find_model_presets", return_value=presets):
                menu.handle_event(pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1, pos=opponent_button.rect.center))
            preset, button = menu.model_picker.rows()[0]
            menu.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=button.rect.center))

            self.assertEqual(menu.model_spec("Đối thủ"), f"file:{preset.checkpoint(False)}")
            menu.sel["Bạn chơi"] = 1
            self.assertEqual(menu.model_spec("Đối thủ"), f"file:{preset.checkpoint(True)}")


if __name__ == "__main__":
    unittest.main()
