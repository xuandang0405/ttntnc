import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chess
import pygame

from match import MatchResult
from plot_logging.game_record import GameResultRecorder
from plot_logging.record_plot import save_record_plot
from play_scene import PlayScene


class GameResultRecorderTests(unittest.TestCase):
    def test_records_player_win_loss_draw_totals_and_autosaves(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "results.json")
            recorder = GameResultRecorder(
                "ai_vs_ai", "White agent", "Black agent", path=path, interval=0)

            recorder.record_result(SimpleNamespace(winner=True, reason="Checkmate"))
            recorder.record_result(SimpleNamespace(winner=None, reason="Stalemate"))
            recorder.record_result(SimpleNamespace(winner=False, reason="Checkmate"))
            self.assertTrue(recorder.save_if_due())

            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["totals"]["games"], 3)
            self.assertEqual(data["totals"]["white"], {"wins": 1, "losses": 1, "draws": 1})
            self.assertEqual(data["totals"]["black"], {"wins": 1, "losses": 1, "draws": 1})
            self.assertEqual(data["games"][0]["outcome"], "white_win")
            self.assertEqual(data["games"][1]["outcome"], "draw")
            self.assertEqual(data["games"][2]["outcome"], "black_win")
            self.assertEqual(data["games"][0]["elo"], {"white": 1216.0, "black": 1184.0})
            self.assertEqual(data["totals"]["elo"], data["games"][-1]["elo"])
            self.assertAlmostEqual(sum(data["totals"]["elo"].values()), 2400.0)

            plot_path = save_record_plot(path)
            self.assertTrue(os.path.isfile(plot_path))

    def test_recording_controls_capture_finished_game_once_and_stop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "results.json")
            recorder = GameResultRecorder("human_vs_ai", "Human", "Random mover", path=path)
            scene = PlayScene(None, "Human", "Random mover", False)
            scene.match.result = MatchResult(chess.WHITE, "Checkmate")

            with patch("play_scene.GameResultRecorder", return_value=recorder):
                scene.handle_recording_key(pygame.K_r)
                self.assertEqual(len(recorder.games), 1)
                scene.handle_recording_key(pygame.K_x)
                self.assertFalse(scene.recording)
                scene.handle_recording_key(pygame.K_r)

            self.assertEqual(len(recorder.games), 1)
            self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
