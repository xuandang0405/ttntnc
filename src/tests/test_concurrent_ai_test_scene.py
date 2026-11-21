import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chess
import pygame
import torch

from algorithms.algo_base import BaseAgent
from concurrent_mode.concurrent_config import NUM_GAMES
from concurrent_mode import concurrent_renderer
from config import BOARD_PX
from encoding import ACTION_SIZE
from match import Match
from play_scene import PlayScene


class ConcurrentAITestSceneTests(unittest.TestCase):
    def test_model_moves_are_inferred_as_a_batch(self):
        class CountingNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []
                self.logits = torch.nn.Parameter(torch.zeros(ACTION_SIZE))

            def forward(self, states):
                self.batch_sizes.append(len(states))
                return self.logits.expand(len(states), -1), torch.zeros(len(states))

        agent = BaseAgent.__new__(BaseAgent)
        agent.net = CountingNet()
        agent.device = torch.device("cpu")
        agent.color = chess.WHITE
        agent.play_temperature = 1.0
        boards = [chess.Board(), chess.Board()]

        moves = agent.select_moves(boards, greedy=True)

        self.assertEqual(agent.net.batch_sizes, [2])
        self.assertTrue(all(move in board.legal_moves for move, board in zip(moves, boards)))
        self.assertTrue(agent.net.training)

    def test_scene_batches_moves_by_player(self):
        from concurrent_mode import concurrent_test_scene as concurrent_scene

        class FakePlayer:
            is_human = False

            def __init__(self, name):
                self.name = name
                self.batch_sizes = []

            def choose_move(self, board):
                return next(iter(board.legal_moves))

            def choose_moves(self, boards):
                self.batch_sizes.append(len(boards))
                return [self.choose_move(board) for board in boards]

        white, black = FakePlayer("White"), FakePlayer("Black")
        recorder = SimpleNamespace(games=[], stats={"white": {"wins": 0}, "black": {"wins": 0}})
        with patch.object(concurrent_scene, "NUM_GAMES", 2), \
                patch.object(concurrent_scene, "GAMES_PER_UPDATE", 2), \
                patch.object(concurrent_scene, "make_player", side_effect=[white, black]), \
                patch.object(concurrent_scene, "GameResultRecorder", return_value=recorder):
            test_scene = concurrent_scene.ConcurrentAITestScene(
                None, "white spec", "black spec")
            test_scene.update()
            test_scene.update()

        self.assertEqual(white.batch_sizes, [2])
        self.assertEqual(black.batch_sizes, [2])

    def test_grid_reuses_board_surface_until_a_move_changes_it(self):
        match = Match(None, None)
        screen = pygame.Surface((BOARD_PX, BOARD_PX))
        with patch.object(concurrent_renderer, "draw_small_board") as draw_board:
            concurrent_renderer.draw_games_grid(screen, {}, [match])
            concurrent_renderer.draw_games_grid(screen, {}, [match])
            self.assertEqual(draw_board.call_count, 1)

            match.push(next(iter(match.board.legal_moves)))
            concurrent_renderer.draw_games_grid(screen, {}, [match])
            self.assertEqual(draw_board.call_count, 2)

    def test_ai_vs_ai_r_opens_mode_popup_and_multiple_starts_concurrent_scene(self):
        scene = PlayScene(None, "Random mover", "Random mover", False)
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        self.assertIsNotNone(scene.recording_popup)
        multiple = scene.recording_popup.multiple
        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=multiple.rect.center)
        with patch("concurrent_mode.concurrent_test_scene.ConcurrentAITestScene") as test_scene:
            scene.handle_event(event)

        test_scene.assert_called_once_with(None, "Random mover", "Random mover", False, False)
        self.assertIsNone(scene.recording_popup)

    def test_multiple_scene_uses_configured_game_count(self):
        from concurrent_mode import concurrent_test_scene as concurrent_scene

        class FakePlayer:
            is_human = False

            def __init__(self, name):
                self.name = name

        with patch.object(concurrent_scene, "make_player",
                          side_effect=[FakePlayer("White"), FakePlayer("Black")]), \
                patch.object(concurrent_scene, "GameResultRecorder"):
            test_scene = concurrent_scene.ConcurrentAITestScene(
                None, "white spec", "black spec")

        self.assertEqual(len(test_scene.matches), NUM_GAMES)

    def test_single_game_choice_starts_existing_recording_flow(self):
        scene = PlayScene(None, "Random mover", "Random mover", False)
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        single = scene.recording_popup.single
        event = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, button=1, pos=single.rect.center)
        with patch("play_scene.GameResultRecorder") as recorder:
            scene.handle_event(event)

        recorder.assert_called_once_with("ai_vs_ai", "Random mover", "Random mover")
        self.assertTrue(scene.recording)
        self.assertIsNone(scene.recording_popup)

    def test_human_vs_ai_r_keeps_existing_recording_behavior(self):
        scene = PlayScene(None, "Human", "Random mover", False)
        with patch("play_scene.GameResultRecorder"):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        self.assertTrue(scene.recording)
        self.assertIsNone(scene.recording_popup)

    def test_multiple_scene_uses_configured_game_count_and_records_results(self):
        from concurrent_mode import concurrent_test_scene as concurrent_scene

        class FakePlayer:
            is_human = False

            def __init__(self, name):
                self.name = name

            def choose_move(self, board):
                sequence = ("f2f3", "e7e5", "g2g4", "d8h4")
                return chess.Move.from_uci(sequence[len(board.move_stack)])

        class FakeRecorder:
            def __init__(self):
                self.games = []
                self.path = "test-results.json"

            def record_result(self, result):
                self.games.append(result)

            def save(self):
                pass

        recorder = FakeRecorder()
        with patch.object(concurrent_scene, "NUM_GAMES", 1), \
                patch.object(concurrent_scene, "GAMES_PER_UPDATE", 8), \
                patch.object(concurrent_scene, "make_player",
                             side_effect=[FakePlayer("White"), FakePlayer("Black")]), \
                patch.object(concurrent_scene, "GameResultRecorder", return_value=recorder):
            test_scene = concurrent_scene.ConcurrentAITestScene(
                None, "white spec", "black spec")
            test_scene.update()

        self.assertEqual(len(test_scene.matches), 1)
        self.assertEqual(test_scene.completed_games, 1)
        self.assertTrue(test_scene.matches[0].board.is_checkmate())
        self.assertEqual(test_scene.recorder.games[0].winner, chess.BLACK)


if __name__ == "__main__":
    unittest.main()
