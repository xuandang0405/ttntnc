import unittest

import chess
import torch

from algorithms.algo_ppo import PPOAgent
from algorithms.algo_dqn import DQNAgent
from curriculum import new_start
from skill_eval import evaluate_skills
from tactical_pipeline import TacticalTrainer, evaluate_tactical_benchmark
from tactics_library import (
    ALL_MOTIFS,
    PATTERNS_BY_CATEGORY,
    VERIFIED_PATTERNS,
    instantiate_pattern,
)


class TacticalPipelineTests(unittest.TestCase):
    def test_verified_patterns_cover_attack_defense_endgame(self):
        self.assertGreaterEqual(len(PATTERNS_BY_CATEGORY["attack"]), 8)
        self.assertGreaterEqual(len(PATTERNS_BY_CATEGORY["defense"]), 5)
        self.assertGreaterEqual(len(PATTERNS_BY_CATEGORY["endgame"]), 6)
        self.assertIn("fork", ALL_MOTIFS)
        self.assertIn("pin", ALL_MOTIFS)
        self.assertIn("perpetual_check", ALL_MOTIFS)
        self.assertIn("lucena", ALL_MOTIFS)
        self.assertIn("philidor", ALL_MOTIFS)

    def test_all_patterns_have_legal_solutions_and_valid_mirror(self):
        for pat in VERIFIED_PATTERNS:
            board = chess.Board(pat.fen)
            self.assertTrue(board.is_valid(), pat.pattern_id)
            self.assertIsNone(board.outcome(claim_draw=True), pat.pattern_id)
            legal = {m.uci() for m in board.legal_moves}
            for uci in pat.best_moves:
                self.assertIn(uci, legal, f"{pat.pattern_id}: {uci}")

            b2, moves2, val2, phase2 = instantiate_pattern(pat, allow_mirror=True)
            self.assertTrue(b2.is_valid())
            self.assertEqual(val2, pat.target_value)
            self.assertEqual(phase2, pat.phase)
            self.assertTrue(all(m in b2.legal_moves for m in moves2))

    def test_tactical_trainer_step_and_checkpoint_roundtrip(self):
        agent = PPOAgent(chess.WHITE, "White", torch.device("cpu"))
        trainer = TacticalTrainer(agent)
        stats = trainer.train_batch(batch_size=8)
        self.assertIsNotNone(stats)
        self.assertIn("tactical_acc", stats)
        self.assertEqual(trainer.drill_steps, 1)

        saved = agent.strategy.state()
        agent2 = PPOAgent(chess.WHITE, "White", torch.device("cpu"))
        agent2.strategy.tactical.train_batch(batch_size=4)
        agent2.strategy.load(saved)
        self.assertEqual(agent2.strategy.tactical.drill_steps, 1)

    def test_dqn_tactical_step_runs_without_error(self):
        agent = DQNAgent(chess.BLACK, "Black", torch.device("cpu"))
        stats = agent.strategy.tactical.train_batch(batch_size=6)
        self.assertIsNotNone(stats)

    def test_benchmark_and_skill_eval_include_tactical_pillars(self):
        agent = PPOAgent(chess.WHITE, "White", torch.device("cpu"))
        bench = evaluate_tactical_benchmark(agent)
        self.assertIn("attack", bench)
        self.assertIn("defense", bench)
        self.assertIn("endgame", bench)
        self.assertIn("motifs", bench)

        skills = evaluate_skills(agent)
        self.assertIn("tactical_attack", skills)
        self.assertIn("tactical_defense", skills)
        self.assertIn("tactical_endgame", skills)

    def test_curriculum_generates_valid_starts(self):
        for kind in ("opening", "middlegame", "endgame"):
            for _ in range(10):
                k, board = new_start(kind=kind)
                self.assertEqual(k, kind)
                self.assertTrue(board.is_valid())
                self.assertIsNone(board.outcome(claim_draw=True))


if __name__ == "__main__":
    unittest.main()
