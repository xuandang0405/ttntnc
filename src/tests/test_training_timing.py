import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import chess
import numpy as np
import torch

from algorithms.algo_a2c import A2CAgent
from algorithms.algo_alphazero import AlphaZeroAgent, run_mcts
from algorithms.algo_dqn import DQNAgent
from algorithms.algo_ppo import PPOAgent
from config import LOG_INTERVAL_SECONDS
from encoding import ACTION_SIZE, INPUT_CHANNELS
from metrics import TrainingMetrics
from plot_logging.training_log import TrainingLog


class TrainingTimingTests(unittest.TestCase):
    class TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(ACTION_SIZE))
            self.value = torch.nn.Parameter(torch.zeros(1))

        def forward(self, states):
            return (
                self.logits.expand(len(states), -1),
                self.value.expand(len(states)),
            )

    def make_agent(self, agent_type):
        agent = agent_type.__new__(agent_type)
        agent.name = "White"
        agent.color = chess.WHITE
        agent.device = torch.device("cpu")
        agent.metrics = TrainingMetrics()
        agent.net = self.TinyNet()
        agent.opt = torch.optim.SGD(agent.net.parameters(), lr=1e-3)
        agent.last_loss = 0.0
        return agent

    def test_a2c_ppo_and_dqn_updates_record_stage_timings(self):
        state = np.zeros((1, INPUT_CHANNELS, 8, 8), dtype=np.float16)
        legal = [np.array([0], dtype=np.int64)]

        acting = self.make_agent(A2CAgent)
        acting.eps = {}
        board = chess.Board()
        self.assertIn(acting.act_batch([0], [board])[0], board.legal_moves)
        self.assertTrue({
            "action_prepare", "action_inference", "action_select"
        }.issubset(acting.metrics.timings["White"]))

        a2c = self.make_agent(A2CAgent)
        a2c.buffer = [(state, np.array([0]), legal, np.array([0.5], dtype=np.float32))]
        a2c.update()

        ppo = self.make_agent(PPOAgent)
        ppo.buffer = [(
            state, np.array([0]), np.array([-0.5], dtype=np.float32),
            np.array([0.2], dtype=np.float32), np.array([0.5], dtype=np.float32), legal,
        )]
        ppo.update()

        dqn = self.make_agent(DQNAgent)
        dqn.target = self.TinyNet()
        dqn.replay = [
            (state[0], 0, 1.0, state[0], np.array([], dtype=np.int64), True, legal[0])
        ]
        dqn.n_updates = 0
        dqn.update()

        expected = {
            "batch_prepare", "train_forward", "loss_compute", "backward", "optimizer_step"
        }
        for agent in (a2c, ppo, dqn):
            with self.subTest(agent=agent.algorithm):
                self.assertTrue(expected.issubset(agent.metrics.timings["White"]))
        self.assertIn("replay_sample", dqn.metrics.timings["White"])
        self.assertIn("target_forward", dqn.metrics.timings["White"])

        alphazero = self.make_agent(AlphaZeroAgent)
        alphazero.replay = [(state[0], legal[0], np.array([1.0], dtype=np.float32), 1.0)]
        with patch("algorithms.algo_alphazero.AZ_MIN_SAMPLES", 1), \
                patch("algorithms.algo_alphazero.AZ_TRAIN_STEPS", 1), \
                patch("algorithms.algo_alphazero.AZ_BATCH", 1):
            alphazero.update()
        self.assertTrue(expected.union({"replay_sample"}).issubset(
            alphazero.metrics.timings["White"]
        ))

    def test_timing_summary_is_written_to_training_log(self):
        metrics = TrainingMetrics()
        metrics.record_timing("White", "action_prepare", 0.001)
        metrics.record_timing("White", "action_prepare", 0.003)
        agent = lambda name: SimpleNamespace(name=name, net=torch.nn.Linear(1, 1))
        game = SimpleNamespace(
            metrics=metrics,
            white=agent("White"),
            black=agent("Black"),
            recent=[],
            recent_len=[],
            games=0,
        )
        log = TrainingLog(time.time(), interval=LOG_INTERVAL_SECONDS)

        log.maybe_record(game, force=True)

        record = log.records[0]
        self.assertEqual(record["white_action_prepare_mean_ms"], 2.0)
        self.assertEqual(record["white_action_prepare_p95_ms"], 3.0)
        self.assertEqual(record["timing_profile"]["White"]["action_prepare"]["calls"], 2)

    def test_mcts_reports_stage_durations(self):
        class ZeroNet(torch.nn.Module):
            def forward(self, states):
                return (
                    torch.zeros((len(states), ACTION_SIZE), device=states.device),
                    torch.zeros(len(states), device=states.device),
                )

        profile = {}
        result = run_mcts(
            ZeroNet(), [chess.Board()], 1, torch.device("cpu"), noise=False,
            profile=lambda part, elapsed: profile.__setitem__(part, elapsed),
        )

        self.assertTrue(result[0][0])
        self.assertTrue({
            "mcts_select_traverse",
            "mcts_position_encode",
            "mcts_network_inference",
            "mcts_expand",
            "mcts_backup",
        }.issubset(profile))
        self.assertTrue(all(elapsed >= 0.0 for elapsed in profile.values()))


if __name__ == "__main__":
    unittest.main()
