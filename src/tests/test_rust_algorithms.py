import unittest

import chess
import numpy as np
import torch
from unittest.mock import patch

try:
    import algorithms_rs
except ImportError:
    algorithms_rs = None

from encoding import ACTION_SIZE, INPUT_CHANNELS, encode, legal_moves_map
from config import GAMMA, SHAPING
from algorithms.algo_a2c import A2CAgent
from algorithms.algo_ppo import PPOAgent
from algorithms.algo_dqn import DQNAgent
from algorithms.algo_alphazero import AlphaZeroAgent, _rust_position_fens
from algorithms.algo_config import GAE_LAMBDA
from algorithms.optimization import BoundedCache


@unittest.skipIf(algorithms_rs is None, "Rust extension is not installed")
class RustAlgorithmTests(unittest.TestCase):
    def test_return_and_advantage_calculations(self):
        self.assertEqual(
            algorithms_rs.discounted_returns([[1.0, 2.0]], 0.5),
            [[2.0, 2.0]],
        )
        advantages, returns = algorithms_rs.generalized_advantage_estimation(
            [[1.0, 2.0]], [[0.5, 0.25]], 0.9, 0.8
        )
        np.testing.assert_allclose(advantages[0], [1.985, 1.75], rtol=1e-6)
        np.testing.assert_allclose(returns[0], [2.485, 2.0], rtol=1e-6)

    def test_reward_shaping_excludes_terminal_reward(self):
        np.testing.assert_allclose(
            algorithms_rs.shaped_rewards([1.0, 4.0], 3.0, 0.1),
            [0.3, -0.1],
        )

    def test_prioritized_sample_is_unique_and_normalized(self):
        indices, weights = algorithms_rs.prioritized_sample(
            [1.0, 2.0, 3.0, 4.0], 0.6, 0.4, 3, 1e-5
        )
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(max(weights), 1.0)

    def test_a2c_and_ppo_episode_calculations_use_rust_results(self):
        state = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.float16)
        episode = {
            "s": [state, state],
            "a": [0, 1],
            "legal": [np.asarray([0]), np.asarray([1])],
            "bal": [1.0, 2.0],
        }

        a2c = A2CAgent.__new__(A2CAgent)
        a2c.episodes_per_update = 2
        a2c.eps = {0: dict(episode)}
        a2c.buffer = []
        a2c.metrics = None
        a2c.finish_episode(0, 3.0, {"win": 1.0})
        rewards = [SHAPING, SHAPING + 1.0]
        np.testing.assert_allclose(
            a2c.buffer[0][3],
            [rewards[0] + GAMMA * rewards[1], rewards[1]],
        )

        ppo = PPOAgent.__new__(PPOAgent)
        ppo.episodes_per_update = 2
        ppo.eps = {0: {**episode, "v": [0.5, 0.25], "logp": [-0.1, -0.2]}}
        ppo.buffer = []
        ppo.metrics = None
        ppo.finish_episode(0, 3.0, {"win": 1.0})
        adv_last = rewards[1] - 0.25
        adv_first = rewards[0] + GAMMA * 0.25 - 0.5 + GAMMA * GAE_LAMBDA * adv_last
        np.testing.assert_allclose(ppo.buffer[0][3], [adv_first, adv_last], rtol=1e-6)
        np.testing.assert_allclose(ppo.buffer[0][4], [adv_first + 0.5, adv_last + 0.25], rtol=1e-6)

    def test_dqn_update_uses_native_prioritized_sampling(self):
        class TinyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Parameter(torch.zeros(ACTION_SIZE))

            def forward(self, states):
                return self.q.expand(len(states), -1), torch.zeros(len(states), device=states.device)

        agent = DQNAgent.__new__(DQNAgent)
        agent.device = torch.device("cpu")
        agent.net = TinyNet()
        agent.target = TinyNet()
        agent.target.load_state_dict(agent.net.state_dict())
        agent.opt = torch.optim.Adam(agent.net.parameters(), lr=1e-3)
        agent.replay_pos = 0
        state = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.float16)
        agent.replay = [
            (state, 0, 1.0, state, np.asarray([1]), False, np.asarray([0])),
            (state, 1, -1.0, state, np.asarray([], dtype=np.int64), True, np.asarray([1])),
        ]
        agent.priorities = np.ones(2, dtype=np.float32)
        agent.max_priority = 1.0
        agent.n_updates = 0
        agent.name = "White"
        agent.metrics = None
        agent.last_loss = 0.0
        agent.update()
        self.assertEqual(agent.n_updates, 1)
        self.assertTrue(np.all(np.isfinite(agent.priorities)))

    def test_native_state_encoding_and_legal_actions_match_python(self):
        boards = [chess.Board()]
        moved = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3"):
            moved.push_uci(uci)
        boards.append(moved)
        en_passant = chess.Board()
        en_passant.push_uci("e2e4")
        boards.append(en_passant)
        boards.append(chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"))

        for board in boards:
            tree = algorithms_rs.MctsTree(_rust_position_fens(board))
            leaf = algorithms_rs.mcts_select_batch([tree], 1.5)[0]
            state = np.asarray(leaf[-1]).reshape(encode(board, board.turn).shape[1:])
            np.testing.assert_array_equal(state, encode(board, board.turn)[0].numpy())
            self.assertEqual(set(leaf[3]), set(legal_moves_map(board, board.turn)))
            self.assertEqual(
                {chess.Move.from_uci(move) for move in leaf[4]},
                set(legal_moves_map(board, board.turn).values()),
            )

    def test_native_search_and_repetition_terminal(self):
        from algorithms.algo_alphazero import run_mcts

        class ZeroNet(torch.nn.Module):
            def forward(self, states):
                return (
                    torch.zeros((len(states), ACTION_SIZE), device=states.device),
                    torch.zeros(len(states), device=states.device),
                )

        board = chess.Board()
        result = run_mcts(ZeroNet(), [board], 4, torch.device("cpu"), noise=True)
        self.assertEqual(set(result[0][0]), set(legal_moves_map(board, board.turn)))

        repeated = chess.Board()
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
        for uci in cycle * 4:
            repeated.push_uci(uci)
        result = run_mcts(ZeroNet(), [repeated], 4, torch.device("cpu"), noise=False)
        self.assertEqual(result[0][0], [])

        claimable_repetition = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"):
            claimable_repetition.push_uci(uci)
        self.assertIsNotNone(claimable_repetition.outcome(claim_draw=True))
        result = run_mcts(
            ZeroNet(), [claimable_repetition], 4, torch.device("cpu"), noise=False
        )
        self.assertEqual(result[0][0], [])

        fifty_move_claim = chess.Board("4k2r/8/8/8/8/8/8/4K2R w - - 99 50")
        self.assertIsNotNone(fifty_move_claim.outcome(claim_draw=True))
        result = run_mcts(
            ZeroNet(), [fifty_move_claim], 4, torch.device("cpu"), noise=False
        )
        self.assertEqual(result[0][0], [])

    def test_alphazero_agent_runs_and_reuses_rust_tree(self):
        class ZeroNet(torch.nn.Module):
            def forward(self, states):
                return (
                    torch.zeros((len(states), ACTION_SIZE), device=states.device),
                    torch.zeros(len(states), device=states.device),
                )

        agent = AlphaZeroAgent.__new__(AlphaZeroAgent)
        agent.color = chess.WHITE
        agent.name = "White"
        agent.device = torch.device("cpu")
        agent.net = ZeroNet()
        agent.eps = {}
        agent._pending_tree = {}
        agent.eval_cache = BoundedCache(64)
        board = chess.Board()

        with patch("algorithms.algo_alphazero.AZ_SIMULATIONS", 2):
            move = agent.act_batch([7], [board])[0]
            self.assertIn(move, board.legal_moves)
            board.push(move)
            board.push(next(iter(board.legal_moves)))
            response = agent.act_batch([7], [board])[0]

        self.assertIn(response, board.legal_moves)


if __name__ == "__main__":
    unittest.main()
