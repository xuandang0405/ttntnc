from collections import deque
import chess

from concurrent_mode.concurrent_agent import ConcurrentAgent, device
from concurrent_mode.concurrent_config import NUM_GAMES
from config import DRAW_REWARD, MAX_PLIES
from encoding import material_balance


class Slot:
    def __init__(self):
        self.board = chess.Board()
        self.last_move = None


class ConcurrentSelfPlay:
    def __init__(self, k=NUM_GAMES):
        self.k = k
        self.device_name = str(device)
        self.white = ConcurrentAgent(chess.WHITE, "White")
        self.black = ConcurrentAgent(chess.BLACK, "Black")
        self.slots = [Slot() for _ in range(k)]
        self.stats_reset()

    def stats_reset(self):
        self.games = 0
        self.wins = {"White": 0, "Black": 0, "Draw": 0}
        self.recent = deque(maxlen=100)
        self.recent_len = deque(maxlen=100)
        self.history = []                     # (white_rate, black_rate, draw_rate, avg_len)
        self.last_result = "-"

    def step(self):
        """Advance every game by one ply using one batched forward pass per agent."""
        by_agent = ((self.white, [i for i, s in enumerate(self.slots) if s.board.turn == chess.WHITE]),
                    (self.black, [i for i, s in enumerate(self.slots) if s.board.turn == chess.BLACK]))
        for agent, ids in by_agent:
            if not ids:
                continue
            moves = agent.act_batch(ids, [self.slots[i].board for i in ids])
            for i, mv in zip(ids, moves):
                self.slots[i].board.push(mv)
                self.slots[i].last_move = mv
        for i, s in enumerate(self.slots):
            outcome = s.board.outcome(claim_draw=True)
            if outcome is not None or len(s.board.move_stack) >= MAX_PLIES:
                self.finish(i, outcome)

    def finish(self, i, outcome):
        board = self.slots[i].board
        winner = outcome.winner if outcome else None
        if winner is None:
            tw = tb = DRAW_REWARD
            label = "Draw"
        elif winner == chess.WHITE:
            tw, tb, label = 1.0, -1.0, "White"
        else:
            tw, tb, label = -1.0, 1.0, "Black"
        self.white.finish_episode(i, material_balance(board, chess.WHITE), tw)
        self.black.finish_episode(i, material_balance(board, chess.BLACK), tb)

        plies = len(board.move_stack)
        self.games += 1
        self.wins[label] += 1
        self.recent.append(label)
        self.recent_len.append(plies)
        self.last_result = f"{label} ({plies} plies)"
        n = len(self.recent)
        self.history.append((self.recent.count("White") / n, self.recent.count("Black") / n,
                             self.recent.count("Draw") / n, sum(self.recent_len) / n))
        self.slots[i] = Slot()

    def save_models(self):
        self.white.save()
        self.black.save()

    def load_models(self):
        return self.white.load() and self.black.load()