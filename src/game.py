from collections import deque

import chess

from agent import Agent
from config import DRAW_REWARD, MAX_PLIES
from encoding import material_balance


class SelfPlay:
    def __init__(self):
        self.white = Agent(chess.WHITE, "White")
        self.black = Agent(chess.BLACK, "Black")
        self.stats_reset()
        self.new_game()

    def stats_reset(self):
        self.games = 0
        self.wins = {"White": 0, "Black": 0, "Draw": 0}
        self.recent = deque(maxlen=100)       # last 100 results
        self.recent_len = deque(maxlen=100)
        self.history = []                     # (white_rate, black_rate, draw_rate, avg_len)
        self.last_result = "-"

    def new_game(self):
        self.board = chess.Board()
        self.last_move = None
        self.white.reset_episode()
        self.black.reset_episode()

    def step(self):
        agent = self.white if self.board.turn == chess.WHITE else self.black
        mv = agent.act(self.board)
        self.board.push(mv)
        self.last_move = mv
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None or len(self.board.move_stack) >= MAX_PLIES:
            self.finish(outcome)

    def finish(self, outcome):
        winner = outcome.winner if outcome else None
        if winner is None:
            tw = tb = DRAW_REWARD
            label = "Draw"
        elif winner == chess.WHITE:
            tw, tb, label = 1.0, -1.0, "White"
        else:
            tw, tb, label = -1.0, 1.0, "Black"
        self.white.learn(material_balance(self.board, chess.WHITE), tw)
        self.black.learn(material_balance(self.board, chess.BLACK), tb)

        plies = len(self.board.move_stack)
        self.games += 1
        self.wins[label] += 1
        self.recent.append(label)
        self.recent_len.append(plies)
        self.last_result = f"{label} ({plies} plies)"
        n = len(self.recent)
        self.history.append((self.recent.count("White") / n, self.recent.count("Black") / n,
                             self.recent.count("Draw") / n, sum(self.recent_len) / n))
        self.new_game()

    def save_models(self):
        self.white.save()
        self.black.save()

    def load_models(self):
        return self.white.load() and self.black.load()