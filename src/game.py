from collections import deque
 
import chess
 
from algorithms import build_agent, load_agent
from checkpoints import Experiment
from config import MAX_PLIES
from encoding import material_balance
from metrics import TrainingMetrics, summarize_game
from model import device
from plot_logging.training_log import TrainingLog
 
DEFAULT_CFG = {"algorithm": "A2C", "path": None}


class SelfPlay:
    def __init__(self, white_cfg=None, black_cfg=None):
        white_cfg, black_cfg = white_cfg or DEFAULT_CFG, black_cfg or DEFAULT_CFG
        self.white = build_agent(white_cfg, chess.WHITE, device, 1)    # 1 game -> update after every game
        self.black = build_agent(black_cfg, chess.BLACK, device, 1)
        self.experiment = Experiment()
        loaded = {n: c["path"] for n, c in (("White", white_cfg), ("Black", black_cfg)) if c.get("path")}
        self.experiment.resumed_from = loaded or None
        self.unsaved_games = 0
        self.log = TrainingLog(self.experiment.started)   # time series for plotting
        self.stats_reset()
        self.new_game()
 
    def stats_reset(self):
        self.games = 0
        self.wins = {"White": 0, "Black": 0, "Draw": 0}
        self.recent = deque(maxlen=100)       # last 100 results
        self.recent_len = deque(maxlen=100)
        self.history = []                     # (white_rate, black_rate, draw_rate, avg_len)
        self.last_result = "-"
        self.metrics = TrainingMetrics()
        self.white.metrics = self.metrics
        self.black.metrics = self.metrics
 
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
        self.metrics.add_plies(1)
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None or len(self.board.move_stack) >= MAX_PLIES:
            self.finish(outcome)
        self.log.maybe_record(self)
 
    def finish(self, outcome):
        label, reason, feats, tw, tb = summarize_game(self.board, outcome)
        self.metrics.record_game(reason, feats)
        self.white.learn(material_balance(self.board, chess.WHITE), tw)
        self.black.learn(material_balance(self.board, chess.BLACK), tb)
 
        plies = len(self.board.move_stack)
        self.games += 1
        self.unsaved_games += 1
        self.wins[label] += 1
        self.recent.append(label)
        self.recent_len.append(plies)
        self.last_result = f"{label} ({plies} plies)"
        n = len(self.recent)
        self.history.append((self.recent.count("White") / n, self.recent.count("Black") / n,
                             self.recent.count("Draw") / n, sum(self.recent_len) / n))
        self.new_game()
 
    def metrics_snapshot(self, max_age=0.0):
        return self.metrics.snapshot((self.white, self.black), max_age)
 
    def load_model(self, color, path):
        agent = load_agent(path, color, device, 1)
        agent.metrics = self.metrics
        if color == chess.WHITE:
            self.white = agent
        else:
            self.black = agent
        self.experiment.resumed_from = {**(self.experiment.resumed_from or {}),
                                        "White" if color == chess.WHITE else "Black": path}
        self.new_game()
 
    def save_models(self):
        if self.unsaved_games == 0:
            print("Nothing new to save.")
            return False
        self.experiment.save(self, str(device))
        self.unsaved_games = 0
        print(f"Saved -> {self.experiment.dir}")
        return True