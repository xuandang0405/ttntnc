import random
from collections import deque
 
import chess
 
from checkpoints import Experiment
from concurrent_mode.concurrent_agent import ConcurrentAgent, device
from concurrent_mode.concurrent_config import EPISODES_PER_UPDATE, NUM_GAMES, RANDOM_OPENING_PLIES
from config import MAX_PLIES
from encoding import material_balance
from metrics import TrainingMetrics, summarize_game
from plot_logging.training_log import TrainingLog
 
class Slot:

    def __init__(self):
        self.board = chess.Board()
        for _ in range(random.randint(*RANDOM_OPENING_PLIES)):
            self.board.push(random.choice(list(self.board.legal_moves)))
            if self.board.outcome(claim_draw=True) is not None:   # never start an already-ended game
                self.board.pop()
                break
        self.last_move = self.board.peek() if self.board.move_stack else None
 
class ConcurrentSelfPlay:
    def __init__(self, k=NUM_GAMES):
        self.k = k
        self.device_name = str(device)
        self.white = ConcurrentAgent(chess.WHITE, "White")
        self.black = ConcurrentAgent(chess.BLACK, "Black")
        self.slots = [Slot() for _ in range(k)]
        self.experiment = Experiment()   # timestamped folder for this session
        self.unsaved_games = 0
        self.log = TrainingLog(self.experiment.started)   # time series for plotting
        self.stats_reset()
 
    def stats_reset(self):
        self.games = 0
        self.wins = {"White": 0, "Black": 0, "Draw": 0}
        self.recent = deque(maxlen=100)
        self.recent_len = deque(maxlen=100)
        self.history = []                     # (white_rate, black_rate, draw_rate, avg_len)
        self.last_result = "-"
        self.metrics = TrainingMetrics()
        self.white.metrics = self.metrics
        self.black.metrics = self.metrics
 
    def step(self):
        by_agent = ((self.white, [i for i, s in enumerate(self.slots) if s.board.turn == chess.WHITE]),
                    (self.black, [i for i, s in enumerate(self.slots) if s.board.turn == chess.BLACK]))
        for agent, ids in by_agent:
            if not ids:
                continue
            moves = agent.act_batch(ids, [self.slots[i].board for i in ids])
            for i, mv in zip(ids, moves):
                self.slots[i].board.push(mv)
                self.slots[i].last_move = mv
            self.metrics.add_plies(len(moves))
        for i, s in enumerate(self.slots):
            outcome = s.board.outcome(claim_draw=True)
            if outcome is not None or len(s.board.move_stack) >= MAX_PLIES:
                self.finish(i, outcome)
        self.log.maybe_record(self)
 
    def finish(self, i, outcome):
        board = self.slots[i].board
        label, reason, feats, tw, tb = summarize_game(board, outcome)
        self.metrics.record_game(reason, feats)
        self.white.finish_episode(i, material_balance(board, chess.WHITE), tw)
        self.black.finish_episode(i, material_balance(board, chess.BLACK), tb)
 
        plies = len(board.move_stack)
        self.games += 1
        self.unsaved_games += 1
        self.wins[label] += 1
        self.recent.append(label)
        self.recent_len.append(plies)
        self.last_result = f"{label} ({plies} plies)"
        n = len(self.recent)
        self.history.append((self.recent.count("White") / n, self.recent.count("Black") / n,
                             self.recent.count("Draw") / n, sum(self.recent_len) / n))
        self.slots[i] = Slot()
 
    def metrics_snapshot(self, max_age=0.0):
        return self.metrics.snapshot((self.white, self.black), max_age)

    def save_models(self):
        if self.unsaved_games == 0:
            print("Nothing new to save.")
            return False
        self.experiment.save(self, self.device_name,
                             {"parallel_games": self.k, "episodes_per_update": EPISODES_PER_UPDATE})
        self.unsaved_games = 0
        print(f"Saved -> {self.experiment.dir}")
        return True
 
    def load_models(self):
        return self.experiment.load_latest(self.white, self.black)