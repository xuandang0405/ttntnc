import statistics
import time
from collections import Counter, deque
 
import chess
 
from config import DRAW_REWARD, REPETITION_PENALTY
 
DRAW_KEYS = ("draw_by_repetition", "draw_by_fifty_moves", "draw_by_stalemate",
             "draw_by_insufficient_material", "draw_by_max_plies", "draw_by_other")
OPT_KEYS = ("actor_loss", "critic_loss", "total_loss", "policy_entropy", "mean_advantage",
            "std_advantage", "mean_return", "mean_value", "value_explained_variance",
            "gradient_norm", "gradient_clip_fraction", "weight_update_norm")
 
 
# per-game analysis
def draw_reason(outcome):
    if outcome is None:
        return "draw_by_max_plies"
    T = chess.Termination
    t = outcome.termination
    if t in (T.THREEFOLD_REPETITION, T.FIVEFOLD_REPETITION):
        return "draw_by_repetition"
    if t in (T.FIFTY_MOVES, T.SEVENTYFIVE_MOVES):
        return "draw_by_fifty_moves"
    if t == T.STALEMATE:
        return "draw_by_stalemate"
    if t == T.INSUFFICIENT_MATERIAL:
        return "draw_by_insufficient_material"
    return "draw_by_other"
 
 
def game_features(board):
    replay = chess.Board()
    seen = Counter([replay.epd()])
    post = set()
    repeated = repeated_moves = reversals = stagnant = 0
    max_count = 1
    own_moves = {chess.WHITE: set(), chess.BLACK: set()}
    last = {chess.WHITE: None, chess.BLACK: None}
    for mv in board.move_stack:
        color = replay.turn
        if not (replay.is_capture(mv) or replay.piece_type_at(mv.from_square) == chess.PAWN):
            stagnant += 1
        uci = mv.uci()
        if uci in own_moves[color]:
            repeated_moves += 1
        own_moves[color].add(uci)
        prev = last[color]
        if prev and mv.from_square == prev.to_square and mv.to_square == prev.from_square:
            reversals += 1
        last[color] = mv
        replay.push(mv)
        key = replay.epd()
        if seen[key] > 0:
            repeated += 1
        seen[key] += 1
        max_count = max(max_count, seen[key])
        post.add(key)
    n = max(1, len(board.move_stack))
    return {"plies": len(board.move_stack), "repeated_positions": repeated,
            "unique_positions": len(post), "unique_ratio": len(post) / n,
            "repetition_rate": repeated / n, "threefold": max_count >= 3,
            "repeated_moves": repeated_moves, "stagnation_rate": stagnant / n,
            "reversal_rate": reversals / n}
 
 
def summarize_game(board, outcome):
    feats = game_features(board)
    rep = -REPETITION_PENALTY * feats["repeated_positions"]
    winner = outcome.winner if outcome else None
    if winner is None:
        tw = {"win_loss": 0.0, "draw": DRAW_REWARD, "repetition": rep}
        return "Draw", draw_reason(outcome), feats, tw, dict(tw)
    s = 1.0 if winner == chess.WHITE else -1.0
    tw = {"win_loss": s, "draw": 0.0, "repetition": rep}
    tb = {"win_loss": -s, "draw": 0.0, "repetition": rep}
    return ("White" if winner == chess.WHITE else "Black"), None, feats, tw, tb
 
 
# collector
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
 
 
def _r(x, nd=4):
    return round(float(x), nd)
 
 
class TrainingMetrics:
    RATE_WINDOW_S = 30.0
 
    def __init__(self, game_window=100, update_window=50):
        self.t0 = time.time()
        self.total_plies = self.total_games = 0
        self.samples = deque()                         # (time, plies, games) for rates
        self.draws = {k: 0 for k in DRAW_KEYS}
        self.games = deque(maxlen=game_window)         # per-game feature dicts
        self.episodes = deque(maxlen=game_window * 2)  # per-episode reward records (both agents)
        self.updates = {"White": deque(maxlen=update_window), "Black": deque(maxlen=update_window)}
        self.update_counts = {"White": 0, "Black": 0}
        self._cache, self._cache_t = None, 0.0
        self._sys, self._sys_t = {}, 0.0
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)               # prime the counters
            psutil.cpu_percent(None)
        except ImportError:
            pass
 
    #  recording 
    def _sample(self, force=False):
        now = time.time()
        if force or not self.samples or now - self.samples[-1][0] >= 0.1:
            self.samples.append((now, self.total_plies, self.total_games))
        while len(self.samples) > 2 and self.samples[0][0] < self.samples[-1][0] - self.RATE_WINDOW_S:
            self.samples.popleft()
 
    def add_plies(self, n=1):
        self.total_plies += n
        self._sample()
 
    def record_game(self, reason, features):
        self.total_games += 1
        if reason:
            self.draws[reason] += 1
        self.games.append(features)
        self._sample(force=True)
 
    def record_episode(self, name, rec):
        self.episodes.append(rec)
 
    def record_update(self, name, stats):
        self.updates[name].append(stats)
        self.update_counts[name] += 1
 
    #  reporting 
    def _rates(self):
        if len(self.samples) < 2:
            return 0.0, 0.0
        (t0, p0, g0), (t1, p1, g1) = self.samples[0], self.samples[-1]
        dt = t1 - t0
        return ((g1 - g0) / dt, (p1 - p0) / dt) if dt > 0 else (0.0, 0.0)
 
    def _system(self):
        now = time.time()
        if self._sys and now - self._sys_t < 1.0:
            return self._sys
        cpu = proc_cpu = gpu = None
        try:
            import psutil
            cpu = psutil.cpu_percent(None)
            proc_cpu = self._proc.cpu_percent(None) if self._proc else None
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                gpu = float(torch.cuda.utilization())
        except Exception:
            gpu = None
        self._sys = {"cpu_percent": cpu, "process_cpu_percent": proc_cpu,
                     "gpu_utilization_percent": gpu}
        self._sys_t = now
        return self._sys
 
    def snapshot(self, agents=(), max_age=0.0):
        now = time.time()
        if max_age and self._cache is not None and now - self._cache_t < max_age:
            return self._cache
        gps, pps = self._rates()
        g = self.games
        eps = list(self.episodes)
        steps = sum(e["steps"] for e in eps)
        ep_rewards = [e["reward"] for e in eps]
 
        optimization = {}
        for name, hist in self.updates.items():
            d = {k: _r(_mean(s[k] for s in hist)) for k in OPT_KEYS}
            d["policy_update_count"] = self.update_counts[name]
            optimization[name] = d
 
        params = {a.name: sum(p.numel() for p in a.net.parameters()) for a in agents}
        trainable = {a.name: sum(p.numel() for p in a.net.parameters() if p.requires_grad)
                     for a in agents}
        params["total"], trainable["total"] = sum(params.values()), sum(trainable.values())
 
        snap = {
            "throughput": {
                "games_per_second": _r(gps), "plies_per_second": _r(pps, 2),
                "training_updates": sum(self.update_counts.values()),
                "total_games": self.total_games, "total_plies": self.total_plies,
                "elapsed_seconds": _r(now - self.t0, 1)},
            "draw_reasons": dict(self.draws),
            "game_quality": {
                "window_games": len(g),
                "repetition_rate": _r(_mean(x["repetition_rate"] for x in g)),
                "threefold_repetition_rate": _r(_mean(1.0 if x["threefold"] else 0.0 for x in g)),
                "repeated_position_count": _r(_mean(x["repeated_positions"] for x in g), 2),
                "repeated_move_count": _r(_mean(x["repeated_moves"] for x in g), 2),
                "stagnation_rate": _r(_mean(x["stagnation_rate"] for x in g)),
                "reversal_move_rate": _r(_mean(x["reversal_rate"] for x in g)),
                "unique_position_ratio": _r(_mean(x["unique_ratio"] for x in g)),
                "avg_unique_positions": _r(_mean(x["unique_positions"] for x in g), 2)},
            "optimization": optimization,
            "rewards": {
                "window_episodes": len(eps),
                "mean_episode_reward": _r(_mean(ep_rewards)),
                "mean_step_reward": _r(sum(ep_rewards) / steps if steps else 0.0),
                "terminal_reward_mean": _r(_mean(e["terminal"] for e in eps)),
                "material_reward_mean": _r(_mean(e["material"] for e in eps)),
                "draw_reward_mean": _r(_mean(e["draw"] for e in eps)),
                "repetition_penalty_mean": _r(_mean(e["repetition"] for e in eps)),
                "reward_std": _r(statistics.pstdev(ep_rewards) if len(ep_rewards) > 1 else 0.0),
                "positive_reward_ratio": _r(sum(e["pos"] for e in eps) / steps if steps else 0.0),
                "negative_reward_ratio": _r(sum(e["neg"] for e in eps) / steps if steps else 0.0)},
            "system": self._system(),
            "parameters": params,
            "trainable_parameters": trainable,
        }
        self._cache, self._cache_t = snap, now
        return snap