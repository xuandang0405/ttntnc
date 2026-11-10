import json
import os
import time
from datetime import datetime
 
from config import LOG_INTERVAL_SECONDS
 
COLUMNS = [
    "time_seconds", "timestamp", "games", "plies", "updates",
    "white_win_rate", "black_win_rate", "draw_rate",
    "white_actor_loss", "white_critic_loss", "black_actor_loss", "black_critic_loss",
    "white_policy_entropy", "black_policy_entropy",
    "repetition_rate", "cpu_percent", "gpu_utilization_percent",
]
 
 
def _opt(snap, name, key):
    o = snap["optimization"][name]
    return o[key] if o["policy_update_count"] > 0 else None
 
 
class TrainingLog:
    def __init__(self, started, interval=LOG_INTERVAL_SECONDS):
        self.started, self.interval = started, interval
        self.records = []
        self._last_t, self._last_plies = 0.0, -1
 
    def maybe_record(self, game, force=False):
        now = time.time()
        if not force:
            if now - self._last_t < self.interval:
                return
            if game.metrics.total_plies == self._last_plies:
                return
        self._last_t, self._last_plies = now, game.metrics.total_plies
 
        snap = game.metrics.snapshot((game.white, game.black))
        n = len(game.recent)
        rate = lambda k: round(game.recent.count(k) / n, 4) if n else None
        gq, sysm = snap["game_quality"], snap["system"]
        self.records.append({
            "time_seconds": round(now - self.started, 2),
            "timestamp": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "games": game.games,
            "plies": snap["throughput"]["total_plies"],
            "updates": snap["throughput"]["training_updates"],
            "white_win_rate": rate("White"), "black_win_rate": rate("Black"), "draw_rate": rate("Draw"),
            "white_actor_loss": _opt(snap, "White", "actor_loss"),
            "white_critic_loss": _opt(snap, "White", "critic_loss"),
            "black_actor_loss": _opt(snap, "Black", "actor_loss"),
            "black_critic_loss": _opt(snap, "Black", "critic_loss"),
            "white_policy_entropy": _opt(snap, "White", "policy_entropy"),
            "black_policy_entropy": _opt(snap, "Black", "policy_entropy"),
            "repetition_rate": gq["repetition_rate"] if gq["window_games"] else None,
            "cpu_percent": sysm["cpu_percent"],
            "gpu_utilization_percent": sysm["gpu_utilization_percent"],
        })
 
    def save(self, path, experiment):
        data = {
            "experiment": experiment.id,
            "resumed_from": experiment.resumed_from,
            "started_at": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "interval_seconds": self.interval,
            "columns": COLUMNS,
            "records": self.records,
        }
        with open(path + ".tmp", "w") as f:
            json.dump(data, f)
        os.replace(path + ".tmp", path)