import json
import os
import time
from datetime import datetime

from config import LOG_INTERVAL_SECONDS

COLUMNS = [
    "time_seconds", "timestamp", "games", "plies", "updates",
    "white_win_rate", "black_win_rate", "draw_rate",
    "white_actor_loss", "black_actor_loss", "white_critic_loss", "black_critic_loss",
    "white_total_loss", "black_total_loss",
    "white_policy_entropy", "black_policy_entropy",
    "white_value_explained_variance", "black_value_explained_variance",
    "white_gradient_norm", "black_gradient_norm",
    "repetition_rate", "threefold_repetition_rate", "reversal_move_rate",
    "stagnation_rate", "unique_position_ratio", "avg_game_length_plies",
    "mean_episode_reward", "terminal_reward_mean", "material_reward_mean",
    "draw_reward_mean", "repetition_penalty_mean", "reward_std",
    "games_per_second", "plies_per_second", "cpu_percent", "gpu_utilization_percent",
    "draw_by_repetition_rate", "draw_by_fifty_moves_rate", "draw_by_stalemate_rate",
    "draw_by_insufficient_material_rate", "draw_by_max_plies_rate", "draw_by_other_rate",
]


def _opt(snap, name, key):
    item = snap.get("optimization", {}).get(name, {})
    if item.get("policy_update_count", 0) <= 0:
        return None
    return item.get(key)


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


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
        quality = snap.get("game_quality", {})
        rewards = snap.get("rewards", {})
        throughput = snap.get("throughput", {})
        system = snap.get("system", {})
        draws = snap.get("draw_reasons", {})
        total_games = max(1, snap.get("throughput", {}).get("total_games", 0))

        record = {
            "time_seconds": round(now - self.started, 2),
            "timestamp": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "games": game.games,
            "plies": throughput.get("total_plies"),
            "updates": throughput.get("training_updates"),
            "white_win_rate": rate("White"),
            "black_win_rate": rate("Black"),
            "draw_rate": rate("Draw"),
            "repetition_rate": quality.get("repetition_rate"),
            "threefold_repetition_rate": quality.get("threefold_repetition_rate"),
            "reversal_move_rate": quality.get("reversal_move_rate"),
            "stagnation_rate": quality.get("stagnation_rate"),
            "unique_position_ratio": quality.get("unique_position_ratio"),
            "avg_game_length_plies": _mean(game.recent_len),
            "mean_episode_reward": rewards.get("mean_episode_reward"),
            "terminal_reward_mean": rewards.get("terminal_reward_mean"),
            "material_reward_mean": rewards.get("material_reward_mean"),
            "draw_reward_mean": rewards.get("draw_reward_mean"),
            "repetition_penalty_mean": rewards.get("repetition_penalty_mean"),
            "reward_std": rewards.get("reward_std"),
            "games_per_second": throughput.get("games_per_second"),
            "plies_per_second": throughput.get("plies_per_second"),
            "cpu_percent": system.get("cpu_percent"),
            "gpu_utilization_percent": system.get("gpu_utilization_percent"),
        }
        for color, name in (("white", "White"), ("black", "Black")):
            for metric in ("actor_loss", "critic_loss", "total_loss", "policy_entropy",
                           "value_explained_variance", "gradient_norm"):
                record[f"{color}_{metric}"] = _opt(snap, name, metric)

        for reason in ("repetition", "fifty_moves", "stalemate", "insufficient_material", "max_plies", "other"):
            count = draws.get(f"draw_by_{reason}", 0)
            record[f"draw_by_{reason}_rate"] = count / total_games

        self.records.append(record)

    def save(self, path, experiment, algorithm):
        algorithms = algorithm.get("algorithms", {}) if isinstance(algorithm, dict) else {}
        title = (f"{experiment.id}: So sánh giữa mô hình Trắng dùng "
                 f"{algorithms.get('White', 'Unknown')} và mô hình Đen dùng "
                 f"{algorithms.get('Black', 'Unknown')}")
        data = {
            "experiment": experiment.id,
            "title": title,
            "algorithm": algorithm,
            "resumed_from": experiment.resumed_from,
            "started_at": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "interval_seconds": self.interval,
            "columns": COLUMNS,
            "records": self.records,
        }
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(path + ".tmp", path)
