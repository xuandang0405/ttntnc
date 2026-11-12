import json
import os
import platform
import sys
import time
from datetime import datetime
 
import torch
 
from config import CKPT_DIR, DRAW_REWARD, GAMMA, LR, MAX_PLIES, SHAPING
 
PREFIX = "exp_"
 
 
# metrics helpers
def process_memory_mb():
    """Memory of this process in MB (current RSS with psutil, else peak RSS, else None)."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        pass
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # KB on Linux, bytes on macOS
        return round(peak / (1e6 if sys.platform == "darwin" else 1e3), 1)
    except ImportError:
        return None
 
 
def model_info(agent, path):
    params = list(agent.net.parameters())
    return {
        "algorithm": agent.algorithm,
        "file": os.path.basename(path),
        "parameters": sum(p.numel() for p in params),
        "trainable_parameters": sum(p.numel() for p in params if p.requires_grad),
        "parameter_size_mb": round(sum(p.numel() * p.element_size() for p in params) / 1e6, 3),
        "checkpoint_file_mb": round(os.path.getsize(path) / 1e6, 3),
        "learning_rate": agent.opt.param_groups[0]["lr"],
        "last_loss": agent.last_loss,
    }
 
 
def training_stats(game):
    n = len(game.recent)
    return {
        "games_played": game.games,
        "wins": dict(game.wins),
        "recent_window": n,
        "recent_rates": {k: round(game.recent.count(k) / n, 3) if n else 0.0
                         for k in ("White", "Black", "Draw")},
        "avg_game_length_plies": round(sum(game.recent_len) / n, 1) if n else 0.0,
        "last_result": game.last_result,
    }
 
 
def base_hyperparameters():
    return {"learning_rate": LR, "gamma": GAMMA, "reward_shaping": SHAPING,
            "draw_reward": DRAW_REWARD, "max_plies": MAX_PLIES}
 
 
# the experiment
class Experiment:
    def __init__(self):
        self.started = time.time()
        self.id = datetime.fromtimestamp(self.started).strftime(PREFIX + "%Y%m%d_%H%M%S")
        self.dir = os.path.join(CKPT_DIR, self.id)
        self.resumed_from = None
        self.saves = 0
 
    def save(self, game, device_name, extra_hyper=None):
        os.makedirs(self.dir, exist_ok=True)
        white_path = game.white.save(self.dir)
        black_path = game.black.save(self.dir)
        self.saves += 1
        now = time.time()
        algorithms = {"White": game.white.algorithm, "Black": game.black.algorithm}
        white_label = getattr(game.white, "label", game.white.algorithm.upper())
        black_label = getattr(game.black, "label", game.black.algorithm.upper())
        self.title = (f"{self.id}: So sánh giữa mô hình Trắng dùng {white_label} "
                        f"và mô hình Đen dùng {black_label}")
 
        metrics = {
            "experiment": self.id,
            "title": self.title,
            "started_at": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "saved_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "elapsed_seconds": round(now - self.started, 1),
            "save_count": self.saves,
            "resumed_from": self.resumed_from,
            "algorithms": algorithms,
            "system": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "device": device_name,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "memory": {
                "process_mb": process_memory_mb(),
                "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1e6, 1)
                if torch.cuda.is_available() else None,
                "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if torch.cuda.is_available() else None,
            },
            "models": {"White": model_info(game.white, white_path),
                       "Black": model_info(game.black, black_path)},
            "hyperparameters": {**base_hyperparameters(), **(extra_hyper or {})},
            "algorithm_hyperparameters": {"White": game.white.hyperparameters(),
                                          "Black": game.black.hyperparameters()},
            "training": training_stats(game),
            "metrics": game.metrics.snapshot((game.white, game.black)),
        }
        out = os.path.join(self.dir, "metrics.json")
        with open(out + ".tmp", "w") as f:
            json.dump(metrics, f, indent=2)
        os.replace(out + ".tmp", out)
 
        game.log.maybe_record(game, force=True) 
        game.log.save(os.path.join(self.dir, "training_log.json"), self, {"algorithms": algorithms})
        return metrics