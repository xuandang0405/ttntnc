import json
import os
import platform
import sys
import time
from datetime import datetime
 
import torch
 
from config import CKPT_DIR, DRAW_REWARD, GAMMA, LR, MAX_PLIES, SHAPING
 
PREFIX = "exp_"

def list_experiments():
    if not os.path.isdir(CKPT_DIR):
        return []
    return sorted(d for d in os.listdir(CKPT_DIR)
                  if d.startswith(PREFIX) and os.path.isdir(os.path.join(CKPT_DIR, d)))
 
 
def latest_checkpoint(name):
    fname = f"{name.lower()}.pt"
    for exp in reversed(list_experiments()):
        path = os.path.join(CKPT_DIR, exp, fname)
        if os.path.exists(path):
            return path
    legacy = os.path.join(CKPT_DIR, fname)
    return legacy if os.path.exists(legacy) else None
 
 
# metrics helpers
def process_memory_mb():
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
    # One training session = one timestamped folder.
    def __init__(self):
        self.started = time.time()
        self.id = datetime.fromtimestamp(self.started).strftime(PREFIX + "%Y%m%d_%H%M%S")
        self.dir = os.path.join(CKPT_DIR, self.id)
        self.resumed_from = None
        self.saves = 0
 
    def load_latest(self, white, black):
        if white.load() and black.load():
            self.resumed_from = os.path.basename(os.path.dirname(white.loaded_from)) or "legacy"
            return True
        return False
 
    def save(self, game, device_name, extra_hyper=None):
        # Write white.pt, black.pt and metrics.json into this experiment's folder.
        os.makedirs(self.dir, exist_ok=True)
        white_path = game.white.save(self.dir)
        black_path = game.black.save(self.dir)
        self.saves += 1
        now = time.time()
 
        metrics = {
            "experiment": self.id,
            "started_at": datetime.fromtimestamp(self.started).isoformat(timespec="seconds"),
            "saved_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "elapsed_seconds": round(now - self.started, 1),
            "save_count": self.saves,
            "resumed_from": self.resumed_from,
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
            "training": training_stats(game),
            "metrics": game.metrics.snapshot((game.white, game.black)),
        }
        out = os.path.join(self.dir, "metrics.json")
        with open(out + ".tmp", "w") as f:
            json.dump(metrics, f, indent=2)
        os.replace(out + ".tmp", out)       # atomic: never leaves a half-written file
        return metrics