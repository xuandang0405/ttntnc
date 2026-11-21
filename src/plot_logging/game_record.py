import json
import os
import time
from datetime import datetime

from config import CKPT_DIR

SAVE_INTERVAL_SECONDS = 120


class GameResultRecorder:
    def __init__(self, mode, white_name, black_name, path=None, interval=SAVE_INTERVAL_SECONDS):
        started = datetime.now()
        filename = f"game_results_{started.strftime('%Y%m%d_%H%M%S_%f')}.json"
        self.path = path or os.path.join(CKPT_DIR, "game_records", filename)
        self.mode = mode
        self.players = {"white": white_name, "black": black_name}
        self.started_at = started.isoformat(timespec="seconds")
        self.interval = interval
        self.games = []
        self.stats = {
            side: {"wins": 0, "losses": 0, "draws": 0}
            for side in ("white", "black")
        }
        self._last_saved = time.monotonic()

    def record_result(self, result):
        if result.winner is None:
            outcome, winner = "draw", "Draw"
        elif result.winner:
            outcome, winner = "white_win", "White"
        else:
            outcome, winner = "black_win", "Black"

        per_game = {}
        for side in ("white", "black"):
            player_result = (
                "draw" if outcome == "draw"
                else "win" if (side == "white") == (winner == "White")
                else "loss"
            )
            stat = {"win": "wins", "loss": "losses", "draw": "draws"}[player_result]
            self.stats[side][stat] += 1
            per_game[side] = {
                "name": self.players[side],
                "result": player_result,
            }

        self.games.append({
            "game": len(self.games) + 1,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "outcome": outcome,
            "winner": winner,
            "reason": result.reason,
            **per_game,
        })

    def save_if_due(self):
        if not self.games or time.monotonic() - self._last_saved < self.interval:
            return False
        self.save()
        return True

    def save(self):
        updated_at = datetime.now().isoformat(timespec="seconds")
        data = {
            "mode": self.mode,
            "players": self.players,
            "started_at": self.started_at,
            "updated_at": updated_at,
            "totals": {
                "games": len(self.games),
                "white": self.stats["white"],
                "black": self.stats["black"],
            },
            "games": self.games,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(self.path + ".tmp", self.path)
        self._last_saved = time.monotonic()
