from collections import OrderedDict

from stockfish.stockfish_config import (COACH_CACHE, COACH_ENABLED, COACH_ENGINES, COACH_EVAL_CLIP,
                              COACH_NODES)
from stockfish.stockfish_engine import StockfishPool


class StockfishCoach:
    def __init__(self, pool, nodes=COACH_NODES, clip=COACH_EVAL_CLIP, cache_size=COACH_CACHE):
        self.pool, self.nodes, self.clip, self.cache_size = pool, nodes, clip, cache_size
        self._cache = OrderedDict()      # position key -> centipawns for the side to move
        self._pending = {}               # position key -> Future
        self.hits = self.misses = 0

    @staticmethod
    def _key(board):
        return board.epd()

    def prefetch(self, boards):
        for board in boards:
            key = self._key(board)
            if key not in self._cache and key not in self._pending:
                self._pending[key] = self.pool.submit(board, self.nodes, multipv=1)

    def _centipawns(self, board):
        key = self._key(board)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        future = self._pending.pop(key, None) or self.pool.submit(board, self.nodes, multipv=1)
        lines = future.result()
        cp = lines[0].cp if lines else 0
        self._cache[key] = cp
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cp

    def potential(self, board, color):
        """Evaluation in pawns from `color`'s point of view, clipped to +-clip."""
        cp = self._centipawns(board)
        if board.turn != color:
            cp = -cp
        return max(-self.clip, min(self.clip, cp / 100.0))

    def info(self):
        return {"enabled": True, "nodes": self.nodes, "eval_clip_pawns": self.clip,
                "engines": len(self.pool._all)}


_coach = None


def get_coach():
    # The shared coach, or None when CHESS_AI_STOCKFISH_COACH is not '1'.
    global _coach
    if not COACH_ENABLED:
        return None
    if _coach is None:
        _coach = StockfishCoach(StockfishPool(engines=COACH_ENGINES))
    return _coach


def coach_info():
    return _coach.info() if _coach is not None else {"enabled": False}
