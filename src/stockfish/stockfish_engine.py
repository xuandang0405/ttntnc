import atexit
import math
import os
import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import chess
import chess.engine

try:
    from .stockfish_config import STOCKFISH_HASH_MB, STOCKFISH_PATH
except ImportError:
    from stockfish_config import STOCKFISH_HASH_MB, STOCKFISH_PATH

MATE_CP = 10_000


def find_stockfish(path=None):
    for candidate in (path, os.environ.get("STOCKFISH_PATH"), STOCKFISH_PATH):
        if candidate and os.path.isfile(candidate):
            return candidate
    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "Stockfish executable not found. Download it from https://stockfishchess.org/download/ and "
        "set STOCKFISH_PATH to the .exe (PowerShell: $env:STOCKFISH_PATH = 'C:\\path\\stockfish.exe')."
        "or copy the excecutable to the current directory (/src/stockfish)."
    )


def cp_to_value(cp):
    # Centipawns -> expected result in [-1, 1] (Lichess win-chance curve).# 
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0


@dataclass(frozen=True)
class Line:
    move: chess.Move
    cp: int            # score for the side to move (mate = +-(10000 - plies))
    value: float       # cp_to_value(cp)


class StockfishPool:
    # N Stockfish processes behind a thread pool; `submit` is non-blocking.
    def __init__(self, path=None, engines=1, threads=1, hash_mb=STOCKFISH_HASH_MB):
        self.path = find_stockfish(path)
        self._free = queue.Queue()
        self._all = []
        engines = max(1, int(engines))
        try:
            for _ in range(engines):
                engine = chess.engine.SimpleEngine.popen_uci(self.path)
                engine.configure({"Threads": threads, "Hash": hash_mb})
                self._all.append(engine)
                self._free.put(engine)
            self._executor = ThreadPoolExecutor(max_workers=engines)
            self._closed = False
            self._close_lock = threading.Lock()
            self._atexit_id = atexit.register(self.close)
        except Exception:
            for engine in self._all:
                try:
                    engine.quit()
                except Exception:
                    try:
                        engine.close()
                    except Exception:
                        pass
            raise

    def _analyse(self, board, nodes, multipv):
        engine = self._free.get()
        try:
            infos = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv,
                                   info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
        finally:
            self._free.put(engine)
        lines = []
        for info in infos:
            pv = info.get("pv")
            if not pv or "score" not in info:
                continue
            cp = int(info["score"].pov(board.turn).score(mate_score=MATE_CP))
            lines.append(Line(pv[0], cp, cp_to_value(cp)))
        lines.sort(key=lambda l: -l.cp)
        return lines

    def submit(self, board, nodes, multipv=1):
        # Start analysing `board` (not modified); returns a Future of a list of Line, best first.
        return self._executor.submit(self._analyse, board.copy(stack=True), nodes, multipv)

    def analyse(self, board, nodes, multipv=1):
        return self.submit(board, nodes, multipv).result()

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                atexit.unregister(self._atexit_id)
            except Exception:
                pass

            # Wait for tasks already inside the pool before terminating UCI transports.
            self._executor.shutdown(wait=True, cancel_futures=True)
            for engine in self._all:
                try:
                    engine.quit()
                except Exception:
                    try:
                        engine.close()
                    except Exception:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
