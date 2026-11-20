import atexit
import math
import os
import queue
import shutil
import threading
import multiprocessing
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import chess
import chess.engine
from stockfish.stockfish_config import STOCKFISH_HASH_MB, STOCKFISH_PATH

MATE_CP = 10_000
ENGINE_TIMEOUT = 60.0
QUIT_TIMEOUT = 5.0


def find_stockfish(path=None):
    bundled_path = os.path.join(os.path.dirname(__file__), "stockfish.exe")
    candidates = (path, os.environ.get("STOCKFISH_PATH"), STOCKFISH_PATH, bundled_path)
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.expandvars(os.path.expanduser(str(candidate).strip().strip('"')))
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return os.path.abspath(found)

    raise FileNotFoundError(
        f"Stockfish executable not found. Place a valid Stockfish binary at {bundled_path!r}, "
        "set STOCKFISH_PATH to its full path, then copy the executable to /src/stockfish."
    )


def cp_to_value(cp):
    # Convert centipawns to expected result in [-1, 1].
    cp = max(-MATE_CP, min(MATE_CP, float(cp)))
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0
@dataclass(frozen=True)
class Line:
    move: chess.Move
    cp: int
    value: float


class StockfishPool:
    # N Stockfish processes behind a thread pool; submit() returns a Future.

    def __init__(self, path=None, engines=1, threads=1, hash_mb=STOCKFISH_HASH_MB):
        self.path = find_stockfish(path)
        self._free = queue.Queue()
        self._all = []
        self._closed = False
        self._close_lock = threading.Lock()
        self._executor = None
        engines = max(1, int(engines))
        threads = max(1, int(threads))
        hash_mb = max(1, int(hash_mb))
        try:
            for _ in range(engines):
                engine = chess.engine.SimpleEngine.popen_uci(self.path, timeout=ENGINE_TIMEOUT)
                self._all.append(engine)
                engine.configure({"Threads": threads, "Hash": hash_mb})
                self._free.put(engine)
            self._executor = ThreadPoolExecutor(max_workers=engines, thread_name_prefix="stockfish")
            # The ProcessPool worker has its own multiprocessing Finalize callback.
            # Register atexit only in the main process to avoid competing shutdown
            # mechanisms inside spawned workers.
            self._atexit_id = None
            if multiprocessing.current_process().name == "MainProcess":
                self._atexit_id = atexit.register(self.close)
        except Exception:
            self._terminate_engines()
            raise

    def _analyse(self, board, nodes, multipv):
        engine = self._free.get()
        try:
            infos = engine.analyse(
                board, chess.engine.Limit(nodes=max(1, int(nodes))),
                multipv=max(1, int(multipv)),
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
            )
            lines = []
            for info in infos:
                pv = info.get("pv")
                if not pv or "score" not in info:
                    continue
                cp = int(info["score"].pov(board.turn).score(mate_score=MATE_CP))
                lines.append(Line(pv[0], cp, cp_to_value(cp)))
            lines.sort(key=lambda line: -line.cp)
            return lines
        finally:
            # Return the transport even when analysis fails. A failed transport will
            # raise again on its next use, rather than silently losing a pool slot.
            self._free.put(engine)

    def submit(self, board, nodes, multipv=1):
        if self._closed:
            raise RuntimeError("StockfishPool is closed")
        return self._executor.submit(self._analyse, board.copy(stack=True), nodes, multipv)

    def analyse(self, board, nodes, multipv=1):
        return self.submit(board, nodes, multipv).result()
    
    @staticmethod
    def _force_kill_engine(engine):
        try:
            transport = getattr(engine, "transport", None)
            process = (
                transport.get_extra_info("subprocess")
                if transport is not None else None
            )
            if process is not None and process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=QUIT_TIMEOUT)
                except Exception:
                    pass
        except BaseException as exc:
            print(f"[POOL] hard-kill failed: {exc!r}", flush=True)

    def _terminate_engines(self):
        engines = list(self._all)
        self._all.clear()

        for i, engine in enumerate(engines):
            print(f"[POOL] closing engine {i}...", flush=True)
            try:
                # python-chess applies this timeout to the UCI quit handshake.
                engine.timeout = QUIT_TIMEOUT
                engine.quit()
                print(f"[POOL] engine {i} quit successfully", flush=True)
            except BaseException as exc:
                print(f"[POOL] engine {i} quit failed: {exc!r}", flush=True)
                # Kill the Stockfish subprocess before closing the protocol
                # transport; this prevents a stuck child from holding up worker exit.
                self._force_kill_engine(engine)
                try:
                    engine.close()
                    print(f"[POOL] engine {i} transport closed", flush=True)
                except BaseException as close_exc:
                    print(f"[POOL] engine {i} close failed: {close_exc!r}", flush=True)

    def close(self):
        with self._close_lock:
            if self._closed:
                return

            self._closed = True
            print("[POOL] close started", flush=True)

            if self._atexit_id is not None:
                try:
                    atexit.unregister(self.close)
                except Exception:
                    pass
                self._atexit_id = None

            if self._executor is not None:
                print("[POOL] executor shutdown started", flush=True)

                self._executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )

                self._executor = None
                print("[POOL] executor shutdown completed", flush=True)

            print("[POOL] engine termination started", flush=True)

            self._terminate_engines()

            print("[POOL] engine termination completed", flush=True)
            print("[POOL] close completed", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
