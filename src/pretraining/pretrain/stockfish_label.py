from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time

import chess
import chess.engine

from pretrain.common import MATE_CP, list_shards, read_jsonl_zst, write_jsonl_zst

_ENGINE = None
_CFG: dict = {}


def _init(path: str, threads: int, hash_mb: int, nodes: int, every: int) -> None:
    global _ENGINE, _CFG
    _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    _ENGINE.configure({"Threads": threads, "Hash": hash_mb})
    _CFG = {"nodes": nodes, "every": every}


def label_game(rec: dict, engine, nodes: int, every: int) -> dict:
    moves = rec["moves"].split()
    n = len(moves)
    cps, best = [None] * n, [None] * n
    board = chess.Board()
    for p, uci in enumerate(moves):
        if p % every == 0 and not board.is_game_over(claim_draw=False):
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
            cps[p] = int(info["score"].pov(board.turn).score(mate_score=MATE_CP))
            pv = info.get("pv")
            best[p] = pv[0].uci() if pv else None
        board.push_uci(uci)
    rec = dict(rec)
    rec["sf"] = {"cp": cps, "best": best, "src": "stockfish"}
    return rec


def _work(job: tuple[str, str]) -> tuple[str, int, float]:
    src, dst = job
    t0, recs = time.time(), [label_game(r, _ENGINE, _CFG["nodes"], _CFG["every"]) for r in read_jsonl_zst(src)]
    write_jsonl_zst(dst, recs)
    return os.path.basename(src), len(recs), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default=shutil.which("stockfish") or "stockfish")
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--every", type=int, default=4, help="label every k-th position (cost is linear in 1/k)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(p, os.path.join(a.out, os.path.basename(p))) for p in list_shards(a.shards)]
    with mp.Pool(a.workers, _init, (a.engine, a.threads, a.hash, a.nodes, a.every)) as pool:
        for name, n, secs in pool.imap_unordered(_work, jobs):
            print(f"{name}: {n} games in {secs:.0f}s")
    src_manifest = os.path.join(a.shards, "manifest.json")
    if os.path.exists(src_manifest):
        shutil.copy(src_manifest, os.path.join(a.out, "manifest.json"))


if __name__ == "__main__":
    main()
