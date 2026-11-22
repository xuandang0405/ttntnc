"""Attach Stockfish evaluations from lichess_db_eval.jsonl.zst (417M positions) to prepared shards.

    python -m pretrain.eval_join data/shards --evals data/lichess_db_eval.jsonl.zst --out data/labeled

One streaming pass over the eval file; only lines whose FEN key matches one of our positions are parsed.
Coverage is high in openings/early middlegames and thins out later. The eval file does not document whether
`cp` is white- or side-to-move-relative, so this script *measures* it: it correlates cp with material balance
and prints the verdict (override with --pov white|stm).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil

import chess
import numpy as np
import zstandard as zstd

from pretrain.common import MATE_CP, list_shards, read_jsonl_zst, write_jsonl_zst

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def key_of(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def material_white(board: chess.Board) -> int:
    return sum(v * (len(board.pieces(pt, True)) - len(board.pieces(pt, False))) for pt, v in VALUES.items())


def best_entry(entry: dict) -> tuple[int, str | None] | None:
    ev = max(entry["evals"], key=lambda e: e["depth"])
    pv = ev["pvs"][0]
    cp = pv["cp"] if "cp" in pv else (MATE_CP if pv["mate"] > 0 else -MATE_CP)
    line = pv.get("line", "").split()
    return cp, (line[0] if line else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards")
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--pov", choices=["auto", "white", "stm"], default="auto")
    a = ap.parse_args()

    games = {p: list(read_jsonl_zst(p)) for p in list_shards(a.shards)}
    wanted: dict[str, list] = {}                       # key -> [(path, game_idx, ply, stm_is_white, material)]
    for path, recs in games.items():
        for gi, rec in enumerate(recs):
            board = chess.Board()
            for p, uci in enumerate(rec["moves"].split()):
                if p % a.every == 0:
                    wanted.setdefault(key_of(board), []).append((path, gi, p, board.turn, material_white(board)))
                board.push_uci(uci)
    print(f"{len(wanted)} distinct positions to look up")

    found: dict[str, tuple[int, str | None]] = {}
    with open(a.evals, "rb") as fh:
        for line in io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
            i = line.find('"fen"')                        # tolerate both {"fen":"..."} and {"fen": "..."}
            start = line.find('"', i + 5) + 1
            key = line[start:line.find('"', start)]
            if key in wanted:
                found[key] = best_entry(json.loads(line))
    print(f"matched {len(found)} positions")
    if not found:
        raise SystemExit("no matches: check the --evals file")

    pov = a.pov
    if pov == "auto":
        cps, mats, stm = [], [], []
        for key, (cp, _) in found.items():
            for _, _, _, white_to_move, mat in wanted[key][:1]:
                cps.append(cp), mats.append(mat), stm.append(1 if white_to_move else -1)
        cps, mats, stm = map(np.asarray, (cps, mats, stm))
        c_white = np.corrcoef(cps, mats)[0, 1]
        c_stm = np.corrcoef(cps, mats * stm)[0, 1]
        pov = "white" if c_white >= c_stm else "stm"
        print(f"corr(cp, material): white-POV {c_white:.3f} vs stm-POV {c_stm:.3f} -> treating cp as {pov}-POV")

    os.makedirs(a.out, exist_ok=True)
    for path, recs in games.items():
        for rec in recs:
            n = len(rec["moves"].split())
            rec["sf"] = {"cp": [None] * n, "best": [None] * n, "src": "lichess_eval"}
        for key, (cp, best) in found.items():
            for (p_, gi, ply, white_to_move, _mat) in wanted[key]:
                if p_ == path:
                    sf = recs[gi]["sf"]
                    sf["cp"][ply] = cp if (pov == "stm" or white_to_move) else -cp
                    sf["best"][ply] = best
        write_jsonl_zst(os.path.join(a.out, os.path.basename(path)), recs)
    manifest = os.path.join(a.shards, "manifest.json")
    if os.path.exists(manifest):
        shutil.copy(manifest, os.path.join(a.out, "manifest.json"))
    total = sum(len(v) for v in wanted.values())
    hit = sum(len(wanted[k]) for k in found)
    print(f"labeled {hit}/{total} positions ({100 * hit / total:.1f}%)")


if __name__ == "__main__":
    main()
