from __future__ import annotations

import argparse
import hashlib
import io
import re
from collections import Counter

import chess
import chess.pgn
import zstandard as zstd
from pathlib import Path

from .common import MATE_CP, RESULTS, ShardWriter


_ZERO_CASTLE = re.compile(r"(?<![\w/-])0-0(-0)?(?![\w-])")

class _CastlingFixer:

    def __init__(self, handle):
        self.handle = handle

    def readline(self, *args):
        line = self.handle.readline(*args)
        if line and not line.startswith("["):
            line = _ZERO_CASTLE.sub(lambda m: "O-O-O" if m.group(1) else "O-O", line)
        return line

    def __iter__(self):
        return iter(self.readline, "")


def open_pgn(path: str):
    fh = open(path, "rb")
    if path.endswith(".zst"):
        text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
    else:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
    return _CastlingFixer(text)


def _int(value: str | None) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def pgn_evals(nodes: list) -> list | None:
    n = len(nodes)
    cps: list = [None] * n                      # cps[p]: position before ply p
    board, have = chess.Board(), 0
    for p, node in enumerate(nodes):
        board.push(node.move)
        ev = node.eval()                        # score of the position *after* this ply
        if ev is not None and p + 1 < n:
            cps[p + 1] = int(ev.pov(board.turn).score(mate_score=MATE_CP))
            have += 1
    return cps if have >= 0.7 * max(n - 1, 1) else None


def convert(game: chess.pgn.Game, a: argparse.Namespace) -> tuple[dict | None, str]:
    h = game.headers
    if game.errors:
        return None, "illegal_or_unparsable"
    if h.get("Variant", "Standard").lower() not in ("standard", "chess") or "FEN" in h or h.get("SetUp") == "1":
        return None, "non_standard_start"
    if h.get("Result") not in RESULTS:
        return None, "unfinished"
    nodes = list(game.mainline())
    n = len(nodes)
    if not a.min_plies <= n <= a.max_plies:
        return None, "length"
    result = RESULTS[h["Result"]]
    if result == 0.0 and n < a.min_draw_plies:
        return None, "short_draw"
    w_elo, b_elo = _int(h.get("WhiteElo")), _int(h.get("BlackElo"))
    if min(w_elo, b_elo) < a.min_elo:
        return None, "elo"
    moves = " ".join(nd.move.uci() for nd in nodes)
    rec = {"id": h.get("Site", ""), "event": h.get("Event", "?"), "round": h.get("Round", "?"),
           "white": h.get("White", "?"), "black": h.get("Black", "?"), "w_elo": w_elo, "b_elo": b_elo,
           "result": result, "moves": moves}
    cps = pgn_evals(nodes)
    if cps is not None:
        rec["sf"] = {"cp": cps, "best": [None] * n, "src": "pgn"}
    return rec, "ok"

def expand_inputs(inputs: list[str]) -> list[str]:
    files = []
    seen = set()

    for raw in inputs:
        path = Path(raw)

        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

        if path.is_dir():
            candidates = sorted(
                p for p in path.rglob("*")
                if p.is_file()
                and p.name.lower().endswith((".pgn", ".pgn.zst"))
            )
        elif path.is_file():
            if not path.name.lower().endswith((".pgn", ".pgn.zst")):
                raise ValueError(f"Unsupported input file: {path}")

            candidates = [path]
        else:
            raise ValueError(f"Invalid input path: {path}")

        for candidate in candidates:
            resolved = str(candidate.resolve())

            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)

    if not files:
        raise FileNotFoundError(
            "No .pgn or .pgn.zst files found in the supplied inputs."
        )

    return files

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".pgn/.pgn.zst files or directories containing PGN files")
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--min-elo", type=int, default=2000, help="both players; unrated (missing) counts as 0")
    ap.add_argument("--min-plies", type=int, default=20)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--min-draw-plies", type=int, default=40, help="drop 'agreed' short draws (no value signal)")
    ap.add_argument("--games-per-shard", type=int, default=1000)
    a = ap.parse_args()

    # Expand files and directories.
    try:
        input_files = expand_inputs(a.inputs)
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))

    print(f"Found {len(input_files)} PGN file(s):")

    for path in input_files:
        print(f"  - {path}")

    writer, stats, seen = (ShardWriter(a.out, a.games_per_shard), Counter(), set())

    for file_idx, path in enumerate(input_files, start=1):
        print(f"\n[{file_idx}/{len(input_files)}] " 
              f"Processing: {path}")
        stream = open_pgn(path)
        try:
            while (game := chess.pgn.read_game(stream)) is not None:
                stats["read"] += 1
                rec, why = convert(game, a)
                if rec is not None:
                    key = hashlib.sha1(
                        f"{rec['white']}|"
                        f"{rec['black']}|"
                        f"{rec['moves']}".encode()
                    ).digest()
                    if key in seen:
                        rec, why = None, "duplicate"
                    else:
                        seen.add(key)
                stats[why] += 1
                if rec is not None:
                    writer.add(rec)
        finally:
            stream.handle.close()

    manifest = writer.close({
        "filters": {
            k: v for k, v in vars(a).items() if k not in ("inputs", "out")
        },
        "stats": dict(stats),
        "inputs": input_files
    })

    print(
        f"\nCompleted: read {stats['read']} games "
        f"-> kept {manifest['games']} games "
        f"({manifest['plies']} plies)"
    )

    print(
        "Rejected:",
        {
            k: v for k, v in stats.items() if k not in ("read", "ok")
        }
    )

if __name__ == "__main__":
    main()
