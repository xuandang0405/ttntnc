from __future__ import annotations

import glob
import io
import json
import math
import os
import zlib
from typing import Iterable, Iterator

import zstandard as zstd

RESULTS = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}    # white's point of view
MATE_CP = 10_000                                        # mate scores are clamped to this many centipawns
CP_SCALE = 543.2                                        # tanh(cp / 543.2) == Lichess' win% curve mapped to [-1, 1]


def cp_to_value(cp: float) -> float:
    return math.tanh(max(-MATE_CP, min(MATE_CP, cp)) / CP_SCALE)


def split_of(event: str, val_pct: int) -> str:
    return "val" if zlib.crc32(event.encode("utf-8")) % 100 < val_pct else "train"


def read_jsonl_zst(path: str) -> Iterator[dict]:
    with open(path, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_zst(path: str, records: Iterable[dict]) -> int:
    payload = "\n".join(json.dumps(r, separators=(",", ":")) for r in records)
    n = payload.count("\n") + 1 if payload else 0
    with open(path, "wb") as fh:
        fh.write(zstd.ZstdCompressor(level=6).compress((payload + "\n").encode("utf-8")))
    return n


def list_shards(directory: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(directory, "shard-*.jsonl.zst")))
    if not paths:
        raise FileNotFoundError(f"no shard-*.jsonl.zst files in {directory}")
    return paths


class ShardWriter:

    def __init__(self, directory: str, games_per_shard: int = 1000):
        os.makedirs(directory, exist_ok=True)
        self.dir, self.per, self.buf, self.index, self.shards = directory, games_per_shard, [], 0, []

    def add(self, rec: dict) -> None:
        self.buf.append(rec)
        if len(self.buf) >= self.per:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        name = f"shard-{self.index:05d}.jsonl.zst"
        write_jsonl_zst(os.path.join(self.dir, name), self.buf)
        self.shards.append({"file": name, "games": len(self.buf),
                            "plies": sum(len(r["moves"].split()) for r in self.buf)})
        self.buf, self.index = [], self.index + 1

    def close(self, extra: dict | None = None) -> dict:
        self.flush()
        manifest = {"games": sum(s["games"] for s in self.shards),
                    "plies": sum(s["plies"] for s in self.shards), "shards": self.shards, **(extra or {})}
        with open(os.path.join(self.dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest
