from __future__ import annotations

import random

import chess
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from pretrain.common import cp_to_value, list_shards, read_jsonl_zst, split_of
from pretrain.project import load_project


class PositionStream(IterableDataset):
    def __init__(self, shard_dir: str, split: str, val_pct: int = 3, *, demo: bool | None = None,
                 policy_min_elo: int = 0, sf_value_weight: float = 0.7, shuffle_buffer: int = 8192,
                 open_games: int = 32, seed: int = 0, shuffle: bool = True):
        self.paths = list_shards(shard_dir)
        self.split, self.val_pct, self.demo = split, val_pct, demo
        self.policy_min_elo, self.sf_value_weight = policy_min_elo, sf_value_weight
        self.shuffle_buffer, self.open_games, self.seed, self.shuffle = shuffle_buffer, open_games, seed, shuffle
        self.epoch = 0
        self.skipped = 0

    def set_epoch(self, epoch: int) -> "PositionStream":
        self.epoch = epoch
        return self

    # bookkeeping
    def count_positions(self) -> int:
        return sum(len(r["moves"].split()) for p in self.paths for r in read_jsonl_zst(p)
                   if split_of(r["event"], self.val_pct) == self.split)

    # sample construction
    def _game_samples(self, rec: dict, proj):
        board, z_white = chess.Board(), rec["result"]
        sf = rec.get("sf")
        for p, uci in enumerate(rec["moves"].split()):
            move, stm = chess.Move.from_uci(uci), board.turn
            legal = proj.legal_moves_map(board, stm)
            inverse = {m: i for i, m in legal.items()}
            action = inverse.get(move)
            if action is None:                               # move not representable by the action space
                self.skipped += 1
                board.push(move)
                continue
            ids = sorted(legal)
            z = z_white if stm == chess.WHITE else -z_white
            value, sf_target = z, -1
            if sf is not None:
                cp, best = sf["cp"][p], sf["best"][p]
                if cp is not None and self.sf_value_weight > 0:
                    value = (1 - self.sf_value_weight) * z + self.sf_value_weight * cp_to_value(cp)
                if best:
                    bi = inverse.get(chess.Move.from_uci(best))
                    sf_target = ids.index(bi) if bi is not None else -1
            elo = rec["w_elo"] if stm == chess.WHITE else rec["b_elo"]
            yield {"state": proj.encode(board, stm)[0].numpy().astype(np.float16),
                   "legal": np.asarray(ids, dtype=np.int16), "target": ids.index(action), "sf_target": sf_target,
                   "value": float(value), "z": float(z), "pol_w": 1.0 if elo >= self.policy_min_elo else 0.0}
            board.push(move)

    def _games(self, paths, rng):
        order = list(paths)
        if self.shuffle:
            rng.shuffle(order)
        for path in order:
            recs = [r for r in read_jsonl_zst(path) if split_of(r["event"], self.val_pct) == self.split]
            if self.shuffle:
                rng.shuffle(recs)
            yield from recs

    def __iter__(self):
        proj = load_project(self.demo)
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 7919 + self.epoch * 104729 + wid)
        games, active, buf, exhausted = self._games(self.paths[wid::nw], rng), [], [], False
        while True:
            while not exhausted and len(active) < self.open_games:
                try:
                    active.append(self._game_samples(next(games), proj))
                except StopIteration:
                    exhausted = True
            if not active:
                break
            i = rng.randrange(len(active))
            try:
                sample = next(active[i])
            except StopIteration:
                active[i] = active[-1]
                active.pop()
                continue
            if len(buf) < self.shuffle_buffer:
                buf.append(sample)
            else:
                j = rng.randrange(len(buf))
                yield buf[j]
                buf[j] = sample
        rng.shuffle(buf)
        yield from buf


def collate(batch: list[dict]) -> dict:
    width = max(len(b["legal"]) for b in batch)
    legal = np.full((len(batch), width), -1, dtype=np.int16)
    for i, b in enumerate(batch):
        legal[i, :len(b["legal"])] = b["legal"]
    t = lambda key, dt: torch.tensor([b[key] for b in batch], dtype=dt)
    return {"state": torch.from_numpy(np.stack([b["state"] for b in batch])), "legal": torch.from_numpy(legal),
            "target": t("target", torch.long), "sf_target": t("sf_target", torch.long),
            "value": t("value", torch.float32), "z": t("z", torch.float32), "pol_w": t("pol_w", torch.float32)}
