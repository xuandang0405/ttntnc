import json
import math
import multiprocessing
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

from encoding import ACTION_SIZE, encode, legal_moves_map, move_to_index
from stockfish.stockfish_config import (TEACHER_BATCH, TEACHER_EPOCHS, TEACHER_HOLDOUT, TEACHER_MAX_PLIES,
                              TEACHER_MOVE_TEMP_CP, TEACHER_MULTIPV, TEACHER_NODES, TEACHER_OPP_W,
                              TEACHER_PHASE_W, TEACHER_POLICY_TEMP_CP, TEACHER_POLICY_W,
                              TEACHER_RANDOM_MOVE_PROB, TEACHER_VALUE_W)
from strategy import game_phase

FORMAT = 2        # 2: every row carries the id of the game it came from


# labelling
def _softmax(cps, temp):
    x = (np.asarray(cps, dtype=np.float64) - max(cps)) / temp
    e = np.exp(x)
    return e / e.sum()


def label_position(board, pool, nodes=TEACHER_NODES, multipv=TEACHER_MULTIPV,
                   temp_cp=TEACHER_POLICY_TEMP_CP):
    # Stockfish labels for `board`, or None if the engine returned nothing.
    lines = pool.analyse(board, nodes, multipv)
    if not lines:
        return None
    color = board.turn
    probs = _softmax([l.cp for l in lines], temp_cp)
    return {
        "lines": lines,
        "idx": [move_to_index(l.move, color) for l in lines],
        "p": probs.tolist(),
        "q": [l.value for l in lines],
        "value": lines[0].value,
        "cp": lines[0].cp,
    }


def choose_move(board, label, rng, move_temp_cp=TEACHER_MOVE_TEMP_CP,
                random_prob=TEACHER_RANDOM_MOVE_PROB):
    # Next move of a generated game: mostly Stockfish-like, sometimes random, for variety. 
    if rng.random() < random_prob:
        return rng.choice(list(board.legal_moves))
    probs = _softmax([l.cp for l in label["lines"]], move_temp_cp)
    return label["lines"][_pick(probs, rng)].move


def _pick(probs, rng):
    r, acc = rng.random(), 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1

class _Rows:
    # Accumulates labelled positions.

    def __init__(self, multipv):
        self.k = multipv
        self.states, self.legal, self.idx, self.p, self.q = [], [], [], [], []
        self.value, self.phase, self.cp, self.game = [], [], [], []

    def add(self, board, label, game=0):
        color = board.turn
        self.states.append(encode(board, color)[0].numpy().astype(np.float16))
        self.legal.append(np.fromiter(legal_moves_map(board, color).keys(), dtype=np.int16))
        pad = self.k - len(label["idx"])
        self.idx.append(label["idx"] + [-1] * pad)
        self.p.append(label["p"] + [0.0] * pad)
        self.q.append(label["q"] + [0.0] * pad)
        self.value.append(label["value"])
        self.phase.append(game_phase(board))
        self.cp.append(label["cp"])
        self.game.append(game)

    def __len__(self):
        return len(self.states)

    def arrays(self):
        width = max((len(l) for l in self.legal), default=1)
        legal = np.full((len(self.legal), width), -1, dtype=np.int16)
        for r, l in enumerate(self.legal):
            legal[r, :len(l)] = l
        return {"states": np.stack(self.states) if self.states else np.zeros((0, 1), np.float16),
                "legal": legal, "pol_idx": np.array(self.idx, dtype=np.int16),
                "pol_p": np.array(self.p, dtype=np.float32), "pol_q": np.array(self.q, dtype=np.float32),
                "value": np.array(self.value, dtype=np.float32),
                "phase": np.array(self.phase, dtype=np.int8), "cp": np.array(self.cp, dtype=np.float32),
                "game": np.array(self.game, dtype=np.int32)}


# generation (worker processes)
_WORKER = {}

def _close_worker():
    pid = os.getpid()
    print(f"[WORKER {pid}] cleanup started", flush=True)
    pool = _WORKER.pop("pool", None)
    if pool is None:
        print(f"[WORKER {pid}] no pool found", flush=True)
        return

    try:
        print(f"[WORKER {pid}] closing StockfishPool...", flush=True)
        pool.close()
        print(f"[WORKER {pid}] StockfishPool closed", flush=True)
    except BaseException as exc:
        print(f"[WORKER {pid}] cleanup error: {exc!r}", flush=True)
    finally:
        print(f"[WORKER {pid}] cleanup completed", flush=True)


def _init_worker(path, hash_mb):
    from stockfish.stockfish_engine import StockfishPool

    pid = os.getpid()
    print(f"[WORKER {pid}] initializing...", flush=True)
    torch.set_num_threads(1)

    pool = StockfishPool(
        path,
        engines=1,
        threads=1,
        hash_mb=hash_mb,
    )
    _WORKER["pool"] = pool

    # ProcessPoolExecutor exits via threading._shutdown before os._exit, so close
    # engine loop threads before Python waits for the worker's non-daemon threads.
    threading._register_atexit(_close_worker)
    print(f"[WORKER {pid}] initialized", flush=True)

def _play_games(task):
    import curriculum
    seed, games, nodes, multipv, max_plies, first_game = task
    rng, rows, pool = random.Random(seed), _Rows(multipv), _WORKER["pool"]
    for game in range(first_game, first_game + games):
        kind, board = curriculum.new_start(rng=rng)
        board = board.copy(stack=True)
        limit = min(max_plies, curriculum.max_plies(kind))
        for _ply in range(limit):
            if board.outcome(claim_draw=True) is not None:
                break
            label = label_position(board, pool, nodes, multipv)
            if label is None:
                break
            rows.add(board, label, game)
            board.push(choose_move(board, label, rng))
    return rows.arrays()


def generate_dataset(path, games, workers=2, nodes=TEACHER_NODES, multipv=TEACHER_MULTIPV,
                     max_plies=TEACHER_MAX_PLIES, seed=0, stockfish_path=None, log=print,
                     compress=True):
    from stockfish.stockfish_engine import find_stockfish
    from stockfish.stockfish_config import STOCKFISH_HASH_MB

    games, workers = int(games), max(1, int(workers))
    nodes, multipv, max_plies = int(nodes), int(multipv), int(max_plies)
    if games < 1:
        raise ValueError("games must be >= 1")
    if nodes < 1 or multipv < 1 or max_plies < 1:
        raise ValueError("nodes, multipv and max_plies must all be >= 1")

    exe = find_stockfish(stockfish_path)
    workers = min(workers, games)
    chunk = max(1, min(25, math.ceil(games / (workers * 4))))
    tasks, left, task_id = [], games, 0
    while left > 0:
        count = min(chunk, left)
        # the last field numbers this chunk's games, so game ids are unique across the whole dataset
        tasks.append((seed * 100_003 + task_id, count, nodes, multipv, max_plies, games - left))
        left -= count
        task_id += 1

    parts, positions = [], 0
    started = time.time()
    log(f"[GENERATE] games={games}, workers={workers}, chunks={len(tasks)}, "
        f"nodes={nodes}, MultiPV={multipv}, max_plies={max_plies}")

    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(exe, STOCKFISH_HASH_MB),
    )
    futures = []
    shutdown_started = False
    try:
        futures = [executor.submit(_play_games, task) for task in tasks]
        for future in as_completed(futures):
            part = future.result()
            parts.append(part)
            count = len(part["value"])
            positions += count
            log(f"[GENERATE] {len(parts)}/{len(tasks)} chunks; "
                f"{positions:,} positions; {time.time() - started:.1f}s")

        log("[WORKERS] all chunks returned; requesting worker shutdown...")
        shutdown_started = True
        _WORKER.clear()  # avoid double-close in the main process
        executor.shutdown(wait=True, cancel_futures=True)

        log(f"[WORKERS] shutdown complete in {time.time() - started:.1f}s")

    except BaseException:
        log("[WORKERS] generation failed; cancelling pending chunks")

        for future in futures:
            future.cancel()
        if not shutdown_started:
            executor.shutdown(wait=True, cancel_futures=True)
        raise

    log("[MERGE] allocating final arrays and copying chunk data...")
    merge_started = time.time()
    data = merge(parts)
    log(f"[MERGE] complete: {len(data['value']):,} rows in "
        f"{time.time() - merge_started:.1f}s")

    data["meta"] = json.dumps({
        "format": FORMAT, "games": games, "nodes": nodes, "multipv": multipv,
        "max_plies": max_plies, "seed": seed, "workers": workers,
        "policy_temp_cp": TEACHER_POLICY_TEMP_CP, "engine": exe,
    })

    output = os.path.abspath(path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temp_path = output + ".tmp"
    save_started = time.time()
    log(f"[SAVE] writing {'compressed' if compress else 'uncompressed'} NPZ: {output}")
    try:
        # Pass a file handle so NumPy does not append an unexpected .npz suffix.
        with open(temp_path, "wb") as stream:
            if compress:
                np.savez_compressed(stream, **data)
            else:
                np.savez(stream, **data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, output)
    except BaseException:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise

    size_mb = os.path.getsize(output) / (1024 * 1024)
    log(f"[SAVE] complete: {size_mb:.1f} MiB in {time.time() - save_started:.1f}s")
    log(f"[DONE] {len(data['value']):,} labelled positions; total {time.time() - started:.1f}s")
    return len(data["value"])


def merge(parts):
    # Merge worker arrays with one destination allocation per field.
    parts = [part for part in parts if len(part["value"])]
    if not parts:
        raise RuntimeError("no positions were generated")

    total = sum(len(part["value"]) for part in parts)
    keys = tuple(parts[0].keys())
    out = {}
    for key in keys:
        if key == "legal":
            width = max(part[key].shape[1] for part in parts)
            target = np.full((total, width), -1, dtype=np.int16)
            offset = 0
            for part in parts:
                arr = part[key]
                end = offset + len(arr)
                target[offset:end, :arr.shape[1]] = arr
                offset = end
        else:
            first = parts[0][key]
            target = np.empty((total,) + first.shape[1:], dtype=first.dtype)
            offset = 0
            for part in parts:
                arr = part[key]
                if arr.shape[1:] != first.shape[1:]:
                    raise ValueError(f"inconsistent shape for {key}: {arr.shape} vs {first.shape}")
                end = offset + len(arr)
                target[offset:end] = arr
                offset = end
        out[key] = target
    return out


# dataset
def infer_game_ids(states, chunk=20_000):
    n = len(states)
    same = np.zeros(max(n - 1, 0), dtype=bool)
    for lo in range(0, n - 1, chunk):
        hi = min(lo + chunk, n - 1)
        cur = states[lo:hi, 24:36]
        swapped = np.concatenate([cur[:, 6:], cur[:, :6]], axis=1)[:, :, ::-1, :]
        same[lo:hi] = (swapped == states[lo + 1:hi + 1, 12:24]).reshape(hi - lo, -1).all(axis=1)
    return np.concatenate([[0], np.cumsum(~same)]).astype(np.int64)[:n]


def position_keys(states):
    """One hash per row for 'is this the same position?' (piece placement, side to move's view, castling)."""
    n = len(states)
    board = (states[:, 24:36].reshape(n, -1) > 0.5).astype(np.uint8)
    castling = (states[:, 36:40, 0, 0] > 0.5).astype(np.uint8)
    packed = np.packbits(np.concatenate([board, castling], axis=1), axis=1)
    return np.fromiter((hash(row.tobytes()) for row in packed), dtype=np.int64, count=n)


class TeacherData:
    def __init__(self, arrays):
        self.a = arrays
        self.n = len(arrays["value"])

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files if k != "meta"})

    @property
    def game_ids(self):
        if getattr(self, "_game_ids", None) is None:
            recorded = "game" in self.a
            self._game_ids = self.a["game"].astype(np.int64) if recorded else infer_game_ids(self.a["states"])
            self.game_source = "recorded" if recorded else "inferred"
        return self._game_ids

    @property
    def position_keys(self):
        if getattr(self, "_position_keys", None) is None:
            self._position_keys = position_keys(self.a["states"])
        return self._position_keys

    def split(self, holdout=TEACHER_HOLDOUT, seed=0):
        groups = self.game_ids
        n_games = int(np.unique(groups).size)
        self.split_info = {"games": n_games, "source": self.game_source, "removed": 0,
                           "train": self.n, "test": 0}
        if holdout <= 0 or n_games < 2 or self.n <= 10:
            return np.arange(self.n), np.arange(0)
        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
        train, test = next(splitter.split(np.zeros(self.n), groups=groups))
        keys = self.position_keys
        leaked = np.isin(keys[train], keys[test])
        train = train[~leaked]
        self.split_info.update({"train": len(train), "test": len(test), "removed": int(leaked.sum()),
                                "train_games": int(np.unique(groups[train]).size),
                                "test_games": int(np.unique(groups[test]).size)})
        return train, test

    def describe_split(self):
        i = getattr(self, "split_info", None)
        if not i:
            return "no split computed"
        if not i["test"]:
            return f"no held-out set ({i['games']} games, {i['source']} ids)"
        return (f"{i['train']:,} train positions / {i['test']:,} held-out positions "
                f"from {i['train_games']} / {i['test_games']} separate games ({i['source']} game ids); "
                f"{i['removed']:,} training positions removed because they also occur in the held-out games")

    def batch(self, rows, device):
        a, n = self.a, len(rows)
        s = torch.from_numpy(a["states"][rows]).float().to(device)
        legal = a["legal"][rows].astype(np.int64)
        r, c = np.nonzero(legal >= 0)
        mask = np.zeros((n, ACTION_SIZE), dtype=bool)
        mask[r, legal[r, c]] = True
        idx = a["pol_idx"][rows].astype(np.int64)
        r, c = np.nonzero(idx >= 0)
        policy = np.zeros((n, ACTION_SIZE), dtype=np.float32)
        policy[r, idx[r, c]] = a["pol_p"][rows][r, c]
        qval = np.zeros((n, ACTION_SIZE), dtype=np.float32)
        qmask = np.zeros((n, ACTION_SIZE), dtype=bool)
        qval[r, idx[r, c]] = a["pol_q"][rows][r, c]
        qmask[r, idx[r, c]] = True
        t = lambda x, **k: torch.from_numpy(x).to(device, **k)
        return {"s": s, "mask": t(mask), "policy": t(policy), "q": t(qval), "qmask": t(qmask),
                "best": torch.from_numpy(idx[:, 0]).to(device),
                "value": t(a["value"][rows]), "phase": torch.from_numpy(a["phase"][rows].astype(np.int64)).to(device)}


# pretraining
def _policy_loss(logits, b):
    logp = F.log_softmax(logits.masked_fill(~b["mask"], float("-inf")), dim=-1).masked_fill(~b["mask"], 0.0)
    return -(b["policy"] * logp).sum(-1).mean()


def _q_loss(q, b):
    labelled = F.smooth_l1_loss(q[b["qmask"]], b["q"][b["qmask"]])
    floor = b["q"].masked_fill(~b["qmask"], float("inf")).min(dim=1).values
    unlabelled = b["mask"] & ~b["qmask"]
    over = F.relu(q - floor[:, None]).masked_fill(~unlabelled, 0.0)
    bound = ((over ** 2).sum(1) / unlabelled.sum(1).clamp_min(1)).mean()
    return labelled + 5.0 * bound


@torch.no_grad()
def evaluate(agent, data, rows, batch=512):
    # Agreement with Stockfish on positions the model was not trained on. 
    net, was_training = agent.net, agent.net.training
    net.eval()
    top1 = top3 = n = 0
    abs_err, preds, targets, pol = 0.0, [], [], 0.0
    for i in range(0, len(rows), batch):
        b = data.batch(rows[i:i + batch], agent.device)
        logits, values = net(b["s"])
        masked = logits.masked_fill(~b["mask"], float("-inf"))
        best3 = masked.topk(3, dim=1).indices
        top1 += int((best3[:, 0] == b["best"]).sum())
        top3 += int((best3 == b["best"][:, None]).any(dim=1).sum())
        if agent.algorithm != "dqn":
            abs_err += float((values - b["value"]).abs().sum())
            preds.append(values.cpu())
            targets.append(b["value"].cpu())
            pol += float(_policy_loss(logits, b)) * len(b["value"])
        n += len(b["value"])
    net.train(was_training)
    out = {"top1": top1 / max(n, 1), "top3": top3 / max(n, 1)}
    if agent.algorithm != "dqn" and n:
        p, t = torch.cat(preds), torch.cat(targets)
        corr = float(torch.corrcoef(torch.stack([p, t]))[0, 1]) if n > 2 and p.std() > 0 else 0.0
        out.update({"value_mae": abs_err / n, "value_corr": corr, "policy_ce": pol / n})
    return out


def pretrain(agent, data, epochs=TEACHER_EPOCHS, batch_size=TEACHER_BATCH, lr=None,
             policy_w=TEACHER_POLICY_W, value_w=TEACHER_VALUE_W, opp_w=TEACHER_OPP_W,
             phase_w=TEACHER_PHASE_W, holdout=TEACHER_HOLDOUT, seed=0, log=print):
    # Supervised training of `agent.net` on Stockfish labels. Returns per-epoch metrics.
    train_rows, test_rows = data.split(holdout, seed)
    log(f"split: {data.describe_split()}")
    rng = np.random.default_rng(seed + 1)
    old_lr = agent.opt.param_groups[0]["lr"]
    if lr is not None:
        for g in agent.opt.param_groups:
            g["lr"] = lr
    dqn = agent.algorithm == "dqn"
    history, steps = [], 0
    if len(test_rows):
        log(f"before training: {_fmt(evaluate(agent, data, test_rows))}")
    agent.net.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(train_rows)
        total, count, started = 0.0, 0, time.time()
        for i in range(0, len(order), batch_size):
            b = data.batch(order[i:i + batch_size], agent.device)
            logits, values = agent.net(b["s"])
            if dqn:
                loss = policy_w * _q_loss(logits, b)
            else:
                loss = policy_w * _policy_loss(logits, b) + value_w * F.mse_loss(values, b["value"])
            opp_logits, phase_logits = agent.net.forward_aux(b["s"])
            loss = loss + opp_w * _policy_loss(opp_logits, b) + phase_w * F.cross_entropy(phase_logits, b["phase"])
            agent._apply(loss)
            total += float(loss.detach()) * len(b["value"])
            count += len(b["value"])
            steps += 1
        entry = {"epoch": epoch, "train_loss": total / max(count, 1), "seconds": time.time() - started}
        if len(test_rows):
            entry.update(evaluate(agent, data, test_rows))
        history.append(entry)
        log(f"epoch {epoch}/{epochs}: loss {entry['train_loss']:.4f}  {_fmt(entry)}")
    for g in agent.opt.param_groups:
        g["lr"] = old_lr
    agent.last_loss = history[-1]["train_loss"] if history else 0.0
    if dqn:
        agent.target.load_state_dict(agent.net.state_dict())      # keep the target network in sync
    agent.strategy.train_steps += steps                           # the opponent head has been trained
    agent.strategy.samples_seen += len(train_rows)
    return history


def _fmt(m):
    keys = ("top1", "top3", "value_mae", "value_corr", "policy_ce")
    return "  ".join(f"{k} {m[k]:.3f}" for k in keys if k in m)
