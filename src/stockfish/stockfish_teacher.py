"""Stockfish as a teacher: label positions, then pretrain a model on the labels.

Labels for every position (all from the side to move's point of view, matching how the
network sees the board):

* soft policy   - softmax over the MultiPV centipawn scores, so near-equal moves share the
                  probability mass instead of forcing a single "right" answer
* move values   - the value of each candidate move (Q-targets for DQN)
* value         - Stockfish's evaluation squashed to [-1, 1] (expected result)
* phase         - opening / middlegame / endgame, for the strategy head
"""
import json
import math
import multiprocessing
import multiprocessing.util
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from encoding import ACTION_SIZE, encode, legal_moves_map, move_to_index
from stockfish.stockfish_config import (TEACHER_BATCH, TEACHER_EPOCHS, TEACHER_HOLDOUT, TEACHER_MAX_PLIES,
                              TEACHER_MOVE_TEMP_CP, TEACHER_MULTIPV, TEACHER_NODES, TEACHER_OPP_W,
                              TEACHER_PHASE_W, TEACHER_POLICY_TEMP_CP, TEACHER_POLICY_W,
                              TEACHER_RANDOM_MOVE_PROB, TEACHER_VALUE_W)
from strategy import game_phase

FORMAT = 1


# labelling
def _softmax(cps, temp):
    x = (np.asarray(cps, dtype=np.float64) - max(cps)) / temp
    e = np.exp(x)
    return e / e.sum()


def label_position(board, pool, nodes=TEACHER_NODES, multipv=TEACHER_MULTIPV,
                   temp_cp=TEACHER_POLICY_TEMP_CP):
    """Stockfish labels for `board`, or None if the engine returned nothing."""
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
    """Next move of a generated game: mostly Stockfish-like, sometimes random, for variety."""
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
    """Accumulates labelled positions."""

    def __init__(self, multipv):
        self.k = multipv
        self.states, self.legal, self.idx, self.p, self.q = [], [], [], [], []
        self.value, self.phase, self.cp = [], [], []

    def add(self, board, label):
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
                "phase": np.array(self.phase, dtype=np.int8), "cp": np.array(self.cp, dtype=np.float32)}


# ------------------------------------------------------------------ generation (worker processes)
_WORKER = {}


def _init_worker(path, hash_mb):
    from stockfish.stockfish_engine import StockfishPool
    torch.set_num_threads(1)
    _WORKER["pool"] = StockfishPool(path, engines=1, threads=1, hash_mb=hash_mb)
    # worker processes leave through os._exit, which skips atexit: close the engine explicitly
    multiprocessing.util.Finalize(None, _WORKER["pool"].close, exitpriority=10)


def _play_games(task):
    import curriculum
    seed, games, nodes, multipv, max_plies = task
    rng, rows, pool = random.Random(seed), _Rows(multipv), _WORKER["pool"]
    for _ in range(games):
        kind, board = curriculum.new_start(rng=rng)
        board = board.copy(stack=True)
        limit = min(max_plies, curriculum.max_plies(kind))
        for _ply in range(limit):
            if board.outcome(claim_draw=True) is not None:
                break
            label = label_position(board, pool, nodes, multipv)
            if label is None:
                break
            rows.add(board, label)
            board.push(choose_move(board, label, rng))
    return rows.arrays()


def generate_dataset(path, games, workers=2, nodes=TEACHER_NODES, multipv=TEACHER_MULTIPV,
                     max_plies=TEACHER_MAX_PLIES, seed=0, stockfish_path=None, log=print):
    """Play `games` curriculum-started games guided by Stockfish and label every position."""
    from stockfish.stockfish_engine import find_stockfish
    from stockfish.stockfish_config import STOCKFISH_HASH_MB
    exe = find_stockfish(stockfish_path)
    chunk = max(1, min(25, math.ceil(games / (workers * 4))))
    tasks, left, n = [], games, 0
    while left > 0:
        tasks.append((seed * 100_003 + n, min(chunk, left), nodes, multipv, max_plies))
        left -= chunk
        n += 1
    parts, started, done = [], time.time(), 0
    # `spawn` on every platform: forking after PyTorch has started its thread pool can deadlock.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_init_worker, initargs=(exe, STOCKFISH_HASH_MB)) as executor:
        for part in executor.map(_play_games, tasks):
            parts.append(part)
            done += 1
            positions = sum(len(p["value"]) for p in parts)
            log(f"  {done}/{len(tasks)} chunks, {positions} positions, {time.time() - started:.0f}s")
    data = merge(parts)
    data["meta"] = json.dumps({"format": FORMAT, "games": games, "nodes": nodes, "multipv": multipv,
                               "policy_temp_cp": TEACHER_POLICY_TEMP_CP, "engine": exe})
    np.savez_compressed(path, **data)
    return len(data["value"])


def merge(parts):
    parts = [p for p in parts if len(p["value"])]
    if not parts:
        raise RuntimeError("no positions were generated")
    width = max(p["legal"].shape[1] for p in parts)
    out = {}
    for key in parts[0]:
        if key == "legal":
            rows = []
            for p in parts:
                pad = np.full((len(p["legal"]), width), -1, dtype=np.int16)
                pad[:, :p["legal"].shape[1]] = p["legal"]
                rows.append(pad)
            out[key] = np.concatenate(rows)
        else:
            out[key] = np.concatenate([p[key] for p in parts])
    return out


# dataset
class TeacherData:
    def __init__(self, arrays):
        self.a = arrays
        self.n = len(arrays["value"])

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files if k != "meta"})

    def split(self, holdout=TEACHER_HOLDOUT, seed=0):
        order = np.random.default_rng(seed).permutation(self.n)
        cut = max(1, int(self.n * holdout)) if self.n > 10 else 0
        return order[cut:], order[:cut]

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
    """Agreement with Stockfish on positions the model was not trained on."""
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
    """Supervised training of `agent.net` on Stockfish labels. Returns per-epoch metrics."""
    train_rows, test_rows = data.split(holdout, seed)
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
