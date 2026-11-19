import math
import random
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F

from algorithms.algo_base import BaseAgent
from algorithms.algo_config import (AZ_BATCH, AZ_BUFFER, AZ_C_PUCT, AZ_DIRICHLET_ALPHA, AZ_MIN_SAMPLES, AZ_NOISE_FRAC,
                         AZ_PLAY_SIMULATIONS, AZ_SIMULATIONS, AZ_TEMP_PLIES, AZ_TRAIN_STEPS, AZ_USE_RUST_MCTS,
                         AZ_USE_RUST_SEARCH)
from config import LR
from encoding import ACTION_SIZE, encode, legal_moves_map, material_balance

try:
    import algorithms_rs
except ImportError:
    algorithms_rs = None

# MCTS
class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "moves")

    def __init__(self, prior):
        self.prior, self.visits, self.value_sum = prior, 0, 0.0
        self.children, self.moves = None, None      # idx -> Node, idx -> chess.Move

@torch.no_grad()
def _evaluate(net, boards, device, timings):
    started = time.perf_counter()
    states_list = [encode(b, b.turn) for b in boards]
    timings["mcts_encode_only"] = (
        timings.get("mcts_encode_only", 0.0)
        + time.perf_counter() - started
    )

    started = time.perf_counter()
    maps = [legal_moves_map(b, b.turn) for b in boards]
    timings["mcts_legal_map"] = (
        timings.get("mcts_legal_map", 0.0)
        + time.perf_counter() - started
    )

    started = time.perf_counter()
    mask = np.zeros((len(boards), ACTION_SIZE), dtype=bool)
    for i, m in enumerate(maps):
        if m:
            mask[i, list(m.keys())] = True

    states = torch.cat(states_list).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)

    timings["mcts_batch_prepare"] = (
        timings.get("mcts_batch_prepare", 0.0)
        + time.perf_counter() - started
    )

    started = time.perf_counter()
    logits, values = net(states)
    logits = logits.masked_fill(~mask_tensor, float("-inf"))

    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    values = values.cpu().numpy()

    timings["mcts_network_inference"] = (
        timings.get("mcts_network_inference", 0.0)
        + time.perf_counter() - started
    )

    return maps, probs, values


def _action_priors(moves, probs, noise_frac=0.0):
    idxs = list(moves.keys())
    p = probs[idxs]
    if noise_frac:
        p = (1 - noise_frac) * p + noise_frac * np.random.dirichlet([AZ_DIRICHLET_ALPHA] * len(idxs))
    return idxs, p


def _expand(node, moves, probs, noise_frac=0.0):
    idxs, p = _action_priors(moves, probs, noise_frac)
    node.moves = moves
    node.children = {i: Node(float(pi)) for i, pi in zip(idxs, p)}


def _select(node):
    sqrt_n = math.sqrt(node.visits + 1)
    best_idx, best_child, best_score = None, None, -1e9
    for idx, c in node.children.items():
        q = -c.value_sum / c.visits if c.visits else 0.0     # child values are from the opponent's view
        score = q + AZ_C_PUCT * c.prior * sqrt_n / (1 + c.visits)
        if score > best_score:
            best_idx, best_child, best_score = idx, c, score
    return best_idx, best_child

def _backup(path, value):
    for node in reversed(path):          # value flips sign at every ply
        node.visits += 1
        node.value_sum += value
        value = -value


@torch.no_grad()
def _evaluate_encoded(net, states, legal, device, timings):
    """Run the network on states already encoded by Rust.

    `legal` is (N, L) int32 of legal action indices padded with -1. Only the
    logits of legal actions are gathered and soft-maxed, so no 20480-wide
    masked softmax is needed. Returns priors (N, L) float32 and values (N,).
    """
    started = time.perf_counter()
    x = torch.from_numpy(states).to(device)
    idx = torch.from_numpy(legal).to(device).long()
    pad = idx < 0
    idx.clamp_(min=0)
    timings["mcts_batch_prepare"] = (
        timings.get("mcts_batch_prepare", 0.0) + time.perf_counter() - started
    )

    started = time.perf_counter()
    logits, values = net(x)
    chosen = logits.gather(1, idx).masked_fill(pad, float("-inf"))
    priors = torch.softmax(chosen, dim=-1)
    packed = torch.cat([priors, values.unsqueeze(1)], dim=1).float().cpu().numpy()   # one sync
    timings["mcts_network_inference"] = (
        timings.get("mcts_network_inference", 0.0) + time.perf_counter() - started
    )
    return np.ascontiguousarray(packed[:, :-1]), np.ascontiguousarray(packed[:, -1])


def _run_native_search(net, boards, sims, device, noise, timings):
    started = time.perf_counter()
    fens = [b.root().fen() for b in boards]
    histories = [[m.uci() for m in b.move_stack] for b in boards]
    search = algorithms_rs.BatchSearch(
        fens, histories, sims, AZ_C_PUCT,
        AZ_NOISE_FRAC if noise else 0.0, AZ_DIRICHLET_ALPHA,
        int(np.random.randint(0, 2 ** 62, dtype=np.int64)),
    )
    timings["mcts_batch_prepare"] = timings.get("mcts_batch_prepare", 0.0) + time.perf_counter() - started
    timings.setdefault("mcts_position_encode", 0.0)
    timings.setdefault("mcts_backup", 0.0)

    while True:
        started = time.perf_counter()
        batch = search.next_batch()
        timings["mcts_select_traverse"] = (
            timings.get("mcts_select_traverse", 0.0) + time.perf_counter() - started
        )
        if batch is None:
            break
        states, legal = batch
        priors, values = _evaluate_encoded(net, states, legal, device, timings)
        started = time.perf_counter()
        search.submit(priors, values)
        timings["mcts_expand"] = timings.get("mcts_expand", 0.0) + time.perf_counter() - started

    out = []
    for idxs, visits, ucis in search.results():
        out.append((
            list(idxs),
            np.asarray(visits, dtype=np.float64),
            {i: chess.Move.from_uci(u) for i, u in zip(idxs, ucis)},
        ))
    return out


@torch.no_grad()
def run_mcts(net, boards, sims, device, noise, profile=None):
    timings = {}
    try:
        return _run_mcts_impl(net, boards, sims, device, noise, timings)
    finally:
        if profile is not None:
            for part, elapsed in timings.items():
                profile(part, elapsed)


def _run_mcts_impl(net, boards, sims, device, noise, timings):
    if AZ_USE_RUST_SEARCH:
        if algorithms_rs is None or not hasattr(algorithms_rs, "BatchSearch"):
            raise RuntimeError(
                "Rust search was requested, but the algorithms_rs extension with BatchSearch "
                "is not installed. Build it with `python -m pip install .` from src/algorithms-rs."
            )
        return _run_native_search(net, boards, sims, device, noise, timings)
    if AZ_USE_RUST_MCTS:
        if algorithms_rs is None:
            raise RuntimeError(
                "Rust MCTS was requested, but the algorithms_rs extension is not installed. "
                "Build it with `python -m pip install .` from src/algorithms-rs."
            )
        return _run_rust_mcts(net, boards, sims, device, noise, timings)

    # Search every board in lock-step. Returns [(move indices, visit counts, idx->move)] per board.
    started = time.perf_counter()
    roots = [Node(1.0) for _ in boards]
    active = [i for i, board in enumerate(boards) if board.outcome(claim_draw=True) is None]
    timings["mcts_select_traverse"] = (
        timings.get("mcts_select_traverse", 0.0) + time.perf_counter() - started
    )
    if not active:
        return [([], np.zeros(0, dtype=np.float64), {}) for _ in boards]
    maps, probs, _ = _evaluate(net, [boards[i] for i in active], device, timings)
    started = time.perf_counter()
    for i, m, p in zip(active, maps, probs):
        _expand(roots[i], m, p, AZ_NOISE_FRAC if noise else 0.0)
    timings["mcts_expand"] = timings.get("mcts_expand", 0.0) + time.perf_counter() - started

    for _ in range(sims):
        pending = []
        started = time.perf_counter()
        backup_seconds = 0.0
        for root, board in zip(roots, boards):
            if not root.children:
                continue
            node, b, path = root, board.copy(stack=True), [root]
            while node.children:
                idx, child = _select(node)
                b.push(node.moves[idx])
                node = child
                path.append(node)
            outcome = b.outcome(claim_draw=True)
            if outcome is not None:
                terminal_value = 0.0 if outcome.winner is None else (1.0 if outcome.winner == b.turn else -1.0)
                backup_started = time.perf_counter()
                _backup(path, terminal_value)
                elapsed = time.perf_counter() - backup_started
                backup_seconds += elapsed
                timings["mcts_backup"] = (
                    timings.get("mcts_backup", 0.0) + elapsed
                )
            else:
                pending.append((path, node, b))
        timings["mcts_select_traverse"] = (
            timings.get("mcts_select_traverse", 0.0)
            + max(0.0, time.perf_counter() - started - backup_seconds)
        )
        if pending:
            maps, probs, values = _evaluate(net, [b for _, _, b in pending], device, timings)
            for (path, node, _), m, p, v in zip(pending, maps, probs, values):
                expand_started = time.perf_counter()
                _expand(node, m, p)
                timings["mcts_expand"] = (
                    timings.get("mcts_expand", 0.0) + time.perf_counter() - expand_started
                )
                backup_started = time.perf_counter()
                _backup(path, float(v))
                timings["mcts_backup"] = (
                    timings.get("mcts_backup", 0.0) + time.perf_counter() - backup_started
                )

    out = []
    for root in roots:
        idxs = list(root.children.keys()) if root.children else []
        out.append((idxs, np.array([root.children[i].visits for i in idxs], dtype=np.float64), root.moves or {}))
    return out


def _run_rust_mcts(net, boards, sims, device, noise, timings):
    started = time.perf_counter()
    trees = [algorithms_rs.MctsTree() for _ in boards]
    node_moves = [[None] for _ in boards]
    active = [i for i, board in enumerate(boards) if board.outcome(claim_draw=True) is None]
    timings["mcts_select_traverse"] = (
        timings.get("mcts_select_traverse", 0.0) + time.perf_counter() - started
    )
    if not active:
        return [([], np.zeros(0, dtype=np.float64), {}) for _ in boards]

    maps, probs, _ = _evaluate(net, [boards[i] for i in active], device, timings)
    started = time.perf_counter()
    for i, moves, policy in zip(active, maps, probs):
        indices, priors = _action_priors(
            moves, policy, AZ_NOISE_FRAC if noise else 0.0
        )
        child_ids = trees[i].expand(0, indices, priors.tolist())
        node_moves[i][0] = moves
        node_moves[i].extend([None] * len(child_ids))
    timings["mcts_expand"] = timings.get("mcts_expand", 0.0) + time.perf_counter() - started

    for _ in range(sims):
        pending = []
        started = time.perf_counter()
        backup_seconds = 0.0
        for i, board in enumerate(boards):
            tree = trees[i]
            if not tree.is_expanded(0):
                continue
            node_id, current, path = 0, board.copy(stack=True), [0]
            while tree.is_expanded(node_id):
                selection = tree.select(node_id, AZ_C_PUCT)
                if selection is None:
                    break
                action_idx, child_id = selection
                current.push(node_moves[i][node_id][action_idx])
                node_id = child_id
                path.append(node_id)
            outcome = current.outcome(claim_draw=True)
            if outcome is not None:
                terminal_value = (
                    0.0 if outcome.winner is None
                    else (1.0 if outcome.winner == current.turn else -1.0)
                )
                backup_started = time.perf_counter()
                tree.backup(path, terminal_value)
                elapsed = time.perf_counter() - backup_started
                backup_seconds += elapsed
                timings["mcts_backup"] = timings.get("mcts_backup", 0.0) + elapsed
            else:
                pending.append((i, path, node_id, current))
        timings["mcts_select_traverse"] = (
            timings.get("mcts_select_traverse", 0.0)
            + max(0.0, time.perf_counter() - started - backup_seconds)
        )

        if pending:
            maps, probs, values = _evaluate(
                net, [current for _, _, _, current in pending], device, timings
            )
            for (i, path, node_id, _), moves, policy, value in zip(
                pending, maps, probs, values
            ):
                expand_started = time.perf_counter()
                indices, priors = _action_priors(moves, policy)
                child_ids = trees[i].expand(node_id, indices, priors.tolist())
                while len(node_moves[i]) < trees[i].node_count():
                    node_moves[i].append(None)
                node_moves[i][node_id] = moves
                if child_ids:
                    node_moves[i][node_id] = moves
                timings["mcts_expand"] = (
                    timings.get("mcts_expand", 0.0) + time.perf_counter() - expand_started
                )
                backup_started = time.perf_counter()
                trees[i].backup(path, float(value))
                timings["mcts_backup"] = (
                    timings.get("mcts_backup", 0.0) + time.perf_counter() - backup_started
                )

    out = []
    for i, tree in enumerate(trees):
        visits = tree.root_visits()
        indices = [action_idx for action_idx, _ in visits]
        counts = np.asarray([count for _, count in visits], dtype=np.float64)
        out.append((indices, counts, node_moves[i][0] or {}))
    return out


class AlphaZeroAgent(BaseAgent):
    algorithm = "alphazero"
    label = "AlphaZero"

    def __init__(self, color, name, device, episodes_per_update=1):
        super().__init__(color, name, device, episodes_per_update)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR, weight_decay=1e-4)
        self.replay, self.replay_pos = [], 0
        self.pending_episodes = 0

    def hyperparameters(self):
        return {"simulations": AZ_SIMULATIONS, "play_simulations": AZ_PLAY_SIMULATIONS, "c_puct": AZ_C_PUCT,
                "dirichlet_alpha": AZ_DIRICHLET_ALPHA, "noise_frac": AZ_NOISE_FRAC, "temp_plies": AZ_TEMP_PLIES,
                "buffer": AZ_BUFFER, "batch": AZ_BATCH, "train_steps": AZ_TRAIN_STEPS,
                "episodes_per_update": self.episodes_per_update}

    def act_batch(self, game_ids, boards):
        self.prefetch_balances(boards)      # Stockfish evaluations run while MCTS searches
        results = run_mcts(
            self.net, boards, AZ_SIMULATIONS, self.device, noise=True,
            profile=self._record_profile_time,
        )
        with self._profile("action_prepare"):
            st = self.encode_boards(boards).numpy().astype(np.float16)
        with self._profile("action_select"):
            moves = []
            for i, (g, board) in enumerate(zip(game_ids, boards)):
                idxs, visits, mmap = results[i]
                if not idxs:
                    raise RuntimeError("MCTS received a terminal position with no legal moves")
                probs = visits / max(visits.sum(), 1.0)
                if len(board.move_stack) < AZ_TEMP_PLIES:
                    choice = int(np.random.choice(len(idxs), p=probs))     # temperature 1: explore
                else:
                    choice = int(np.argmax(visits))
                ep = self.eps.setdefault(g, {"s": [], "legal": [], "pi": [], "bal": []})
                ep["s"].append(st[i])
                ep["legal"].append(np.array(idxs, dtype=np.int64))
                ep["pi"].append(probs.astype(np.float32))
                ep["bal"].append(self.balance_of(board))
                moves.append(mmap[idxs[choice]])
        return moves
 
    def finish_episode(self, game_id, final_balance, terminal):
        ep = self.eps.pop(game_id, None)
        if not ep or not ep["s"]:
            return
        with self._profile("episode_prepare"):
            self._rewards(ep["bal"], final_balance, terminal)       # logged for the reward metrics
            z = float(np.clip(sum(terminal.values()), -1.0, 1.0))   # game result from this side's view
            for s, legal, pi in zip(ep["s"], ep["legal"], ep["pi"]):
                item = (s, legal, pi, z)
                if len(self.replay) < AZ_BUFFER:
                    self.replay.append(item)
                else:
                    self.replay[self.replay_pos] = item
                    self.replay_pos = (self.replay_pos + 1) % AZ_BUFFER
        self.pending_episodes += 1
        if self.pending_episodes >= self.episodes_per_update:
            self.pending_episodes = 0
            self.update()

    def update(self):
        if len(self.replay) < AZ_MIN_SAMPLES:
            return
        dev = self.device
        for _ in range(AZ_TRAIN_STEPS):
            with self._profile("replay_sample"):
                batch = random.sample(self.replay, min(AZ_BATCH, len(self.replay)))
            with self._profile("batch_prepare"):
                s = torch.from_numpy(np.stack([t[0] for t in batch])).float().to(dev)
                mask_np = self.mask_from_legal([t[1] for t in batch])
                pi_np = np.zeros((len(batch), ACTION_SIZE), dtype=np.float32)
                for r, t in enumerate(batch):
                    pi_np[r, t[1]] = t[2]
                mask = torch.from_numpy(mask_np).to(dev)
                pi = torch.from_numpy(pi_np).to(dev)
                z = torch.tensor([t[3] for t in batch], dtype=torch.float32, device=dev)
 
            with self._profile("train_forward"):
                logits, values = self.net(s)
            with self._profile("loss_compute"):
                logp = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
                safe = logp.masked_fill(~mask, 0.0)
                policy_loss = -(pi * safe).sum(-1).mean()
                value_loss = F.mse_loss(values, z)
                loss = policy_loss + value_loss
                entropy = -(logp.exp().detach() * safe.detach()).sum(-1).mean()
            grad_norm, update_norm = self._apply(loss)
            self._record(policy_loss, value_loss, loss, entropy, z - values.detach(), z, values,
                         grad_norm, update_norm)

    def select_move(self, board, greedy=False):
        idxs, visits, mmap = run_mcts(self.net, [board], AZ_PLAY_SIMULATIONS, self.device, noise=False)[0]
        if not idxs:
            raise RuntimeError("Cannot select a move from a terminal chess position")
        if greedy:
            choice = int(np.argmax(visits))
        else:
            total = float(visits.sum())
            probs = visits / total if total > 0 else np.full(len(visits), 1.0 / len(visits))
            choice = int(np.random.choice(len(idxs), p=probs))
        return mmap[idxs[choice]]