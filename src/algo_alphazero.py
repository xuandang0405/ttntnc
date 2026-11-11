import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from algo_base import BaseAgent
from algo_config import (AZ_BATCH, AZ_BUFFER, AZ_C_PUCT, AZ_DIRICHLET_ALPHA, AZ_MIN_SAMPLES, AZ_NOISE_FRAC,
                         AZ_PLAY_SIMULATIONS, AZ_SIMULATIONS, AZ_TEMP_PLIES, AZ_TRAIN_STEPS)
from config import LR
from encoding import encode, legal_moves_map, material_balance

# MCTS
class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "moves")

    def __init__(self, prior):
        self.prior, self.visits, self.value_sum = prior, 0, 0.0
        self.children, self.moves = None, None      # idx -> Node, idx -> chess.Move


@torch.no_grad()
def _evaluate(net, boards, device):
    # Batch-evaluate positions from the side to move's view: (move maps, priors, values).
    states = torch.cat([encode(b, b.turn) for b in boards]).to(device)
    maps = [legal_moves_map(b, b.turn) for b in boards]
    mask = np.zeros((len(boards), 4096), dtype=bool)
    for i, m in enumerate(maps):
        mask[i, list(m.keys())] = True
    logits, values = net(states)
    logits = logits.masked_fill(~torch.from_numpy(mask).to(device), float("-inf"))
    return maps, torch.softmax(logits, dim=-1).cpu().numpy(), values.cpu().numpy()


def _expand(node, moves, probs, noise_frac=0.0):
    idxs = list(moves.keys())
    p = probs[idxs]
    if noise_frac:
        p = (1 - noise_frac) * p + noise_frac * np.random.dirichlet([AZ_DIRICHLET_ALPHA] * len(idxs))
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
def run_mcts(net, boards, sims, device, noise):
    # Search every board in lock-step. Returns [(move indices, visit counts, idx->move)] per board.
    roots = [Node(1.0) for _ in boards]
    maps, probs, _ = _evaluate(net, boards, device)
    for root, m, p in zip(roots, maps, probs):
        _expand(root, m, p, AZ_NOISE_FRAC if noise else 0.0)

    for _ in range(sims):
        pending = []
        for root, board in zip(roots, boards):
            node, b, path = root, board.copy(stack=False), [root]
            while node.children:
                idx, child = _select(node)
                b.push(node.moves[idx])
                node = child
                path.append(node)
            if b.is_game_over():
                _backup(path, -1.0 if b.is_checkmate() else 0.0)   # side to move is mated / drawn
            else:
                pending.append((path, node, b))
        if pending:
            maps, probs, values = _evaluate(net, [b for _, _, b in pending], device)
            for (path, node, _), m, p, v in zip(pending, maps, probs, values):
                _expand(node, m, p)
                _backup(path, float(v))

    out = []
    for root in roots:
        idxs = list(root.children.keys())
        out.append((idxs, np.array([root.children[i].visits for i in idxs], dtype=np.float64), root.moves))
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
        results = run_mcts(self.net, boards, AZ_SIMULATIONS, self.device, noise=True)
        st = self.encode_boards(boards).numpy().astype(np.uint8)
        moves = []
        for i, (g, board) in enumerate(zip(game_ids, boards)):
            idxs, visits, mmap = results[i]
            probs = visits / max(visits.sum(), 1.0)
            if len(board.move_stack) < AZ_TEMP_PLIES:
                choice = int(np.random.choice(len(idxs), p=probs))     # temperature 1: explore
            else:
                choice = int(np.argmax(visits))
            ep = self.eps.setdefault(g, {"s": [], "legal": [], "pi": [], "bal": []})
            ep["s"].append(st[i])
            ep["legal"].append(np.array(idxs, dtype=np.int64))
            ep["pi"].append(probs.astype(np.float32))
            ep["bal"].append(material_balance(board, self.color))
            moves.append(mmap[idxs[choice]])
        return moves
 
    def finish_episode(self, game_id, final_balance, terminal):
        ep = self.eps.pop(game_id, None)
        if not ep or not ep["s"]:
            return
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
            batch = random.sample(self.replay, min(AZ_BATCH, len(self.replay)))
            s = torch.from_numpy(np.stack([t[0] for t in batch])).float().to(dev)
            mask_np = self.mask_from_legal([t[1] for t in batch])
            pi_np = np.zeros((len(batch), 4096), dtype=np.float32)
            for r, t in enumerate(batch):
                pi_np[r, t[1]] = t[2]
            mask = torch.from_numpy(mask_np).to(dev)
            pi = torch.from_numpy(pi_np).to(dev)
            z = torch.tensor([t[3] for t in batch], dtype=torch.float32, device=dev)
 
            logits, values = self.net(s)
            logp = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
            safe = logp.masked_fill(~mask, 0.0)
            policy_loss = -(pi * safe).sum(-1).mean()
            value_loss = F.mse_loss(values, z)
            loss = policy_loss + value_loss
            entropy = -(logp.exp().detach() * safe.detach()).sum(-1).mean()
            grad_norm, update_norm = self._apply(loss)
            self._record(policy_loss, value_loss, loss, entropy, z - values.detach(), z, values,
                         grad_norm, update_norm)

    @torch.no_grad()
    def select_move(self, board, greedy=False):
        idxs, visits, mmap = run_mcts(self.net, [board], AZ_PLAY_SIMULATIONS, self.device, noise=False)[0]
        if greedy:
            choice = int(np.argmax(visits))
        else:
            choice = int(np.random.choice(len(idxs), p=visits / visits.sum()))
        return mmap[idxs[choice]]