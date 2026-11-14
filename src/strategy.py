import random

import chess
import numpy as np
import torch
import torch.nn.functional as F

from encoding import encode, legal_moves_map, move_to_index
from strategy_config import (LOOKAHEAD_CANDIDATES, LOOKAHEAD_REPLIES, LOOKAHEAD_TEMPERATURE,
                             PHASE_ENDGAME_MATERIAL, PHASE_OPENING_MOVES, STRATEGY_BATCH,
                             STRATEGY_BUFFER, STRATEGY_ENABLED, STRATEGY_EVAL_SAMPLES,
                             STRATEGY_MIN_SAMPLES, STRATEGY_MIN_STEPS, STRATEGY_PHASE_WEIGHT,
                             STRATEGY_STEPS, STRATEGY_TRAIN_EVERY_GAMES)

_VALUE = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def game_phase(board):
    # 0 = opening, 1 = middlegame, 2 = endgame (from the material left and the move number).
    total = sum(_VALUE.get(p.piece_type, 0) for p in board.piece_map().values())
    if total <= PHASE_ENDGAME_MATERIAL:
        return 2
    if board.fullmove_number <= PHASE_OPENING_MOVES and total >= 54:
        return 0
    return 1


def _mask(legal_lists):
    mask = np.zeros((len(legal_lists), 4096), dtype=bool)
    for r, legal in enumerate(legal_lists):
        mask[r, legal] = True
    return mask


class StrategyLearner:
    def __init__(self, agent):
        self.agent = agent
        self.replay, self.pos = [], 0
        self.unseen = []            # newest samples: used to measure accuracy before training on them
        self.samples_seen = 0
        self.train_steps = 0
        self._games = 0

    @property
    def ready(self):
        return self.samples_seen >= STRATEGY_MIN_SAMPLES and self.train_steps >= STRATEGY_MIN_STEPS

    def state(self):
        return {"samples_seen": self.samples_seen, "train_steps": self.train_steps}

    def load(self, state):
        if state:
            self.samples_seen = int(state.get("samples_seen", 0))
            self.train_steps = int(state.get("train_steps", 0))

    # data
    def observe(self, board, move, mover_color):
        # Record a move made by this agent's OPPONENT (call before the move is pushed).
        if not STRATEGY_ENABLED:
            return
        legal = legal_moves_map(board, mover_color)
        idx = move_to_index(move, mover_color)
        if idx not in legal:                                  # e.g. an under-promotion
            return
        state = encode(board, mover_color).numpy().astype(np.uint8)[0]
        self.unseen.append((state, idx, np.fromiter(legal.keys(), dtype=np.int64), game_phase(board)))
        self.samples_seen += 1

    def _store(self, item):
        if len(self.replay) < STRATEGY_BUFFER:
            self.replay.append(item)
        else:
            self.replay[self.pos] = item
            self.pos = (self.pos + 1) % STRATEGY_BUFFER

    def _collate(self, items):
        dev = self.agent.device
        s = torch.from_numpy(np.stack([t[0] for t in items])).float().to(dev)
        a = torch.tensor([t[1] for t in items], dtype=torch.long, device=dev)
        m = torch.from_numpy(_mask([t[2] for t in items])).to(dev)
        ph = torch.tensor([t[3] for t in items], dtype=torch.long, device=dev)
        return s, a, m, ph

    # learning
    def train(self):
        # Call after every finished game; trains every STRATEGY_TRAIN_EVERY_GAMES calls.
        if not STRATEGY_ENABLED:
            return
        self._games += 1
        if self._games < STRATEGY_TRAIN_EVERY_GAMES:
            return
        self._games = 0
        if self.unseen:
            self._evaluate(self.unseen[-STRATEGY_EVAL_SAMPLES:])
            for item in self.unseen:
                self._store(item)
            self.unseen = []
        if len(self.replay) < STRATEGY_BATCH:
            return
        for _ in range(STRATEGY_STEPS):
            self._step(random.sample(self.replay, STRATEGY_BATCH))

    @torch.no_grad()
    def _evaluate(self, items):
        # Accuracy on moves the model has not trained on yet (an honest estimate).
        s, a, m, ph = self._collate(items)
        opp, phase = self.agent.net.forward_aux(s)
        top3 = opp.masked_fill(~m, float("-inf")).topk(3, dim=1).indices
        stats = {"opp_top1": float((top3[:, 0] == a).float().mean()),
                 "opp_top3": float((top3 == a[:, None]).any(dim=1).float().mean()),
                 "phase_accuracy": float((phase.argmax(1) == ph).float().mean())}
        if self.agent.metrics:
            self.agent.metrics.record_strategy(self.agent.name, stats)

    def _step(self, items):
        s, a, m, ph = self._collate(items)
        opp, phase = self.agent.net.forward_aux(s)
        opp_loss = F.cross_entropy(opp.masked_fill(~m, float("-inf")), a)
        phase_loss = F.cross_entropy(phase, ph)
        self.agent._apply(opp_loss + STRATEGY_PHASE_WEIGHT * phase_loss)
        self.train_steps += 1
        if self.agent.metrics:
            self.agent.metrics.record_strategy(self.agent.name, {"opp_loss": float(opp_loss),
                                                                 "phase_loss": float(phase_loss)})

@torch.no_grad()
def lookahead_move(agent, board, greedy=False):
    dev, net = agent.device, agent.net
    color, opp = board.turn, not board.turn
    moves = legal_moves_map(board, color)
    idx = list(moves.keys())
    logits, _ = net(encode(board, color).to(dev))
    k = min(LOOKAHEAD_CANDIDATES, len(idx))
    cands = [moves[idx[i]] for i in logits[0, idx].topk(k).indices.tolist()]

    after, value = [], {}
    for n, m in enumerate(cands):
        b2 = board.copy(stack=False)
        b2.push(m)
        after.append(b2)
        if b2.is_game_over():                         # our move ended the game: mate wins, otherwise a draw
            value[n] = 1.0 if b2.is_checkmate() else 0.0
    live = [n for n in range(len(cands)) if n not in value]

    if live:
        opp_logits, _ = net.forward_aux(torch.cat([encode(after[n], opp) for n in live]).to(dev))
        maps = [legal_moves_map(after[n], opp) for n in live]
        mask = torch.from_numpy(_mask([list(mp.keys()) for mp in maps])).to(dev)
        probs = torch.softmax(opp_logits.masked_fill(~mask, float("-inf")), dim=-1).cpu().numpy()

        replies, owner, weight = [], [], []
        for j, n in enumerate(live):
            keys = list(maps[j].keys())
            p = probs[j, keys]
            order = np.argsort(-p)[:LOOKAHEAD_REPLIES]
            total = float(p[order].sum()) or 1.0
            for o in order:
                b3 = after[n].copy(stack=False)
                b3.push(maps[j][keys[o]])
                replies.append(b3)
                owner.append(n)
                weight.append(float(p[o]) / total)

        vals = [0.0] * len(replies)
        to_eval = []
        for t, b3 in enumerate(replies):
            if b3.is_game_over():                      # the reply mated us (-1) or drew
                vals[t] = -1.0 if b3.is_checkmate() else 0.0
            else:
                to_eval.append(t)
        if to_eval:
            _, v = net(torch.cat([encode(replies[t], color) for t in to_eval]).to(dev))
            for t, x in zip(to_eval, v.cpu().tolist()):
                vals[t] = x
        for n in live:
            value[n] = 0.0
        for t in range(len(replies)):
            value[owner[t]] += weight[t] * vals[t]

    scores = torch.tensor([value[n] for n in range(len(cands))])
    if greedy:
        return cands[int(scores.argmax())]
    return cands[int(torch.distributions.Categorical(logits=scores / LOOKAHEAD_TEMPERATURE).sample())]