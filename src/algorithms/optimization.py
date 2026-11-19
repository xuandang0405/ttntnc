"""Bounded chess-state keys, replay deduplication and selective alpha-beta search."""
from collections import OrderedDict
import hashlib
import chess
import chess.polyglot
import numpy as np
import torch

def position_key(board):
    # Polyglot key covers piece placement, side to move, castling and EP.
    # Halfmove clock is included because it changes draw-claim semantics.
    return (chess.polyglot.zobrist_hash(board), int(board.halfmove_clock))


def transition_key(item):
    h = hashlib.blake2b(digest_size=16)
    for value in item:
        if isinstance(value, np.ndarray):
            h.update(value.tobytes())
        else:
            h.update(repr(value).encode("utf-8"))
            h.update(b"|")
    return h.digest()


def encoded_state_key(board):
    # Mirrors the temporal/meta context consumed by the network without
    # allocating a full input tensor just to form a cache key.
    current = board.copy(stack=True)
    hashes = [chess.polyglot.zobrist_hash(current)]
    for _ in range(2):
        if current.move_stack:
            current.pop()
            hashes.append(chess.polyglot.zobrist_hash(current))
        else:
            hashes.append(0)
    hashes.reverse()
    from encoding import _repetition_count
    return (tuple(hashes), int(board.halfmove_clock),
            int(_repetition_count(board)), min(int(board.fullmove_number), 100))


class BoundedCache:
    def __init__(self, capacity):
        self.capacity = max(0, int(capacity))
        self.data = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key not in self.data:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        if self.capacity <= 0:
            return
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.capacity:
            self.data.popitem(last=False)


@torch.no_grad()
def alpha_beta_move(net, board, device, depth=2, candidates=4, perspective=None,
                    root_candidates=None, material_eval=False):
    """Selective negamax alpha-beta using the network value head at leaves."""
    from encoding import encode, legal_moves_map
    color = board.turn
    root_moves = legal_moves_map(board, color)
    if not root_moves:
        return None
    if root_candidates:
        allowed = set(root_candidates)
        root_moves = {idx: move for idx, move in root_moves.items() if move in allowed}
    idxs = list(root_moves)
    if not idxs:
        return None
    logits, _ = net(encode(board, color).to(device))
    k = min(max(1, int(candidates)), len(idxs))
    order = logits[0, idxs].topk(k).indices.tolist()
    ordered_moves = [root_moves[idxs[j]] for j in order]
    table = {}

    def evaluate(b):
        if b.is_checkmate():
            return -1.0
        if b.is_game_over(claim_draw=True):
            return 0.0
        if material_eval and perspective is not None:
            from encoding import material_balance
            v = float(material_balance(b, perspective)) / 39.0
            return v if b.turn == perspective else -v
        _, value = net(encode(b, b.turn).to(device))
        v = float(value.reshape(-1)[0])
        if perspective is not None and b.turn != perspective:
            v = -v
        return v

    def search(b, remaining, alpha, beta):
        if b.is_game_over(claim_draw=True):
            if b.is_checkmate():
                return -1.0
            return 0.0
        if remaining <= 0:
            return evaluate(b)
        # Include move history: two identical boards can have different future
        # repetition outcomes. This keeps TT scores history-correct.
        key = (chess.polyglot.zobrist_hash(b), b.halfmove_clock, remaining,
               tuple(move.uci() for move in b.move_stack))
        cached = table.get(key)
        if cached is not None:
            return cached
        best = -float("inf")
        cutoff = False
        for move in b.legal_moves:
            b.push(move)
            score = -search(b, remaining - 1, -beta, -alpha)
            b.pop()
            if score > best:
                best = score
            alpha = max(alpha, score)
            if alpha >= beta:
                cutoff = True
                break
        # Store only exact values; a cutoff result is merely a bound.
        if not cutoff:
            table[key] = best
        return best

    best_move, best_score = ordered_moves[0], -float("inf")
    alpha, beta = -float("inf"), float("inf")
    for move in ordered_moves:
        board.push(move)
        score = -search(board, max(0, int(depth) - 1), -beta, -alpha)
        board.pop()
        if score > best_score:
            best_move, best_score = move, score
        alpha = max(alpha, score)
    return best_move
