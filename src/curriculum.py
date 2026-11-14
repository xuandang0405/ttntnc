import random

import chess

from concurrent_mode.concurrent_config import RANDOM_OPENING_PLIES
from config import MAX_PLIES
from encoding import material_balance
from strategy_config import CURRICULUM_WEIGHTS, ENDGAME_MAX_PLIES, MIDDLEGAME_PLIES

KINDS = ("opening", "middlegame", "endgame")
_PIECE = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT,
          "P": chess.PAWN}

ENDGAME_TEMPLATES = ["KQvK", "KRvK", "KRRvK", "KPvK", "KPPvK", "KBNvK", "KQvKR", "KRPvKR",
                     "KQPvKQ", "KRvKB", "KRvKN"]


def _random_playout(plies, rng):
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if board.outcome(claim_draw=True) is not None:      # never start an already-ended game
            board.pop()
            break
    return board


def opening_position(rng=random):
    return _random_playout(rng.randint(*RANDOM_OPENING_PLIES), rng)


def middlegame_position(rng=random):
    board = None
    for _ in range(20):
        board = _random_playout(rng.randint(*MIDDLEGAME_PLIES), rng)
        if abs(material_balance(board, chess.WHITE)) <= 6:   # skip hopelessly lopsided positions
            break
    return board


def endgame_position(template=None, strong=None, rng=random):
    template = template or rng.choice(ENDGAME_TEMPLATES)
    strong = rng.choice([chess.WHITE, chess.BLACK]) if strong is None else strong
    mine, theirs = template.split("v")
    pieces = [(ch, strong) for ch in mine] + [(ch, not strong) for ch in theirs]
    for _ in range(500):
        board = chess.Board(None)
        for (ch, color), sq in zip(pieces, rng.sample(range(64), len(pieces))):
            board.set_piece_at(sq, chess.Piece(_PIECE[ch], color))
        board.turn = strong
        if board.pawns & (chess.BB_RANK_1 | chess.BB_RANK_8):
            continue
        if board.is_valid() and board.outcome() is None:
            return board
    return chess.Board()


def choose_kind(rng=random):
    kinds = [k for k in KINDS if CURRICULUM_WEIGHTS.get(k, 0) > 0]
    if not kinds:
        return "opening"
    return rng.choices(kinds, weights=[CURRICULUM_WEIGHTS[k] for k in kinds])[0]


def new_start(kind=None, rng=random):
    # Returns (kind, board) for a new training game.
    kind = kind or choose_kind(rng)
    board = {"opening": opening_position, "middlegame": middlegame_position,
             "endgame": endgame_position}[kind](rng=rng)
    return kind, board


def max_plies(kind):
    return ENDGAME_MAX_PLIES if kind == "endgame" else MAX_PLIES