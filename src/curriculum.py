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


def _random_playout(plies, rng, starting_color=None):
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        if board.outcome(claim_draw=True) is not None:      # never start an already-ended game
            board.pop()
            break
    if starting_color is not None and board.turn != starting_color:
        safe_moves = []
        for move in board.legal_moves:
            board.push(move)
            if board.outcome(claim_draw=True) is None:
                safe_moves.append(move)
            board.pop()
        if safe_moves:
            board.push(rng.choice(safe_moves))
        else:
            board = chess.Board()
            if starting_color == chess.BLACK:
                board.push(next(iter(board.legal_moves)))
    return board


def _plies_for_start(plies_range, starting_color, rng):
    if starting_color is None:
        return rng.randint(*plies_range)
    desired_parity = 0 if starting_color == chess.WHITE else 1
    candidates = [plies for plies in range(plies_range[0], plies_range[1] + 1)
                  if plies % 2 == desired_parity]
    return rng.choice(candidates)


def opening_position(rng=random, starting_color=None):
    return _random_playout(
        _plies_for_start(RANDOM_OPENING_PLIES, starting_color, rng), rng, starting_color)


def middlegame_position(rng=random, starting_color=None):
    board = None
    for _ in range(20):
        board = _random_playout(
            _plies_for_start(MIDDLEGAME_PLIES, starting_color, rng), rng, starting_color)
        if (abs(material_balance(board, chess.WHITE)) <= 6
                and (starting_color is None or board.turn == starting_color)):
            break
    return board


def endgame_position(template=None, strong=None, rng=random, starting_color=None):
    template = template or rng.choice(ENDGAME_TEMPLATES)
    strong = rng.choice([chess.WHITE, chess.BLACK]) if strong is None else strong
    mine, theirs = template.split("v")
    pieces = [(ch, strong) for ch in mine] + [(ch, not strong) for ch in theirs]
    for _ in range(500):
        board = chess.Board(None)
        for (ch, color), sq in zip(pieces, rng.sample(range(64), len(pieces))):
            board.set_piece_at(sq, chess.Piece(_PIECE[ch], color))
        board.turn = strong if starting_color is None else starting_color
        if board.pawns & (chess.BB_RANK_1 | chess.BB_RANK_8):
            continue
        if board.is_valid() and board.outcome() is None:
            return board
    board = chess.Board()
    if starting_color is not None:
        board.turn = starting_color
    return board


def choose_kind(rng=random):
    kinds = [k for k in KINDS if CURRICULUM_WEIGHTS.get(k, 0) > 0]
    if not kinds:
        return "opening"
    return rng.choices(kinds, weights=[CURRICULUM_WEIGHTS[k] for k in kinds])[0]


def new_start(kind=None, rng=random, match_number=None):
    # Returns (kind, board) for a new training game.
    kind = kind or choose_kind(rng)
    starting_color = None
    if match_number is not None:
        starting_color = chess.WHITE if match_number % 2 else chess.BLACK
    board = {"opening": opening_position, "middlegame": middlegame_position,
             "endgame": endgame_position}[kind](
                 rng=rng, starting_color=starting_color)
    return kind, board


def max_plies(kind):
    return ENDGAME_MAX_PLIES if kind == "endgame" else MAX_PLIES