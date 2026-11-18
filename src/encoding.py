import chess
import numpy as np
import torch

from config import PIECE_VALUE

HISTORY_LENGTH = 3
PIECE_PLANES = 12
META_PLANES = 8
INPUT_CHANNELS = HISTORY_LENGTH * PIECE_PLANES + META_PLANES
BOARD_ACTIONS = 64 * 64
PROMOTION_TYPES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMOTION_OFFSET = BOARD_ACTIONS
ACTION_SIZE = BOARD_ACTIONS * (1 + len(PROMOTION_TYPES))
_PROMOTION_TO_SLOT = {piece: i for i, piece in enumerate(PROMOTION_TYPES)}


def view_sq(sq, color):
    return sq if color == chess.WHITE else chess.square_mirror(sq)


def move_to_index(move, color):
    base = view_sq(move.from_square, color) * 64 + view_sq(move.to_square, color)
    if move.promotion is None:
        return base
    try:
        slot = _PROMOTION_TO_SLOT[move.promotion]
    except KeyError as exc:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}") from exc
    return PROMOTION_OFFSET + slot * BOARD_ACTIONS + base


def _position_history(board):
    root = board.root()
    positions = [root.copy(stack=False)]
    for move in board.move_stack:
        root.push(move)
        positions.append(root.copy(stack=False))
    return positions[-HISTORY_LENGTH:]


def _repetition_count(board):
    # python-chess performs an exact history-aware scan when the move stack exists.
    if board.is_repetition(3):
        return 3
    if board.is_repetition(2):
        return 2
    return 1


def encode(board, color):
    x = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.float32)
    history = _position_history(board)
    offset = HISTORY_LENGTH - len(history)

    for t, position in enumerate(history, start=offset):
        base = t * PIECE_PLANES
        for sq, piece in position.piece_map().items():
            view = view_sq(sq, color)
            plane = piece.piece_type - 1 + (0 if piece.color == color else 6)
            x[base + plane, view // 8, view % 8] = 1.0

    meta = HISTORY_LENGTH * PIECE_PLANES
    opponent = not color
    x[meta + 0].fill(float(board.has_kingside_castling_rights(color)))
    x[meta + 1].fill(float(board.has_queenside_castling_rights(color)))
    x[meta + 2].fill(float(board.has_kingside_castling_rights(opponent)))
    x[meta + 3].fill(float(board.has_queenside_castling_rights(opponent)))
    if board.ep_square is not None:
        ep = view_sq(board.ep_square, color)
        x[meta + 4, ep // 8, ep % 8] = 1.0
    x[meta + 5].fill(min(board.halfmove_clock, 100) / 100.0)
    x[meta + 6].fill(min(_repetition_count(board), 3) / 3.0)
    x[meta + 7].fill(min(board.fullmove_number, 100) / 100.0)
    return torch.from_numpy(x).unsqueeze(0)


def legal_moves_map(board, color):
    # Keep all legal promotions distinct; dictionary collisions are impossible.
    return {move_to_index(move, color): move for move in board.legal_moves}


def material_balance(board, color):
    bal = 0
    for piece in board.piece_map().values():
        val = PIECE_VALUE[piece.piece_type]
        bal += val if piece.color == color else -val
    return bal