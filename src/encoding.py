import chess
import numpy as np
import torch
from config import PIECE_VALUE

def view_sq(sq, color):
    return sq if color == chess.WHITE else chess.square_mirror(sq)


def encode(board, color):
    x = np.zeros((16, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        v = view_sq(sq, color)
        plane = piece.piece_type - 1 + (0 if piece.color == color else 6)
        x[plane, v // 8, v % 8] = 1.0
    opp = not color
    x[12] = board.has_kingside_castling_rights(color)
    x[13] = board.has_queenside_castling_rights(color)
    x[14] = board.has_kingside_castling_rights(opp)
    x[15] = board.has_queenside_castling_rights(opp)
    return torch.from_numpy(x).unsqueeze(0)


def move_to_index(move, color):
    return view_sq(move.from_square, color) * 64 + view_sq(move.to_square, color)


def legal_moves_map(board, color):
    m = {}
    for mv in board.legal_moves:
        if mv.promotion and mv.promotion != chess.QUEEN:
            continue
        m[move_to_index(mv, color)] = mv
    return m


def material_balance(board, color):
    bal = 0
    for piece in board.piece_map().values():
        val = PIECE_VALUE[piece.piece_type]
        bal += val if piece.color == color else -val
    return bal