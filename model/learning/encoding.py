"""Versioned board features and a collision-free action vocabulary."""
import chess
import numpy as np

CHANNELS = 20
ACTION_SIZE = 64 * 64 * 5
ENCODER_VERSION = 'planes20-v1'
PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def repetition_feature(board: chess.Board) -> float:
    return 1.0 if board.is_repetition(3) else .5 if board.is_repetition(2) else 0.0


def encode_board(board: chess.Board, repetition: float | None = None) -> np.ndarray:
    """Absolute orientation: rank 1 is row 0; value targets use side to move.

    FEN lacks repetition history. JSONL samples therefore store the feature
    computed from the original board stack separately.
    """
    x = np.zeros((CHANNELS, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        c = piece.piece_type - 1 + (0 if piece.color else 6)
        x[c, chess.square_rank(square), chess.square_file(square)] = 1
    x[12].fill(1 if board.turn else -1)
    for c, (color, king_side) in enumerate(
        [(chess.WHITE, True), (chess.WHITE, False),
         (chess.BLACK, True), (chess.BLACK, False)], start=13
    ):
        right = (board.has_kingside_castling_rights(color) if king_side
                 else board.has_queenside_castling_rights(color))
        x[c].fill(float(right))
    if board.ep_square is not None and board.has_legal_en_passant():
        x[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    x[18].fill(min(board.halfmove_clock / 150, 1))
    x[19].fill(repetition_feature(board) if repetition is None else repetition)
    return x


def move_id(move: chess.Move) -> int:
    if move == chess.Move.null() or move.drop is not None:
        raise ValueError('Chỉ hỗ trợ nước cờ vua tiêu chuẩn.')
    return (
        (move.from_square * 64 + move.to_square) * 5
        + PROMOTIONS.index(move.promotion)
    )


def id_move(index: int) -> chess.Move:
    if not 0 <= index < ACTION_SIZE:
        raise ValueError('Mã hành động ngoài phạm vi.')
    pair, promotion = divmod(index, 5)
    source, target = divmod(pair, 64)
    return chess.Move(source, target, promotion=PROMOTIONS[promotion])


def position_key(fen: str) -> str:
    # Ignore clocks for conservative duplicate detection across data splits.
    return ' '.join(fen.split()[:4])
