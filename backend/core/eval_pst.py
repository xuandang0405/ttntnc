import chess

PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}

def material_score(board: chess.Board) -> int:
    return sum(value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK))) for piece, value in PIECE_VALUES.items() if piece != chess.KING)

def evaluate(board: chess.Board) -> int:
    score = material_score(board)
    centers = {chess.D4, chess.D5, chess.E4, chess.E5}
    for square, piece in board.piece_map().items():
        bonus = 20 if square in centers else 0
        if piece.piece_type == chess.PAWN:
            bonus += 10 * (chess.square_rank(square) if piece.color else 7 - chess.square_rank(square))
        score += bonus if piece.color else -bonus
    return score
