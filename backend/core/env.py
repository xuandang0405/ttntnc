from __future__ import annotations
import chess
import numpy as np
from backend.core.eval_pst import material_score

PIECE_NAMES = {1:'pawn',2:'knight',3:'bishop',4:'rook',5:'queen',6:'king'}

class ChessEnv:
    """Own the environment and adjudication; python-chess supplies standard laws."""
    def __init__(self, max_ply: int = 80, fen: str = chess.STARTING_FEN):
        self.max_ply = max_ply
        self.initial_fen = fen
        self.board = chess.Board(fen)
        if not self.board.is_valid():
            raise ValueError('Thế cờ FEN không hợp lệ.')
        self.ply = 0

    def reset(self, seed: int | None = None) -> str:
        self.board = chess.Board()
        self.initial_fen = self.board.fen()
        self.ply = 0
        return self.board.fen()

    def legal_actions(self) -> list[str]:
        return [] if self.is_terminal() else [m.uci() for m in self.board.legal_moves]

    def step(self, action_uci: str) -> tuple[str,int,bool]:
        if self.is_terminal():
            raise ValueError('Ván cờ đã kết thúc.')
        try:
            move = chess.Move.from_uci(action_uci)
        except ValueError as exc:
            raise ValueError('Nước đi UCI không hợp lệ.') from exc
        if move not in self.board.legal_moves:
            raise ValueError('Nước đi không đúng luật.')
        self.board.push(move)
        self.ply += 1
        return self.board.fen(), self.result(), self.is_terminal()

    def is_terminal(self) -> bool:
        return self.board.outcome(claim_draw=True) is not None or self.ply >= self.max_ply

    def result(self) -> int:
        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            return 0 if outcome.winner is None else (1 if outcome.winner else -1)
        if self.ply >= self.max_ply:
            score = material_score(self.board)
            return 1 if score >= 300 else -1 if score <= -300 else 0
        return 0

    def reward(self, role: int) -> float:
        if role not in (-1,1):
            raise ValueError('Role must be +1 or -1')
        return float(self.result() * role)

    def clone(self) -> ChessEnv:
        other = ChessEnv(self.max_ply, self.initial_fen)
        # Preserve repetition history so cloned states retain correct draw rules.
        other.board = self.board.copy(stack=True)
        other.ply = self.ply
        return other

    def undo(self) -> None:
        if not self.board.move_stack:
            raise ValueError('Chưa có nước đi để hoàn tác.')
        self.board.pop()
        self.ply -= 1

    def encode_tensor(self) -> np.ndarray:
        tensor = np.zeros((14,8,8), dtype=np.float32)
        for square, piece in self.board.piece_map().items():
            channel = piece.piece_type - 1 + (0 if piece.color else 6)
            tensor[channel,chess.square_rank(square),chess.square_file(square)] = 1
        tensor[12].fill(1 if self.board.turn else -1)
        tensor[13].fill(self.ply / self.max_ply)
        return tensor

    def render(self, mode: str = 'json') -> dict:
        if mode != 'json':
            raise ValueError('Only JSON rendering is supported')
        pieces = [{'square':chess.square_name(s),'type':PIECE_NAMES[p.piece_type],'color':'white' if p.color else 'black','symbol':p.symbol()} for s,p in sorted(self.board.piece_map().items())]
        outcome = self.board.outcome(claim_draw=True)
        reason = outcome.termination.name.lower() if outcome else ('max_ply' if self.ply >= self.max_ply else None)
        king = self.board.king(self.board.turn)
        return {'fen':self.board.fen(),'board':[self.board.piece_at(s).symbol() if self.board.piece_at(s) else None for s in chess.SQUARES],
                'pieces':pieces,'legal_actions':self.legal_actions(),'turn':'white' if self.board.turn else 'black','player':1 if self.board.turn else -1,
                'in_check':self.board.is_check(),'checked_king':chess.square_name(king) if self.board.is_check() and king is not None else None,
                'material_cp':material_score(self.board),'ply':self.ply,'max_ply':self.max_ply,'terminal':self.is_terminal(),'result':self.result(),'reason':reason}
