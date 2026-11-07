from dataclasses import dataclass
from typing import Optional

import chess


@dataclass
class MatchResult:
    winner: Optional[bool] 
    reason: str

class Match:
    def __init__(self, white, black, max_plies=None):
        self.white, self.black = white, black
        self.max_plies = max_plies
        self.board = chess.Board()
        self.last_move = None
        self.san = []
        self.result = None

    def current_player(self):
        return self.white if self.board.turn == chess.WHITE else self.black

    def humans(self):
        return [c for c, p in ((chess.WHITE, self.white), (chess.BLACK, self.black)) if p.is_human]

    def push(self, move):
        self.san.append(self.board.san(move))
        self.board.push(move)
        self.last_move = move
        outcome = self.board.outcome(claim_draw=True)
        if outcome:
            self.result = MatchResult(outcome.winner,
                                      outcome.termination.name.replace("_", " ").capitalize())
        elif self.max_plies and len(self.board.move_stack) >= self.max_plies:
            self.result = MatchResult(None, "Move limit reached")