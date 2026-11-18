from typing import Literal
import chess
from pydantic import BaseModel, Field, field_validator

class NewGame(BaseModel):
    mode: Literal['ai','local'] = 'ai'
    human_color: Literal['white','black'] = 'white'
    level: Literal[1,2,3,4] = 1
    seed: int | None = Field(default=None,ge=0,le=2**32-1)
    fen: str = Field(default=chess.STARTING_FEN,max_length=150)

    @field_validator('fen')
    @classmethod
    def valid_fen(cls, value):
        try:
            board = chess.Board(value)
            if not board.is_valid():
                raise ValueError()
        except ValueError as exc:
            raise ValueError('Thế cờ FEN không hợp lệ.') from exc
        return board.fen()

class GameAction(BaseModel):
    game_id: str = Field(min_length=32,max_length=36,pattern=r'^[0-9a-f-]+$')
    revision: int = Field(ge=0)

class HumanMove(GameAction):
    uci: str = Field(pattern=r'^[a-h][1-8][a-h][1-8][qrbn]?$')

class AiMove(GameAction):
    level: Literal[1,2,3,4] | None = None
    time_ms: int | None = Field(default=None,ge=1,le=5000)

class PauseAction(GameAction):
    paused: bool

class FinishAction(GameAction):
    kind: Literal['resign','draw']
