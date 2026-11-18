from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='CHESS_', env_file=ROOT/'.env',extra='ignore')
    database: str = 'data/chess.sqlite3'
    max_ply: int = Field(default=80, ge=2, le=1000)
    allowed_origins: list[str] = ['http://127.0.0.1:5173','http://localhost:5173','http://127.0.0.1:8000']

    @property
    def database_path(self) -> Path:
        p = Path(self.database)
        return p if p.is_absolute() else ROOT/p
