from contextlib import contextmanager
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
from backend.api.errors import AppError

class GameRepository:
    def __init__(self, path: Path):
        self.path = path

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def migrate(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript((Path(__file__).parent/'migrations/001_initial.sql').read_text(encoding='utf-8'))

    @staticmethod
    def authorize(row, token: str):
        if row is None:
            raise AppError('Không tìm thấy ván cờ.','GAME_NOT_FOUND',404)
        if not token or not hmac.compare_digest(row['token_hash'],hashlib.sha256(token.encode()).hexdigest()):
            raise AppError('Phiên ván cờ không hợp lệ.','UNAUTHORIZED',401)
        return {'id':row['id'],'version':row['version'],'data':json.loads(row['body'])}

    def create(self, game_id: str, token: str, data: dict):
        with self.connect() as db:
            db.execute('INSERT INTO games(id,token_hash,body) VALUES(?,?,?)',(game_id,hashlib.sha256(token.encode()).hexdigest(),json.dumps(data)))
        return {'id':game_id,'version':0,'data':data}

    def read(self, game_id: str, token: str):
        with self.connect() as db:
            return self.authorize(db.execute('SELECT * FROM games WHERE id=?',(game_id,)).fetchone(),token)

    @contextmanager
    def edit(self, game_id: str, token: str, revision: int):
        db = self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            record = self.authorize(db.execute('SELECT * FROM games WHERE id=?',(game_id,)).fetchone(),token)
            if record['version'] != revision:
                raise AppError('Ván cờ đã thay đổi. Hãy tải lại trạng thái mới nhất.','STALE_STATE',409)
            yield record
            record['version'] += 1
            db.execute('UPDATE games SET body=?,version=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',(json.dumps(record['data']),record['version'],game_id))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
