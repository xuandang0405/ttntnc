from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import secrets
import time
from uuid import uuid4
import chess
import chess.pgn
from backend.agents.level1_random import RandomAgent
from backend.agents.search_agent import SearchAgent
from backend.api.errors import AppError
from backend.api.games.repository import GameRepository
from backend.api.games.schemas import NewGame
from backend.core.env import ChessEnv, PIECE_NAMES

_CHAMPION_AGENTS: dict[int, any] = {}

def get_champion_agent(level: int):
    """Tải và lưu đệm siêu tác tử Champion từ checkpoint mạnh nhất đã huấn luyện."""
    global _CHAMPION_AGENTS
    if level in _CHAMPION_AGENTS:
        return _CHAMPION_AGENTS[level]
    try:
        champ_file = Path('experiments/champion_model.json')
        if not champ_file.exists():
            return None
        info = json.loads(champ_file.read_text(encoding='utf-8'))
        ckpt = info.get('checkpoint')
        if not ckpt or not Path(ckpt).exists():
            return None
        from model.learning.agents import NeuralAgent, LearnedAlphaBetaAgent, SuperNightmareAgent
        if level == 2:
            agent = NeuralAgent(ckpt, mode='policy', device='cpu')
        elif level == 3:
            agent = LearnedAlphaBetaAgent(ckpt, depth=2, device='cpu')
        elif level == 4:
            agent = SuperNightmareAgent(ckpt, depth=4, max_qdepth=4, device='cpu')
        else:
            return None
        _CHAMPION_AGENTS[level] = agent
        return agent
    except Exception:
        return None

def piece_json(piece: chess.Piece | None):
    return None if piece is None else {'type':PIECE_NAMES[piece.piece_type],'color':'white' if piece.color else 'black','symbol':piece.symbol()}

def move_event(board: chess.Board, move: chess.Move) -> dict:
    piece = board.piece_at(move.from_square)
    en_passant = board.is_en_passant(move)
    captured_square = move.to_square - (8 if board.turn else -8) if en_passant else move.to_square
    captured = board.piece_at(captured_square)
    rook = None
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook = {'from':chess.square_name(chess.square(7 if kingside else 0,rank)),'to':chess.square_name(chess.square(5 if kingside else 3,rank))}
    return {'uci':move.uci(),'san':board.san(move),'from':chess.square_name(move.from_square),'to':chess.square_name(move.to_square),
            'piece':piece_json(piece),'captured':{**piece_json(captured),'square':chess.square_name(captured_square)} if captured else None,
            'rook':rook,'promotion':PIECE_NAMES[move.promotion] if move.promotion else None,'is_en_passant':en_passant,
            'move_number':board.fullmove_number}

class GameService:
    def __init__(self, repository: GameRepository, max_ply: int = 80):
        self.repository, self.max_ply = repository, max_ply

    def restore(self, data):
        env = ChessEnv(data['max_ply'],data['initial_fen'])
        for uci in data['moves']:
            env.board.push_uci(uci)
            env.ply += 1
        return env

    def state(self, record):
        data = record['data']; env = self.restore(data)
        state = env.render()
        history = []; captured = {'white':[],'black':[]}
        board = chess.Board(data['initial_fen'])
        for index, uci in enumerate(data['moves']):
            move = chess.Move.from_uci(uci)
            event = move_event(board,move)
            history.append({**event,'ply':index+1})
            if event['captured']:
                captured[event['captured']['color']].append(event['captured']['type'])
            board.push(move)
        if data.get('forced_result') is not None:
            state.update(terminal=True,result=data['forced_result'],reason=data['forced_reason'],legal_actions=[])
        elapsed = data.get('elapsed_ms',0)
        if data.get('resumed_at') is not None:
            elapsed += max(0,int((time.time()-data['resumed_at'])*1000))
        state.update(game_id=record['id'],revision=record['version'],mode=data['mode'],human_color=data['human_color'],level=data.get('level',1),seed=data['seed'],
                     paused=data['paused'],history=history,captured=captured,created_at=data['created_at'],elapsed_ms=elapsed,
                     telemetry=data.get('telemetry'),can_undo=self.can_undo(data),initial_fen=data['initial_fen'])
        return state

    @staticmethod
    def can_undo(data):
        if not data['moves']:
            return False
        return not (data['mode']=='ai' and data['human_color']=='black' and len(data['moves'])==1)

    def response(self, record, event=None):
        return {'game_id':record['id'],'state':self.state(record),'telemetry':record['data'].get('telemetry'),'move':event}

    def new(self, options: NewGame):
        seed = options.seed if options.seed is not None else secrets.randbits(32)
        data = {'mode':options.mode,'human_color':options.human_color,'level':options.level,'seed':seed,'initial_fen':options.fen,'moves':[],
                'max_ply':self.max_ply,'paused':False,'created_at':datetime.now(timezone.utc).isoformat(),'elapsed_ms':0,'resumed_at':time.time(),
                'forced_result':None,'forced_reason':None,'telemetry':None,'rng_state':random.Random(seed).getstate()}
        if self.restore(data).is_terminal():
            data['resumed_at'] = None
        game_id, token = str(uuid4()), secrets.token_urlsafe(32)
        record = self.repository.create(game_id,token,data)
        return {**self.response(record),'token':token}

    def get(self, game_id, token):
        return self.response(self.repository.read(game_id,token))

    def ensure_active(self, data, env):
        if data.get('forced_result') is not None or env.is_terminal():
            raise AppError('Ván cờ đã kết thúc. Hãy bắt đầu ván mới.','GAME_OVER',409)
        if data['paused']:
            raise AppError('Ván cờ đang tạm dừng.','GAME_PAUSED',409)

    @staticmethod
    def stop_timer(data):
        if data.get('resumed_at') is not None:
            data['elapsed_ms'] += max(0,int((time.time()-data['resumed_at'])*1000))
            data['resumed_at'] = None

    def play(self, game_id, token, revision, *, uci=None, ai=False, time_ms=None, level=None):
        event = None
        with self.repository.edit(game_id,token,revision) as record:
            data = record['data']; env = self.restore(data)
            self.ensure_active(data,env)
            human_turn = ('white' if env.board.turn else 'black') == data['human_color']
            if ai and (data['mode']!='ai' or human_turn):
                raise AppError('Hiện chưa phải lượt của máy.','WRONG_TURN',409)
            if not ai and data['mode']=='ai' and not human_turn:
                raise AppError('Hãy đợi máy hoàn thành nước đi.','WRONG_TURN',409)
            if ai:
                saved_level=data.get('level',1)
                if level is not None and level!=saved_level:
                    raise AppError('Độ khó không khớp với ván hiện tại.','LEVEL_MISMATCH',422)
                rng = random.Random()
                version, internal, gaussian = data['rng_state']
                rng.setstate((version,tuple(internal),gaussian))
                agent = None
                if os.getenv('USE_NEURAL_AI', '0') == '1' and saved_level > 1:
                    agent = get_champion_agent(saved_level)
                if agent is None:
                    agent = RandomAgent(rng=rng) if saved_level==1 else SearchAgent(saved_level)
                uci, telemetry = agent.select_action(env,time_ms)
                data['rng_state'] = rng.getstate()
                data['telemetry'] = telemetry
            try:
                move = chess.Move.from_uci(uci)
                if move not in env.board.legal_moves:
                    raise ValueError()
                event = move_event(env.board,move)
                env.step(uci)
            except ValueError as exc:
                raise AppError('Nước đi không đúng luật. Vui lòng chọn ô được gợi ý.','ILLEGAL_MOVE',422) from exc
            data['moves'].append(uci)
            if env.is_terminal():
                self.stop_timer(data)
        return self.response(record,event)

    def undo(self, game_id, token, revision):
        with self.repository.edit(game_id,token,revision) as record:
            data = record['data']
            if not self.can_undo(data):
                raise AppError('Chưa có lượt của bạn để hoàn tác.','NOTHING_TO_UNDO',409)
            env = self.restore(data)
            count = 1
            if data['mode']=='ai':
                env.undo()
                human_turn = ('white' if env.board.turn else 'black') == data['human_color']
                if not human_turn and len(data['moves'])>=2:
                    count = 2
            del data['moves'][-count:]
            data.update(forced_result=None,forced_reason=None,telemetry=None)
            if not data['paused'] and data.get('resumed_at') is None:
                data['resumed_at'] = time.time()
            # Replay RNG calls for the retained AI turns; undo/replay is reproducible.
            rng = random.Random(data['seed']); replay = ChessEnv(data['max_ply'],data['initial_fen'])
            for uci in data['moves']:
                is_ai = data['mode']=='ai' and ('white' if replay.board.turn else 'black') != data['human_color']
                if is_ai and data.get('level',1)==1:
                    rng.choice(replay.legal_actions())
                replay.step(uci)
            data['rng_state'] = rng.getstate()
        return self.response(record)

    def pause(self, game_id, token, revision, paused):
        with self.repository.edit(game_id,token,revision) as record:
            data = record['data']
            if self.state(record)['terminal']:
                raise AppError('Ván cờ đã kết thúc.','GAME_OVER',409)
            if paused:
                self.stop_timer(data)
            elif data['paused']:
                data['resumed_at'] = time.time()
            data['paused'] = paused
        return self.response(record)

    def finish(self, game_id, token, revision, kind):
        with self.repository.edit(game_id,token,revision) as record:
            data = record['data']; env = self.restore(data)
            if self.state(record)['terminal']:
                raise AppError('Ván cờ đã kết thúc.','GAME_OVER',409)
            role = data['human_color'] if data['mode']=='ai' else ('white' if env.board.turn else 'black')
            data['forced_result'] = 0 if kind=='draw' else (-1 if role=='white' else 1)
            data['forced_reason'] = 'agreement' if kind=='draw' else 'resignation'
            data['paused'] = False
            self.stop_timer(data)
        return self.response(record)

    def hint(self, game_id, token):
        record = self.repository.read(game_id,token); data = record['data']; env = self.restore(data)
        self.ensure_active(data,env)
        action, telemetry = RandomAgent(data['seed'] ^ env.ply ^ 0xA13).select_action(env)
        return {'uci':action,'san':env.board.san(chess.Move.from_uci(action)),'kind':'random','probability':telemetry['selected_probability']}

    def pgn(self, game_id, token):
        record = self.repository.read(game_id,token); data = record['data']
        game = chess.pgn.Game.from_board(self.restore(data).board)
        level=data.get('level',1)
        opponent={1:'Random',2:'Normal',3:'Hard',4:'Nightmare'}[level]
        game.headers['Event'] = f'Co Vua - {opponent}' if data['mode']=='ai' else 'Co Vua - Local'
        game.headers['White'] = opponent if data['mode']=='ai' and data['human_color']=='black' else 'Human'
        game.headers['Black'] = opponent if data['mode']=='ai' and data['human_color']=='white' else 'Human'
        game.headers['Level']=str(level)
        state = self.state(record)
        game.headers['Result'] = ('1-0' if state['result']==1 else '0-1' if state['result']==-1 else '1/2-1/2') if state['terminal'] else '*'
        game.headers['Seed'] = str(data['seed'])
        return str(game)+'\n'
