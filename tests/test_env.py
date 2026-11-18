import chess
import numpy as np
import pytest
from backend.agents.level1_random import RandomAgent
from backend.core.env import ChessEnv

def test_starting_board_and_tensor():
    env=ChessEnv()
    assert len(env.legal_actions())==20
    tensor=env.encode_tensor()
    assert tensor.shape==(14,8,8) and tensor.dtype==np.float32
    assert tensor[:12].sum()==32
    assert np.all(tensor[12]==1) and np.all(tensor[13]==0)
    original=env.board.fen()
    with pytest.raises(ValueError):env.step('e2e5')
    assert env.board.fen()==original and env.ply==0
    env.step('e2e4')
    assert np.all(env.encode_tensor()[12]==-1)
    clone=env.clone();clone.step('e7e5')
    assert env.ply==1 and clone.ply==2
    env.undo();assert env.board.fen()==original

def test_checkmate_stalemate_and_no_moves_after_game_over():
    env=ChessEnv()
    for move in ['f2f3','e7e5','g2g4','d8h4']:env.step(move)
    assert env.is_terminal() and env.result()==-1 and env.reward(-1)==1
    assert env.render()['reason']=='checkmate' and not env.legal_actions()
    with pytest.raises(ValueError):env.step('a2a3')
    stale=ChessEnv(fen='7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
    assert stale.is_terminal() and stale.result()==0 and stale.render()['reason']=='stalemate'

def test_pins_castling_en_passant_and_underpromotion():
    pinned=ChessEnv(fen='k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1')
    assert 'e2d2' not in pinned.legal_actions()
    castle=ChessEnv(fen='r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    castle.step('e1g1')
    assert castle.board.piece_at(chess.F1).piece_type==chess.ROOK
    assert castle.board.piece_at(chess.G1).piece_type==chess.KING
    ep=ChessEnv(fen='4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
    ep.step('e5d6');assert ep.board.piece_at(chess.D5) is None
    promote=ChessEnv(fen='4k3/P7/8/8/8/8/8/4K3 w - - 0 1')
    assert len([m for m in promote.legal_actions() if m.startswith('a7a8')])==4
    promote.step('a7a8n');assert promote.board.piece_at(chess.A8).piece_type==chess.KNIGHT

def test_adjudication_and_repetition_survive_clone():
    env=ChessEnv(max_ply=2,fen='4k3/8/8/8/8/8/8/Q3K3 w - - 0 1')
    env.step('a1a2');env.step('e8d8')
    assert env.result()==1 and env.render()['reason']=='max_ply'
    env.undo();assert not env.is_terminal()
    repeat=ChessEnv()
    for move in ['g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1']:repeat.step(move)
    assert repeat.is_terminal() and repeat.clone().is_terminal()
    assert repeat.render()['reason']=='threefold_repetition'

def test_random_agent_is_seeded_and_uniform_over_legal_actions():
    env=ChessEnv();a=RandomAgent(123);b=RandomAgent(123)
    outputs=[a.select_action(env) for _ in range(30)]
    assert [m for m,_ in outputs]==[b.select_action(env)[0] for _ in range(30)]
    for move,telemetry in outputs:
        assert move in env.legal_actions()
        assert telemetry['nodes']==0 and telemetry['selected_probability']==pytest.approx(1/20)
        assert len(telemetry['top_moves'])==20
        assert all(p==pytest.approx(1/20) for p in telemetry['top_moves'].values())
