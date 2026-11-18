from time import perf_counter
import chess
import pytest
from backend.agents.search_agent import SearchAgent
from backend.core.env import ChessEnv

def test_normal_captures_unprotected_queen_without_mutating_board():
    env=ChessEnv(fen='6k1/8/8/3q4/8/8/8/3R2K1 w - - 0 1')
    before=env.board.fen()
    move,info=SearchAgent(2).select_action(env)
    assert move=='d1d5'
    assert env.board.fen()==before and env.ply==0
    assert info['depth']==1 and info['algorithm']=='Greedy'

@pytest.mark.parametrize('level',[3,4])
def test_higher_levels_find_mate(level):
    env=ChessEnv(fen='7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
    move,info=SearchAgent(level).select_action(env)
    board=env.board.copy();board.push_uci(move)
    assert board.is_checkmate() and info['depth']>=1

def test_deadline_keeps_legal_fallback_and_restores_board():
    env=ChessEnv();before=env.board.fen();start=perf_counter()
    move,info=SearchAgent(4).select_action(env,1)
    assert move in env.legal_actions()
    assert env.board.fen()==before and not env.board.move_stack
    assert perf_counter()-start<.5
