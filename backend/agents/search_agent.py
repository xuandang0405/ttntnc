"""Small, bounded chess opponents for the three levels above Random."""
from time import perf_counter
import chess
from backend.agents.base_agent import BaseAgent
from backend.core.env import ChessEnv
from backend.core.eval_pst import evaluate, material_score, PIECE_VALUES

LEVEL_DEPTH={2:1,3:2,4:4}
LEVEL_BUDGET_MS={2:180,3:600,4:1400}
MATE=100_000

class SearchTimeout(Exception):
    pass

class SearchAgent(BaseAgent):
    def __init__(self, level: int):
        if level not in LEVEL_DEPTH:
            raise ValueError('Unsupported search level')
        self.level=level

    def select_action(self, env: ChessEnv, time_limit_ms: float | None=None):
        start=perf_counter()
        legal=env.legal_actions()
        if not legal:
            raise ValueError('Không có nước đi hợp lệ.')
        budget=LEVEL_BUDGET_MS[self.level] if time_limit_ms is None else min(LEVEL_BUDGET_MS[self.level],max(1,time_limit_ms))
        deadline=start+budget/1000
        board=env.board.copy(stack=True)
        nodes=0

        def terminal(ply):
            outcome=board.outcome(claim_draw=True)
            if outcome:
                return 0 if outcome.winner is None else (MATE-ply)*(1 if outcome.winner==board.turn else -1)
            if env.ply+ply>=env.max_ply:
                score=material_score(board)
                white=1 if score>=300 else -1 if score<=-300 else 0
                return white*10_000*(1 if board.turn else -1)
            return None

        def ordered():
            def priority(move):
                victim=board.piece_at(move.to_square)
                capture=PIECE_VALUES[victim.piece_type] if victim else 100 if board.is_en_passant(move) else 0
                attacker=board.piece_at(move.from_square)
                return capture*10-(PIECE_VALUES[attacker.piece_type] if capture else 0)+(PIECE_VALUES.get(move.promotion,0)*8)
            return sorted(board.legal_moves,key=lambda move:(-priority(move),move.uci()))

        def search(depth,alpha,beta,ply):
            nonlocal nodes
            if perf_counter()>=deadline:
                raise SearchTimeout()
            nodes+=1
            ended=terminal(ply)
            if ended is not None:
                return ended
            if depth==0:
                return evaluate(board)*(1 if board.turn else -1)
            best=-MATE*2
            for move in ordered():
                board.push(move)
                try:
                    score=-search(depth-1,-beta,-alpha,ply+1)
                finally:
                    board.pop()
                best=max(best,score);alpha=max(alpha,score)
                if alpha>=beta:
                    break
            return best

        chosen=ordered()[0].uci();best_score=evaluate(board)*(1 if board.turn else -1);completed_depth=0
        for depth in range(1,LEVEL_DEPTH[self.level]+1):
            best=-MATE*2;candidate=chosen;alpha=-MATE*2
            try:
                moves=ordered()
                moves.sort(key=lambda move:move.uci()!=chosen)
                for move in moves:
                    board.push(move)
                    try:
                        score=-search(depth-1,-MATE*2,-alpha,1)
                    finally:
                        board.pop()
                    if score>best:
                        best=score;candidate=move.uci()
                    alpha=max(alpha,score)
            except SearchTimeout:
                break
            chosen,best_score,completed_depth=candidate,best,depth
            if best>=MATE-100:
                break
        return chosen,{'chosen':chosen,'algorithm':'Greedy' if self.level==2 else 'AlphaBeta','depth':completed_depth,
                       'nodes':nodes,'latency_ms':round((perf_counter()-start)*1000,4),
                       'eval_score':best_score*(1 if env.board.turn else -1)/100,'top_moves':{chosen:1},
                       'legal_count':len(legal),'selected_probability':1}
