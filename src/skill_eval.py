import random
import time

import chess

from curriculum import endgame_position
from strategy_config import SKILL_ENDGAME_GAMES, SKILL_ENDGAME_MAX_PLIES, SKILL_POSITIONS

_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
_suite = {}

def _mating_moves(board):
    out = []
    for m in list(board.legal_moves):
        board.push(m)
        if board.is_checkmate():
            out.append(m)
        board.pop()
    return out


def _free_captures(board):
    out = []
    for m in board.legal_moves:
        victim = board.piece_at(m.to_square)
        if victim and _VALUE[victim.piece_type] >= 3 and not board.is_en_passant(m):
            board.push(m)
            safe = not board.is_attacked_by(board.turn, m.to_square)   # board.turn is now the opponent
            board.pop()
            if safe:
                out.append(m)
    return out


def _safe_moves(board):
    out = []
    for m in list(board.legal_moves):
        board.push(m)
        if not _mating_moves(board):
            out.append(m)
        board.pop()
    return out


def _build_suite(n):
    if n in _suite:
        return _suite[n]
    rng = random.Random(1234)
    suite = {"mate_in_1": [], "avoid_mate": [], "win_free_piece": []}
    for _ in range(8000):
        if all(len(v) >= n for v in suite.values()):
            break
        board = chess.Board()
        for _ in range(rng.randint(14, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or board.is_check():
            continue
        if len(suite["mate_in_1"]) < n and _mating_moves(board):
            suite["mate_in_1"].append(board.fen())
        if len(suite["win_free_piece"]) < n and _free_captures(board):
            suite["win_free_piece"].append(board.fen())
        if len(suite["avoid_mate"]) < n:
            passed = board.copy(stack=False)
            passed.push(chess.Move.null())
            if _mating_moves(passed):
                good = _safe_moves(board)
                if good and len(good) < board.legal_moves.count():
                    suite["avoid_mate"].append(board.fen())
    _suite[n] = suite
    return suite


def _success(name, board, move):
    if name == "mate_in_1":
        board.push(move)
        ok = board.is_checkmate()
    elif name == "avoid_mate":
        board.push(move)
        ok = not _mating_moves(board)
    else:
        ok = move in _free_captures(board)
    return ok


def _endgame_conversion(agent, games):
    wins = total = 0
    rng = random.Random(99)
    for template in ("KQvK", "KRvK"):
        for _ in range(games):
            board = endgame_position(template, strong=agent.color, rng=rng)
            for _ in range(SKILL_ENDGAME_MAX_PLIES):
                if board.turn == agent.color:
                    move = agent.select_move(board, greedy=True)
                else:
                    move = rng.choice(list(board.legal_moves))
                board.push(move)
                if board.is_game_over():
                    break
            total += 1
            wins += board.is_checkmate() and board.turn != agent.color
    return wins / max(1, total)


def evaluate_skills(agent):
    slow = agent.algorithm == "alphazero"          # tree search makes every move expensive
    n = max(5, SKILL_POSITIONS // 4) if slow else SKILL_POSITIONS
    games = 1 if slow else SKILL_ENDGAME_GAMES
    suite = _build_suite(n)
    result = {}
    for name, fens in suite.items():
        hits = 0
        for fen in fens:
            board = chess.Board(fen)
            hits += bool(_success(name, board, agent.select_move(chess.Board(fen), greedy=True)))
        result[name] = round(hits / max(1, len(fens)), 3)
    result["endgame_conversion"] = round(_endgame_conversion(agent, games), 3)
    result["positions"] = {k: len(v) for k, v in suite.items()}
    return result


def run_skill_evaluation(game):
    entry = {"games": game.games, "time": time.strftime("%H:%M:%S"),
             "White": evaluate_skills(game.white), "Black": evaluate_skills(game.black),
             "algorithms": {"White": game.white.algorithm, "Black": game.black.algorithm}}
    game.skills.append(entry)
    return entry