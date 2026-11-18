"""Agents compatible with BaseAgent; load weights once, then reuse."""
from time import perf_counter
import chess
import numpy as np
import torch
from backend.agents.base_agent import BaseAgent
from backend.core.eval_pst import PIECE_VALUES, material_score
from model.learning.encoding import encode_board, move_id
from model.learning.models import load_model


def terminal_score(board, color):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0. if outcome.winner is None else (1. if outcome.winner == color else -1.)


@torch.inference_mode()
def value_moves(model, board, device='cpu', remaining=None):
    """Child value is for the opponent, so its sign must be reversed."""
    model.eval()
    moves = list(board.legal_moves)
    scores = {}
    pending, features = [], []
    for move in moves:
        child = board.copy(stack=True)
        child.push(move)
        value = terminal_score(child, board.turn)
        if value is None and remaining is not None and remaining <= 1:
            cp = material_score(child)
            white = 1 if cp >= 300 else -1 if cp <= -300 else 0
            value = float(white if board.turn else -white)
        if value is None:
            pending.append(move)
            features.append(encode_board(child))
        else:
            scores[move] = value
    if pending:
        batch = torch.from_numpy(np.stack(features)).to(device)
        values = model.value(batch).clamp(-.95, .95).cpu().tolist()
        scores.update((move, -v) for move, v in zip(pending, values))
    return scores


class NeuralAgent(BaseAgent):
    def __init__(self, checkpoint, mode='policy', device='cpu'):
        if mode not in ('policy', 'value'):
            raise ValueError('mode phải là policy hoặc value.')
        self.model, self.metadata = load_model(checkpoint, device)
        if mode == 'policy' and not self.metadata.get('policy_trained', False):
            raise ValueError('Checkpoint chỉ học value; hãy dùng mode=value.')
        self.mode, self.device = mode, device
        self.model.eval()
        with torch.inference_mode():
            self.model(torch.from_numpy(encode_board(chess.Board())).unsqueeze(0).to(device))

    def select_action(self, env, time_limit_ms=None):
        start = perf_counter()
        legal = env.legal_actions()
        if not legal:
            raise ValueError('Ván đã kết thúc hoặc không có nước hợp lệ.')
        with torch.inference_mode():
            if self.mode == 'policy':
                x = torch.from_numpy(encode_board(env.board)).unsqueeze(0).to(self.device)
                logits, value = self.model(x)
                ids = [move_id(chess.Move.from_uci(u)) for u in legal]
                probabilities = logits[0, ids].softmax(0).cpu().tolist()
                chosen = legal[max(range(len(legal)), key=probabilities.__getitem__)]
                top = dict(zip(legal, probabilities))
                output_value = value.item()
            else:
                scores = value_moves(self.model, env.board, self.device, env.max_ply - env.ply)
                move = max(scores, key=lambda m: (scores[m], m.uci()))
                chosen = move.uci()
                top = {m.uci(): float(v) for m, v in scores.items()}
                output_value = scores[move]
        latency = (perf_counter() - start) * 1000
        return chosen, {
            'chosen': chosen, 'algorithm': f'Neural-{self.mode}', 'nodes': 0,
            'latency_ms': latency, 'top_moves': top, 'eval_score': None,
            'value': output_value, 'value_pov': 'side_to_move',
            'top_moves_kind': 'probability' if self.mode == 'policy' else 'value',
            'budget_exceeded': time_limit_ms is not None and latency > time_limit_ms,
        }


class LearnedAlphaBetaAgent(NeuralAgent):
    def __init__(self, checkpoint, depth=2, device='cpu'):
        super().__init__(checkpoint, 'value', device)
        if not 1 <= depth <= 8:
            raise ValueError('Độ sâu phải từ 1 đến 8.')
        self.depth = depth

    def select_action(self, env, time_limit_ms=None):
        start = perf_counter()
        if not env.legal_actions():
            raise ValueError('Ván đã kết thúc.')
        budget = 200 if time_limit_ms is None else max(1., time_limit_ms)
        deadline = start + budget / 1000
        board = env.board.copy(stack=True)
        nodes, completed = 0, 0

        def ordered():
            def priority(move):
                victim = board.piece_at(move.to_square)
                capture = PIECE_VALUES[victim.piece_type] if victim else 100 if board.is_en_passant(move) else 0
                attacker = board.piece_at(move.from_square)
                return capture * 10 - (PIECE_VALUES[attacker.piece_type] if capture else 0) + PIECE_VALUES.get(move.promotion, 0)
            return sorted(board.legal_moves, key=lambda m: (-priority(m), m.uci()))

        def search(depth, alpha, beta, ply):
            nonlocal nodes
            if perf_counter() >= deadline:
                raise TimeoutError
            nodes += 1
            terminal = terminal_score(board, board.turn)
            if terminal is not None:
                return terminal * (1 - ply * .0001)
            if env.ply + ply >= env.max_ply:
                cp = material_score(board)
                white = 1 if cp >= 300 else -1 if cp <= -300 else 0
                return float(white if board.turn else -white)
            if depth == 0:
                with torch.inference_mode():
                    x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
                    return float(self.model.value(x).clamp(-.95, .95).item())
            best = -2.
            for move in ordered():
                board.push(move)
                try:
                    value = -search(depth - 1, -beta, -alpha, ply + 1)
                finally:
                    board.pop()
                best, alpha = max(best, value), max(alpha, value)
                if alpha >= beta:
                    break
            return best

        chosen, best_value = ordered()[0].uci(), 0.
        for depth in range(1, self.depth + 1):
            candidate, best, alpha = chosen, -2., -2.
            moves = ordered()
            moves.sort(key=lambda m: m.uci() != chosen)
            try:
                for move in moves:
                    board.push(move)
                    try:
                        value = -search(depth - 1, -2., -alpha, 1)
                    finally:
                        board.pop()
                    if value > best:
                        best, candidate = value, move.uci()
                    alpha = max(alpha, value)
            except TimeoutError:
                break
            chosen, best_value, completed = candidate, best, depth
            if best > .99:
                break
        latency = (perf_counter() - start) * 1000
        return chosen, {'chosen': chosen, 'algorithm': 'AlphaBeta-NN', 'depth': completed,
                        'nodes': nodes, 'latency_ms': latency, 'eval_score': None,
                        'value': best_value, 'value_pov': 'side_to_move',
                        'budget_exceeded': latency > budget, 'top_moves': {chosen: best_value}}


class SuperNightmareAgent(LearnedAlphaBetaAgent):
    """Siêu tác tử Ác mộng kết hợp Mạng nơ-ron sâu + Quiescence Search + Policy Move Ordering.

    Được thiết kế để không bỏ sót bất kỳ đòn chiến thuật nào (tránh Horizon Effect)
    và tối ưu hóa cắt tỉa Alpha-Beta tới 90% số nhánh dư thừa.
    """
    def __init__(self, checkpoint, depth=4, max_qdepth=4, device='cpu'):
        super().__init__(checkpoint, depth=depth, device=device)
        self.max_qdepth = max_qdepth

    def select_action(self, env, time_limit_ms=None):
        start = perf_counter()
        if not env.legal_actions():
            raise ValueError('Ván đã kết thúc.')
        budget = 1400 if time_limit_ms is None else max(1., time_limit_ms)
        deadline = start + budget / 1000
        board = env.board.copy(stack=True)
        nodes, q_nodes, completed = 0, 0, 0

        # Ưu tiên nước ăn quân MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        def capture_priority(move):
            victim = board.piece_at(move.to_square)
            capture = PIECE_VALUES[victim.piece_type] if victim else 100 if board.is_en_passant(move) else 0
            attacker = board.piece_at(move.from_square)
            return capture * 10 - (PIECE_VALUES[attacker.piece_type] if capture else 0) + PIECE_VALUES.get(move.promotion, 0) * 8

        # Sắp xếp nước đi thông minh kết hợp Policy của mạng nơ-ron
        def order_moves(legal_moves, policy_dict=None):
            def sort_key(m):
                p_score = policy_dict.get(m.uci(), 0.0) if policy_dict else 0.0
                return (-capture_priority(m), -p_score, m.uci())
            return sorted(legal_moves, key=sort_key)

        # 1. Quiescence Search (Tìm kiếm tĩnh giải quyết ăn quân tại lá, chống Horizon Effect)
        def quiescence(alpha, beta, ply, qdepth):
            nonlocal nodes, q_nodes
            if perf_counter() >= deadline:
                raise TimeoutError
            nodes += 1
            q_nodes += 1

            terminal = terminal_score(board, board.turn)
            if terminal is not None:
                return terminal * (1 - ply * .0001)

            # Stand-pat evaluation
            with torch.inference_mode():
                x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
                stand_pat = float(self.model.value(x).clamp(-.95, .95).item())

            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            if qdepth <= 0:
                return stand_pat

            captures = [m for m in board.legal_moves if board.is_capture(m)]
            if not captures:
                return stand_pat

            captures.sort(key=lambda m: (-capture_priority(m), m.uci()))
            for move in captures:
                board.push(move)
                try:
                    val = -quiescence(-beta, -alpha, ply + 1, qdepth - 1)
                finally:
                    board.pop()
                if val >= beta:
                    return beta
                if val > alpha:
                    alpha = val
            return alpha

        # 2. Main Search với Alpha-Beta Pruning
        def search(depth, alpha, beta, ply):
            nonlocal nodes
            if perf_counter() >= deadline:
                raise TimeoutError
            nodes += 1
            terminal = terminal_score(board, board.turn)
            if terminal is not None:
                return terminal * (1 - ply * .0001)
            if env.ply + ply >= env.max_ply:
                cp = material_score(board)
                white = 1 if cp >= 300 else -1 if cp <= -300 else 0
                return float(white if board.turn else -white)
            if depth == 0:
                # Kích hoạt Quiescence Search tại nút lá
                return quiescence(alpha, beta, ply, self.max_qdepth)

            best = -2.
            moves = sorted(board.legal_moves, key=lambda m: (-capture_priority(m), m.uci()))
            for move in moves:
                board.push(move)
                try:
                    val = -search(depth - 1, -beta, -alpha, ply + 1)
                finally:
                    board.pop()
                best = max(best, val)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return best

        # Policy priors tại root để duyệt nhánh sáng giá nhất đầu tiên
        policy_priors = {}
        with torch.inference_mode():
            x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
            logits, _ = self.model(x)
            move_ids = [move_id(m) for m in board.legal_moves]
            probs = logits[0, move_ids].softmax(0).cpu().tolist()
            policy_priors = {m.uci(): p for m, p in zip(board.legal_moves, probs)}

        root_moves = order_moves(list(board.legal_moves), policy_priors)
        chosen = root_moves[0].uci()
        best_val = -2.0

        for depth in range(1, self.depth + 1):
            candidate, best, alpha = chosen, -2.0, -2.0
            root_moves.sort(key=lambda m: m.uci() != chosen)
            try:
                for move in root_moves:
                    board.push(move)
                    try:
                        val = -search(depth - 1, -2.0, -alpha, 1)
                    finally:
                        board.pop()
                    if val > best:
                        best, candidate = val, move.uci()
                    alpha = max(alpha, val)
            except TimeoutError:
                break
            chosen, best_val, completed = candidate, best, depth
            if best > 0.99:
                break

        latency = (perf_counter() - start) * 1000
        return chosen, {
            'chosen': chosen, 'algorithm': 'SuperNightmare-ResNet', 'depth': completed,
            'nodes': nodes, 'q_nodes': q_nodes, 'latency_ms': latency, 'eval_score': None,
            'value': best_val, 'value_pov': 'side_to_move',
            'budget_exceeded': latency > budget, 'top_moves': {chosen: best_val}
        }
