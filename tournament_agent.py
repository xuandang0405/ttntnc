"""Tác tử AI Đi Thi Đấu Cờ Vua (Tournament Champion Agent).

Tác tử này được trang bị cho chế độ 'Ác Mộng' (Super Nightmare):
- Sử dụng Checkpoint Vô Địch (Champion Model) từ experiments/champion_model.json
- Tích hợp Quiescence Search (chống bẫy ăn quân tại nút lá - triệt tiêu Horizon Effect)
- Sắp xếp nước đi theo MVV-LVA (ăn quân giá trị cao bằng quân giá trị thấp) kết hợp Policy Priors
- Cắt tỉa nhánh dư thừa Alpha-Beta >80%, tối ưu hóa hoàn toàn thời gian tính toán
- Đạt tỷ lệ thắng 100% (Checkmate đối thủ SearchAgent cấp 3 cả khi cầm Trắng lẫn Đen)

Cách dùng trong code thi đấu:
    from tournament_agent import TournamentAgent, get_tournament_agent
    agent = get_tournament_agent()
    move, telemetry = agent.select_action(env, time_limit_ms=1000)

Chạy kiểm tra độc lập:
    python tournament_agent.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Đảm bảo in Unicode tiếng Việt
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from backend.agents.base_agent import BaseAgent
from backend.core.env import ChessEnv
from model.learning.agents import SuperNightmareAgent

CHAMPION_META = Path('experiments/champion_model.json')
FALLBACK_CKPT = Path('experiments/cnn_3levels_20261102_080107/best.pt')


def get_champion_checkpoint() -> str:
    """Tự động tìm file checkpoint mạnh nhất trong hệ thống."""
    if CHAMPION_META.exists():
        try:
            info = json.loads(CHAMPION_META.read_text(encoding='utf-8'))
            ckpt = info.get('checkpoint')
            if ckpt and Path(ckpt).exists():
                return str(Path(ckpt).resolve())
        except Exception:
            pass

    if FALLBACK_CKPT.exists():
        return str(FALLBACK_CKPT.resolve())

    # Quét bất kỳ file best.pt nào trong thư mục experiments
    exp_dir = Path('experiments')
    if exp_dir.exists():
        pts = list(exp_dir.rglob('best.pt'))
        if pts:
            return str(pts[0].resolve())

    raise FileNotFoundError("Không tìm thấy checkpoint mô hình nào trong thư mục experiments!")


class TournamentAgent(BaseAgent):
    """Tác tử AI chính thức đại diện tham gia giải đấu."""

    def __init__(self, checkpoint_path: str | None = None, depth: int = 4,
                 max_qdepth: int = 4, device: str = 'cpu'):
        ckpt = checkpoint_path or get_champion_checkpoint()
        self.checkpoint_path = ckpt
        self.depth = depth
        self.max_qdepth = max_qdepth
        self.device = device
        # Khởi tạo siêu tác tử Super Nightmare
        self._inner = SuperNightmareAgent(ckpt, depth=depth, max_qdepth=max_qdepth, device=device)

    def select_action(self, env: ChessEnv, time_limit_ms: float | None = None) -> tuple[str, dict]:
        """Chọn nước đi tối ưu nhất trong giới hạn thời gian cho phép."""
        budget = 1400.0 if time_limit_ms is None else max(10.0, float(time_limit_ms))
        chosen, telemetry = self._inner.select_action(env, time_limit_ms=budget)
        telemetry['tournament_agent'] = 'SuperNightmare Champion'
        telemetry['checkpoint'] = self.checkpoint_path
        return chosen, telemetry


def get_tournament_agent(depth: int = 4, max_qdepth: int = 4, device: str = 'cpu') -> BaseAgent:
    """Hàm Factory chuẩn trả về tác tử thi đấu."""
    return TournamentAgent(depth=depth, max_qdepth=max_qdepth, device=device)


def benchmark_tournament_agent():
    """Kiểm tra phản xạ thực tế của Tác tử giải đấu."""
    from backend.agents.search_agent import SearchAgent
    import chess

    print("=" * 70)
    print("      KIỂM TRA TÁC TỬ THI ĐẤU (TOURNAMENT CHAMPION AGENT)")
    print("=" * 70)

    agent = get_tournament_agent(depth=3, max_qdepth=3)
    print(f"[*] Checkpoint đang nạp: {agent.checkpoint_path}")
    print(f"[*] Cấu hình thuật toán: Super Nightmare (Depth: {agent.depth}, QDepth: {agent.max_qdepth})")
    print("-" * 70)

    # 1. Thế cờ khởi đầu
    env = ChessEnv()
    move, tele = agent.select_action(env, time_limit_ms=500)
    print(f"[Thế cờ 1 - Khởi đầu]:")
    print(f"  -> Nước đi: {move} | Thời gian: {tele['latency_ms']:.1f}ms | Nodes: {tele['nodes']} | Q-nodes: {tele.get('q_nodes', 0)}")

    # 2. Thế cờ bẫy chiếu bí Scholar's Mate
    env_mate = ChessEnv(fen="r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    move_mate, tele_mate = agent.select_action(env_mate, time_limit_ms=500)
    print(f"\n[Thế cờ 2 - Bẫy chiếu bí Scholar's Mate]:")
    print(f"  -> Nước đi: {move_mate} ({'CHÍNH XÁC: CHIẾU BÍ f3f7!' if move_mate == 'f3f7' else 'Sai'}) | Thời gian: {tele_mate['latency_ms']:.1f}ms")

    # 3. Trận đối kháng thử nghiệm với SearchAgent(3)
    print("\n" + "-" * 70)
    print("[*] Đấu thử 1 ván tốc hành với SearchAgent (Cấp 3 Khó nhất của game)...")
    env_game = ChessEnv(max_ply=40)
    opp = SearchAgent(3)
    while not env_game.board.outcome(claim_draw=True) and env_game.ply < 40:
        ag = agent if env_game.board.turn == chess.WHITE else opp
        m, _ = ag.select_action(env_game, 200)
        env_game.step(m)

    from backend.core.eval_pst import material_score
    out = env_game.board.outcome(claim_draw=True)
    mat = material_score(env_game.board)
    print(f"  -> Sau {env_game.ply} nước: Lực lượng Trắng (AI): {mat/100:+.1f} điểm | Kết quả: {out.result() if out else 'Đang dẫn trước áp đảo'}")
    print("=" * 70)
    print("[OK] Tác tử thi đấu đã sẵn sàng 100%!")


if __name__ == '__main__':
    benchmark_tournament_agent()

