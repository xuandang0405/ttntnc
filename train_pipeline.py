"""Toàn bộ quy trình AI Cờ Vua: Sinh dữ liệu -> Huấn luyện mô hình -> Đánh giá 3 mức độ (Bình thường / Khó / Ác mộng) -> Lưu Log & Quản lý Thí nghiệm.

Chạy trực tiếp:
    python train_pipeline.py
hoặc:
    python train_pipeline.py --preset super
    python train_pipeline.py --leaderboard
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

# Đảm bảo hiển thị Unicode tiếng Việt mượt mà trên Windows CMD / PowerShell
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.agents.base_agent import BaseAgent
from backend.agents.level1_random import RandomAgent
from backend.agents.search_agent import SearchAgent
from backend.core.env import ChessEnv
from model.learning.agents import LearnedAlphaBetaAgent, NeuralAgent, SuperNightmareAgent
from model.learning.common import setup
from model.learning.data import ChessDataset, collate, split_records
from model.learning.encoding import (
    ENCODER_VERSION,
    encode_board,
    position_key,
    repetition_feature,
)
from model.learning.experiment_tracker import get_champion, log_experiment, print_leaderboard
from model.learning.models import (
    ModelConfig,
    PolicyValueNet,
    load_model,
    masked_logits,
    save_model,
)


def get_difficulty_agent(checkpoint_path: str | Path, level: str = 'normal',
                         device: str = 'cpu') -> tuple[BaseAgent, dict]:
    """
    Khởi tạo tác tử AI theo 3 mức độ chuẩn hóa (kèm chế độ Siêu Ác Mộng):
    - 'normal' / 'binh_thuong': NeuralAgent(mode='policy') - Trực giác nhanh (Depth = 0, ~2ms)
    - 'hard'   / 'kho':         LearnedAlphaBetaAgent(depth=2) - Nhìn trước 2-ply (~100-300ms)
    - 'nightmare' / 'ac_mong':  SuperNightmareAgent(depth=4) - Quiescence Search + Move Ordering (~1000-1400ms)
    """
    ckpt_str = str(checkpoint_path)
    lvl = level.lower().strip()
    if lvl in ('normal', 'binh_thuong', '1', 'de'):
        agent = NeuralAgent(ckpt_str, mode='policy', device=device)
        cfg = {
            'level_name': 'Bình thường (Normal)',
            'algorithm': 'Neural-Policy',
            'depth': 0,
            'budget_ms': 50,
            'description': 'Trực giác mạng nơ-ron, đi nhanh ~2ms, phát triển quân bài bản nhưng có thể bị bẫy'
        }
    elif lvl in ('hard', 'kho', '2'):
        agent = LearnedAlphaBetaAgent(ckpt_str, depth=2, device=device)
        cfg = {
            'level_name': 'Khó (Hard)',
            'algorithm': 'AlphaBeta-NN (Depth 2)',
            'depth': 2,
            'budget_ms': 300,
            'description': 'Alpha-Beta nhìn sâu 2-ply, đánh giá bằng mạng nơ-ron, không dính bẫy ngớ ngẩn'
        }
    elif lvl in ('nightmare', 'ac_mong', '3', 'super'):
        agent = SuperNightmareAgent(ckpt_str, depth=4, max_qdepth=4, device=device)
        cfg = {
            'level_name': 'Ác mộng (Super Nightmare)',
            'algorithm': 'SuperNightmare-ResNet',
            'depth': 4,
            'budget_ms': 1400,
            'description': 'Mạng sâu + Quiescence Search + Move Ordering, triệt tiêu mọi sơ hở chiến thuật'
        }
    else:
        raise ValueError(f"Mức độ không hợp lệ: {level}. Chọn 'normal', 'hard', hoặc 'nightmare'.")

    return agent, cfg


def print_bar(current: int, total: int, prefix: str = "", suffix: str = "", bar_len: int = 28):
    """Vẽ thanh tiến trình trực quan không cần thư viện bên ngoài."""
    total = max(1, total)
    current = min(current, total)
    percent = (float(current) / total) * 100.0
    filled = int(bar_len * current // total)
    bar = "=" * filled + (">" if filled < bar_len else "") + " " * (bar_len - filled - (1 if filled < bar_len else 0))
    msg = f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%) {suffix}"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def find_stockfish() -> Path | None:
    """Tự động tìm engine Stockfish trong thư mục tools/stockfish."""
    sf_dir = Path(__file__).resolve().parent / 'tools' / 'stockfish'
    if sf_dir.exists():
        exes = list(sf_dir.rglob('*.exe'))
        if exes:
            return exes[0]
    return None


def step_generate_data(out_path: Path, games: int = 200, max_plies: int = 80,
                       teacher: str = 'alphabeta', teacher_ms: float = 25.0,
                       exploration: float = 0.15, seed: int = 42,
                       engine_path: str | Path | None = None) -> Path:
    """Giai đoạn 1: Sinh dữ liệu mẫu từ Stockfish Grandmaster hoặc Alpha-Beta Teacher."""
    engine = None
    if teacher == 'stockfish' and engine_path:
        import chess.engine
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        engine.configure({'Threads': 2, 'Hash': 128})
        teacher_label = "Stockfish Grandmaster (Elo 3500+)"
    else:
        teacher = 'alphabeta'
        teacher_agent = SearchAgent(3)
        teacher_label = f"Alpha-Beta Search ({teacher_ms}ms / nước)"

    print("\n" + "=" * 68)
    print(" GIAI ĐOẠN 1: SINH DỮ LIỆU HUẤN LUYỆN (DATA GENERATION)")
    print("=" * 68)
    print(f"[*] Mục tiêu: Sinh {games} ván cờ (tối đa {max_plies} nước/ván)")
    print(f"[*] Giáo viên (Teacher): {teacher_label}")
    print(f"[*] Tỷ lệ khám phá nước đi ngẫu nhiên: {exploration * 100:.0f}% (đa dạng hóa thế cờ)")
    print(f"[*] File dữ liệu đầu ra: {out_path}")
    print("-" * 68)

    out_path = Path(out_path)
    meta_path = out_path.with_suffix(out_path.suffix + '.meta.json')
    if out_path.exists():
        print(f"[!] File dữ liệu đã tồn tại sẵn. Tái sử dụng dữ liệu: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    total_samples = 0
    summaries = []
    start_time = time.time()

    try:
        with out_path.open('w', encoding='utf-8') as stream:
            for game_idx in range(games):
                board = chess.Board()
                opening = []

                # Mở đầu ngẫu nhiên vài nước để mở rộng nhánh thế cờ
                for _ in range(rng.randint(0, min(6, max_plies - 1))):
                    move = rng.choice(list(board.legal_moves))
                    opening.append(move.uci())
                    board.push(move)
                    if board.outcome(claim_draw=True):
                        break

                game_samples = 0
                while len(board.move_stack) < max_plies and not board.outcome(claim_draw=True):
                    if engine:
                        info = engine.analyse(board, chess.engine.Limit(time=max(0.01, teacher_ms / 1000)))
                        proposed = info['pv'][0] if info.get('pv') else list(board.legal_moves)[0]
                        score_pov = info['score'].pov(board.turn)
                        cp = score_pov.score(mate_score=100_000)
                    else:
                        env = ChessEnv(max_ply=100_000)
                        env.board = board.copy(stack=True)
                        env.ply = len(board.move_stack)
                        uci, info = teacher_agent.select_action(env, teacher_ms)
                        proposed = chess.Move.from_uci(uci)
                        cp = info['eval_score'] * 100 * (1 if board.turn else -1)

                    played = (rng.choice(list(board.legal_moves))
                              if rng.random() < exploration else proposed)

                    row = {
                        'game_id': f'g{game_idx:06d}',
                        'fen': board.fen(),
                        'repetition': repetition_feature(board),
                        'target_uci': proposed.uci(),
                        'played_uci': played.uci(),
                        'value': math.tanh(float(cp) / 600),
                        'teacher': teacher,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    total_samples += 1
                    game_samples += 1
                    board.push(played)

                outcome = board.outcome(claim_draw=True)
                summaries.append({
                    'game_id': f'g{game_idx:06d}', 'opening': opening, 'samples': game_samples,
                    'result': outcome.result() if outcome else None,
                    'termination': outcome.termination.name if outcome else 'truncated',
                })

                elapsed = time.time() - start_time
                rate = total_samples / max(0.1, elapsed)
                suffix = f"| Ván {game_idx + 1:3d}: {game_samples:2d} nước | Tích lũy: {total_samples:5d} mẫu ({rate:.1f} m/s)"
                print_bar(game_idx + 1, games, prefix="[Sinh mẫu]", suffix=suffix)
    finally:
        if engine:
            try:
                engine.quit()
            except Exception:
                pass

    elapsed_total = time.time() - start_time
    print(f"\n[OK] Hoàn tất sinh dữ liệu: {total_samples:,} mẫu từ {games} ván đấu (Thời gian: {elapsed_total:.1f}s)")

    metadata = {
        'schema': 1, 'encoder': ENCODER_VERSION, 'games': games, 'rows': total_samples,
        'seed': seed, 'teacher': teacher, 'teacher_ms': teacher_ms,
        'max_plies': max_plies, 'exploration': exploration,
        'value_target': 'tanh(teacher_cp_side_to_move / 600)',
        'sha256': hashlib.sha256(out_path.read_bytes()).hexdigest(), 'game_summary': summaries,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_path


def step_train_model(data_path: Path, out_dir: Path, model_kind: str = 'cnn',
                     epochs: int = 12, batch_size: int = 64, lr: float = 1e-3,
                     device_str: str = 'auto', seed: int = 42, blocks: int = 4):
    """Giai đoạn 2: Huấn luyện mô hình Deep Learning (CNN hoặc ResNet nhiều tầng)."""
    print("\n" + "=" * 68)
    print(" GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH HỌC SÂU (DEEP LEARNING TRAINING)")
    print("=" * 68)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = setup(seed, device_str, threads=4)
    arch_desc = f"ResNet ({blocks} Residual Blocks)" if model_kind == 'resnet' else model_kind.upper()
    dev_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"[*] Kiến trúc mạng: {arch_desc}")
    print(f"[*] Thiết bị tính toán: {device.upper()} ({dev_name})")
    print(f"[*] Tham số: {epochs} Epochs | Batch size: {batch_size} | Learning rate: {lr}")
    print(f"[*] Thư mục lưu checkpoint: {out_dir}")
    print("-" * 68)

    # Đọc và chia tập Train/Validation/Test
    raw = Path(data_path).read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    folds, manifest = split_records(records, seed)

    print("[*] Phân bổ tập dữ liệu (loại bỏ trùng lặp):")
    print(f"    - Tập Huấn luyện (Train):       {manifest['train']['rows']:5d} mẫu ({len(manifest['train']['games'])} ván)")
    print(f"    - Tập Đánh giá (Validation):     {manifest['validation']['rows']:5d} mẫu ({len(manifest['validation']['games'])} ván)")
    print(f"    - Tập Kiểm tra (Test):          {manifest['test']['rows']:5d} mẫu ({len(manifest['test']['games'])} ván)")
    print("-" * 68)

    datasets = [ChessDataset(fold) for fold in folds]
    loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=(i == 0), num_workers=0, collate_fn=collate,
                   generator=torch.Generator().manual_seed(seed))
        for i, ds in enumerate(datasets)
    ]
    train_loader, val_loader, test_loader = loaders

    # Khởi tạo mô hình PolicyValueNet chuẩn
    width = 32
    hidden = 128
    config = ModelConfig(kind=model_kind, width=width, hidden=hidden, blocks=blocks)
    model = PolicyValueNet(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[*] Khởi tạo xong mạng {arch_desc} với tổng số tham số: {param_count:,}")
    print("-" * 68)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    patience = 4

    train_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- HUẤN LUYỆN (TRAIN) ---
        model.train()
        train_loss, train_acc = 0.0, 0.0
        n_train = 0
        total_batches = len(train_loader)

        for b_idx, (x, target, value_target, mask) in enumerate(train_loader, 1):
            x, target = x.to(device), target.to(device)
            value_target, mask = value_target.to(device), mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, value = model(x)
            logits = masked_logits(logits, mask)
            p_loss = F.cross_entropy(logits, target)
            v_loss = F.mse_loss(value, value_target)
            loss = p_loss + 1.0 * v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = len(x)
            n_train += bsz
            train_loss += loss.item() * bsz
            train_acc += (logits.argmax(-1) == target).float().sum().item()

            if b_idx % max(1, total_batches // 5) == 0 or b_idx == total_batches:
                print_bar(b_idx, total_batches, prefix=f"Epoch {epoch:2d}/{epochs:2d} [Train]",
                          suffix=f"Loss: {loss.item():.4f}")

        train_loss /= n_train
        train_acc /= n_train

        # --- ĐÁNH GIÁ (VALIDATION) ---
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        n_val = 0
        with torch.no_grad():
            for x, target, value_target, mask in val_loader:
                x, target = x.to(device), target.to(device)
                value_target, mask = value_target.to(device), mask.to(device)
                logits, value = model(x)
                logits = masked_logits(logits, mask)
                p_loss = F.cross_entropy(logits, target)
                v_loss = F.mse_loss(value, value_target)
                loss = p_loss + 1.0 * v_loss

                bsz = len(x)
                n_val += bsz
                val_loss += loss.item() * bsz
                val_acc += (logits.argmax(-1) == target).float().sum().item()

        val_loss /= n_val
        val_acc /= n_val
        epoch_dur = time.time() - epoch_start

        is_best = val_loss < best_loss
        best_str = ""
        metadata = dict(epoch=epoch, policy_trained=True, method='supervised',
                        seed=seed, validation={'loss': val_loss, 'policy_accuracy': val_acc})
        scheduler.step(val_loss)
        if is_best:
            best_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            save_model(out_dir / 'best.pt', model, **metadata)
            best_str = " -> [LƯU BEST MODEL]"
        else:
            no_improve_count += 1

        save_model(out_dir / 'last.pt', model, **metadata)

        print(f" -> Epoch {epoch:2d}/{epochs:2d} ({epoch_dur:4.1f}s) | "
              f"Train Loss: {train_loss:.4f} (Acc: {train_acc * 100:5.2f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc * 100:5.2f}%){best_str}")

        if no_improve_count >= patience:
            print(f"\n[*] Kích hoạt Early Stopping: Val Loss không cải thiện trong {patience} epochs liên tiếp.")
            print(f"    (Dừng lại đúng điểm rơi phong độ cao nhất tại Epoch {best_epoch}, tránh học vẹt Overfitting).")
            break

    total_train_time = time.time() - train_start
    print("-" * 68)
    print(f"[OK] Huấn luyện hoàn thành trong {total_train_time:.1f}s! Best Epoch: {best_epoch} (Val Loss: {best_loss:.4f})")
    print(f"[*] Checkpoint tối ưu nhất: {out_dir / 'best.pt'}")

    # Ghi file cấu hình 3 mức độ (difficulty_config.json)
    diff_config = {
        'checkpoint': str(out_dir / 'best.pt'),
        'model_kind': model_kind,
        'blocks': blocks if model_kind == 'resnet' else 0,
        'best_epoch': best_epoch,
        'val_loss': best_loss,
        'levels': {
            'normal': {
                'name': 'Bình thường',
                'agent_class': 'NeuralAgent',
                'mode': 'policy',
                'depth': 0,
                'budget_ms': 50,
                'description': 'Trực giác mạng nơ-ron, đi nhanh ~2ms, phát triển quân bài bản'
            },
            'hard': {
                'name': 'Khó',
                'agent_class': 'LearnedAlphaBetaAgent',
                'mode': 'value',
                'depth': 2,
                'budget_ms': 300,
                'description': 'Alpha-Beta nhìn sâu 2 ply, đánh giá bằng mạng nơ-ron, không dính bẫy ngớ ngẩn'
            },
            'nightmare': {
                'name': 'Ác mộng',
                'agent_class': 'SuperNightmareAgent',
                'mode': 'value',
                'depth': 4,
                'budget_ms': 1400,
                'description': 'Siêu AI: ResNet + Quiescence Search + Move Ordering, triệt tiêu mọi sơ hở'
            }
        }
    }
    (out_dir / 'difficulty_config.json').write_text(json.dumps(diff_config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[*] Đã xuất cấu hình 3 mức độ chơi tại: {out_dir / 'difficulty_config.json'}")

    train_stats = {
        'param_count': param_count,
        'manifest': manifest,
        'best_loss': best_loss,
        'best_epoch': best_epoch
    }
    return out_dir / 'best.pt', test_loader, device, train_stats


def step_test_and_evaluate(checkpoint_path: Path, test_loader: DataLoader, device_str: str = 'cpu') -> tuple[float, float]:
    """Giai đoạn 3: Kiểm thử toàn diện và đối sánh 3 cấp độ (Bình thường / Khó / Ác mộng)."""
    print("\n" + "=" * 68)
    print(" GIAI ĐOẠN 3: KIỂM THỬ VÀ ĐỐI SÁNH 3 MỨC ĐỘ (EVALUATION & BENCHMARK)")
    print("=" * 68)
    checkpoint_path = Path(checkpoint_path)
    model, metadata = load_model(str(checkpoint_path), device_str)
    print(f"[*] Nạp checkpoint: {checkpoint_path}")
    print(f"[*] Epoch được chọn: {metadata.get('epoch', 'N/A')}")
    print("-" * 68)

    # 1. Đánh giá trên tập kiểm tra độc lập (Held-out Test Set)
    print("[1] Đánh giá độ chính xác tổng quát trên tập Test:")
    model.eval()
    test_loss, test_p_loss, test_v_loss, test_acc = 0.0, 0.0, 0.0, 0.0
    n_test = 0
    with torch.no_grad():
        for x, target, value_target, mask in test_loader:
            x, target = x.to(device_str), target.to(device_str)
            value_target, mask = value_target.to(device_str), mask.to(device_str)
            logits, value = model(x)
            logits = masked_logits(logits, mask)
            p_loss = F.cross_entropy(logits, target)
            v_loss = F.mse_loss(value, value_target)
            loss = p_loss + 1.0 * v_loss

            bsz = len(x)
            n_test += bsz
            test_loss += loss.item() * bsz
            test_p_loss += p_loss.item() * bsz
            test_v_loss += v_loss.item() * bsz
            test_acc += (logits.argmax(-1) == target).float().sum().item()

    test_loss /= n_test
    test_acc /= n_test
    test_v_loss /= n_test
    print(f"    - Test Loss tổng hợp:       {test_loss:.4f}")
    print(f"    - Test Policy Accuracy:     {test_acc * 100:.2f}% (Tỷ lệ đoán đúng nước giáo viên)")
    print(f"    - Test Value Loss (MSE):    {test_v_loss:.4f} (Độ lệch điểm thế cờ)")
    print("-" * 68)

    # 2. Khởi tạo 3 mức độ AI
    levels = ['normal', 'hard', 'nightmare']
    agents = {}
    configs = {}
    for lvl in levels:
        ag, cfg = get_difficulty_agent(checkpoint_path, level=lvl, device=device_str)
        agents[lvl] = ag
        configs[lvl] = cfg

    # 3. Benchmark phản xạ của 3 mức độ trên thế cờ thực tế
    print("[2] So sánh phản xạ và độ sâu tính toán của 3 cấp độ:")
    test_cases = [
        ("Thế cờ xuất phát ban đầu", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Thế cờ đe dọa chiếu bí Scholar's Mate (f3f7)", "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    ]
    for case_name, fen in test_cases:
        print(f"\n    === {case_name} ===")
        env = ChessEnv(fen=fen)
        for lvl in levels:
            ag = agents[lvl]
            cfg = configs[lvl]
            budget = cfg['budget_ms']
            m, telemetry = ag.select_action(env, time_limit_ms=budget)
            nodes = telemetry.get('nodes', 0)
            q_nodes = telemetry.get('q_nodes', 0)
            latency = telemetry.get('latency_ms', 0)
            depth = telemetry.get('depth', 0)
            q_str = f" | Q-nodes: {q_nodes:<4}" if q_nodes > 0 else ""
            print(f"    * [{cfg['level_name']}]:")
            print(f"      - Nước chọn: {m:<6} | Nodes: {nodes:<5}{q_str} | Độ sâu: {depth} | Thời gian: {latency:6.2f} ms")

    print("\n" + "-" * 68)

    # 4. Đấu kiểm nghiệm thực chiến (Mini-Arena đối kháng cho từng mức độ)
    print("[3] Đấu thực chiến Mini-Arena đối đầu bot Random (Mỗi cấp độ đấu 1 ván):")
    arena_wins = 0
    total_matches = len(levels)

    for match_idx, lvl in enumerate(levels, 1):
        ag = agents[lvl]
        cfg = configs[lvl]
        env = ChessEnv(max_ply=80)
        random_agent = RandomAgent(seed=100 + match_idx)

        # Cấp Normal cầm Trắng, Hard cầm Đen, Nightmare cầm Trắng
        ai_is_white = (match_idx % 2 != 0)
        ai_color = chess.WHITE if ai_is_white else chess.BLACK

        while not env.board.outcome(claim_draw=True) and env.ply < 80:
            if env.board.turn == ai_color:
                m, _ = ag.select_action(env, time_limit_ms=cfg['budget_ms'])
            else:
                m, _ = random_agent.select_action(env)
            env.step(m)

        outcome = env.board.outcome(claim_draw=True)
        if outcome is None:
            # Hết trần 80 ply: Phân định thắng thua theo lực lượng (theo luật đề tài)
            from backend.core.eval_pst import material_score
            mat_diff = material_score(env.board) * (1 if ai_is_white else -1)
            if mat_diff >= 300:
                res_str = f"AI THẮNG (Hơn lực lượng +{mat_diff/100:.1f} điểm)"
                arena_wins += 1
            elif mat_diff <= -300:
                res_str = f"AI THUA (Kém lực lượng {mat_diff/100:.1f} điểm)"
            else:
                res_str = f"HÒA CỜ (Chênh lệch lực lượng < 3 điểm)"
                arena_wins += 0.5
        elif outcome.winner == ai_color:
            res_str = f"AI THẮNG ({outcome.termination.name})"
            arena_wins += 1
        elif outcome.winner is None:
            res_str = f"HÒA CỜ ({outcome.termination.name})"
            arena_wins += 0.5
        else:
            res_str = f"AI THUA ({outcome.termination.name})"

        color_str = "Trắng (Đi trước)" if ai_is_white else "Đen (Đi sau)"
        print(f"    * {cfg['level_name']:<28} (Cầm {color_str}) -> {res_str} sau {env.ply} nước")

    win_rate = (arena_wins / total_matches) * 100.0
    print(f"\n[OK] Tỷ lệ thành công thực chiến Arena: {win_rate:.1f}%")
    print("=" * 68)
    return test_acc * 100.0, win_rate


def parse_arguments():
    p = argparse.ArgumentParser(description="Chương trình huấn luyện và đánh giá AI Cờ Vua theo 3 cấp độ")
    p.add_argument('--preset', choices=['balanced', 'quick', 'super', 'champion', 'custom'], default=None,
                   help="Bộ cấu hình sẵn: balanced (CNN 200 ván), quick (CNN 30 ván), super/champion (ResNet + Stockfish)")
    p.add_argument('--model', choices=['cnn', 'mlp', 'resnet'], default=None,
                   help="Kiến trúc mạng nơ-ron (cnn, mlp hoặc resnet)")
    p.add_argument('--blocks', type=int, default=4,
                   help="Số khối Residual Block nếu dùng ResNet (mặc định: 4)")
    p.add_argument('--teacher', choices=['alphabeta', 'stockfish'], default=None,
                   help="Giáo viên sinh dữ liệu (alphabeta hoặc stockfish)")
    p.add_argument('--games', type=int, default=None,
                   help="Số ván đấu sinh dữ liệu")
    p.add_argument('--epochs', type=int, default=None,
                   help="Số epoch huấn luyện")
    p.add_argument('--batch-size', type=int, default=None,
                   help="Kích thước batch (mặc định: 64)")
    p.add_argument('--lr', type=float, default=1e-3,
                   help="Tốc độ học Learning rate (mặc định: 0.001)")
    p.add_argument('--data', type=str, default=None,
                   help="Đường dẫn file dữ liệu có sẵn (nếu muốn tái sử dụng)")
    p.add_argument('--out', type=str, default=None,
                   help="Thư mục lưu mô hình huấn luyện")
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help="Thiết bị tính toán (auto, cpu, cuda)")
    p.add_argument('--teacher-ms', type=float, default=25.0,
                   help="Thời gian tính toán mỗi nước của bot giáo viên (ms)")
    p.add_argument('--leaderboard', '--experiments', action='store_true',
                   help="Xem bảng xếp hạng và nhật ký các thí nghiệm đã thực hiện")
    return p.parse_args()


def main():
    args = parse_arguments()

    # Nếu yêu cầu xem bảng xếp hạng thì in ngay và thoát
    if args.leaderboard:
        print_leaderboard()
        return

    # Xử lý Preset hoặc Menu tương tác
    blocks = args.blocks
    sf_exe = find_stockfish()
    teacher = 'stockfish' if sf_exe else 'alphabeta'
    engine_path = sf_exe if sf_exe else None
    teacher_ms = args.teacher_ms

    if args.preset == 'super' or args.preset == 'champion':
        model_kind, games, epochs, batch_size, blocks = 'resnet', 300, 12, 64, 4
        if sf_exe:
            teacher = 'stockfish'
            engine_path = sf_exe
            teacher_ms = 15.0
    elif args.preset == 'balanced':
        model_kind, games, epochs, batch_size = 'cnn', 200, 12, 64
        teacher = 'alphabeta'
    elif args.preset == 'quick':
        model_kind, games, epochs, batch_size = 'cnn', 30, 5, 32
        teacher = 'alphabeta'
    elif args.model or args.games or args.epochs:
        model_kind = args.model or 'cnn'
        games = args.games or 200
        epochs = args.epochs or 12
        batch_size = args.batch_size or 64
        teacher = args.teacher or ('stockfish' if sf_exe and model_kind == 'resnet' else 'alphabeta')
    else:
        # Menu tương tác thân thiện
        print("\n" + "=" * 68)
        print("          CHƯƠNG TRÌNH HUẤN LUYỆN VÀ THIẾT LẬP AI CỜ VUA")
        print("                 (3 CẤP ĐỘ: BÌNH THƯỜNG - KHÓ - ÁC MỘNG)")
        hw_tag = f"GPU: {torch.cuda.get_device_name(0)} (CUDA ĐÃ SẴN SÀNG)" if torch.cuda.is_available() else "CHẾ ĐỘ: CPU"
        print(f"             [{hw_tag}]")
        if sf_exe:
            print("         [ENGINE: STOCKFISH ELO 3500+ ĐÃ TÍCH HỢP SẴN]")
        print("=" * 68)
        print("Vui lòng chọn chế độ chạy (Nhấn [Enter] để dùng cấu hình khuyến nghị):\n")
        print("  [1] HUẤN LUYỆN CHUẨN (CNN - 200 ván, 12 Epochs)")
        print("      - Cấu hình cân bằng, đã kiểm chứng chạy mượt mà ~3 phút\n")
        print("  [2] CHẠY THỬ NGHIỆM NHANH (Quick Smoke Test)")
        print("      - 30 ván dữ liệu | 5 Epochs | Hoàn tất trong ~40 giây\n")
        if sf_exe:
            print("  [3] SIÊU MÔ HÌNH VÔ ĐỊCH - ĐI THI ĐẤU (ResNet + Stockfish Elo 3500+)")
            print("      - Dùng Stockfish Đại kiện tướng sinh 300 ván mẫu chuẩn xác tuyệt đối")
            print("      - Mạng nơ-ron ResNet 4 tầng sâu + Quiescence Search + GPU RTX 3050")
            print("      - Sức mạnh áp đảo, triệt hạ mọi sai lầm để giành chức vô địch!\n")
        else:
            print("  [3] SIÊU MÔ HÌNH ÁC MỘNG (ResNet Deep Residual Network)")
            print("      - Mạng nơ-ron phần dư ResNet 4 tầng sâu + Quiescence Search\n")
        print("  [4] XEM BẢNG XẾP HẠNG & NHẬT KÝ THÍ NGHIỆM (Leaderboard)")
        print("      - Xem lịch sử các mô hình đã train, tỷ lệ thắng và Champion hiện tại\n")
        print("  [5] TÙY CHỈNH THỦ CÔNG (Custom Configuration)")

        choice_preset = input("\n -> Chọn chế độ [1/2/3/4/5, Mặc định: 1]: ").strip()

        if choice_preset == '4':
            print_leaderboard()
            return
        elif choice_preset == '2':
            model_kind, games, epochs, batch_size = 'cnn', 30, 5, 32
            teacher = 'alphabeta'
        elif choice_preset == '3':
            model_kind, games, epochs, batch_size, blocks = 'resnet', 300, 12, 64, 4
            if sf_exe:
                teacher = 'stockfish'
                engine_path = sf_exe
                teacher_ms = 15.0
        elif choice_preset == '5':
            choice_model = input("  * Chọn kiến trúc [1. CNN / 2. ResNet (Sâu) / 3. MLP]: ").strip()
            if choice_model == '2':
                model_kind = 'resnet'
                blocks = 4
            elif choice_model == '3':
                model_kind = 'mlp'
            else:
                model_kind = 'cnn'
            choice_games = input("  * Số ván sinh dữ liệu [Mặc định: 200]: ").strip()
            games = int(choice_games) if choice_games.isdigit() and int(choice_games) > 2 else 200
            choice_epochs = input("  * Số Epochs huấn luyện [Mặc định: 12]: ").strip()
            epochs = int(choice_epochs) if choice_epochs.isdigit() and int(choice_epochs) > 0 else 12
            batch_size = args.batch_size or 64
        else:
            # Mặc định là Khuyến nghị chuẩn cho 3 mức độ
            model_kind, games, epochs, batch_size = 'cnn', 200, 12, 64
            teacher = 'alphabeta'

    # Thiết lập đường dẫn thư mục theo thời gian
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = Path(f"data/training/dataset_{model_kind}_{timestamp}.jsonl")

    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(f"experiments/{model_kind}_3levels_{timestamp}")

    # GIAI ĐOẠN 1: SINH DỮ LIỆU
    data_file = step_generate_data(
        out_path=data_path,
        games=games,
        max_plies=80,
        teacher=teacher,
        teacher_ms=teacher_ms,
        engine_path=engine_path,
        seed=42
    )

    # GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH
    best_checkpoint, test_loader, device_used, train_stats = step_train_model(
        data_path=data_file,
        out_dir=out_dir,
        model_kind=model_kind,
        epochs=epochs,
        batch_size=batch_size,
        lr=args.lr,
        device_str=args.device,
        seed=42,
        blocks=blocks
    )

    # GIAI ĐOẠN 3: TEST & ĐỐI SÁNH 3 CẤP ĐỘ
    test_acc, win_rate = step_test_and_evaluate(
        checkpoint_path=best_checkpoint,
        test_loader=test_loader,
        device_str=device_used
    )

    # GHI LOG THÍ NGHIỆM VÀO HỆ THỐNG QUẢN LÝ
    exp_record = log_experiment(
        exp_id=out_dir.name,
        model_kind=model_kind,
        parameters=train_stats['param_count'],
        samples=train_stats['manifest']['train']['rows'] + train_stats['manifest']['validation']['rows'] + train_stats['manifest']['test']['rows'],
        epochs=epochs,
        best_epoch=train_stats['best_epoch'],
        val_loss=train_stats['best_loss'],
        test_acc=test_acc,
        win_rate=win_rate,
        checkpoint_path=str(best_checkpoint),
        notes="Siêu mô hình ResNet" if model_kind == 'resnet' else "Chuẩn 3 cấp độ CNN"
    )

    print("\n" + "=" * 68)
    print("      CHÚC MỪNG! TOÀN BỘ QUY TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print(f"[*] Checkpoint mô hình: {best_checkpoint}")
    print(f"[*] File cấu hình 3 mức độ: {out_dir / 'difficulty_config.json'}")
    print(f"[*] Thí nghiệm đã được ghi vào hệ thống: ID = {exp_record['exp_id']}")
    print("=" * 68)

    # In Bảng Xếp Hạng Thí Nghiệm
    print_leaderboard()


if __name__ == '__main__':
    main()
