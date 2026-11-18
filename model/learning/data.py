"""Teacher-labelled JSONL and game-level splits with transposition purging."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import chess
import chess.engine
import numpy as np
import torch
from torch.utils.data import Dataset
from model.learning.encoding import ACTION_SIZE, ENCODER_VERSION, encode_board, move_id
from model.learning.encoding import position_key, repetition_feature


def generate(path, games=100, max_plies=80, seed=42, teacher='alphabeta',
             teacher_ms=30, exploration=.15, engine_path=None):
    if games < 3 or max_plies < 2 or teacher_ms <= 0 or not 0 <= exploration <= 1:
        raise ValueError('Cần games >= 3, max_plies >= 2, teacher_ms > 0 và exploration trong [0,1].')
    if teacher not in ('alphabeta', 'stockfish'):
        raise ValueError('Nguồn nhãn không hợp lệ.')
    if teacher == 'stockfish' and (not engine_path or not Path(engine_path).is_file()):
        raise ValueError('Cần --engine-path trỏ đến engine UCI có sẵn.')
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + '.meta.json')
    if path.exists() or meta_path.exists():
        raise FileExistsError(f'Dữ liệu đã tồn tại: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = (chess.engine.SimpleEngine.popen_uci(str(engine_path))
              if teacher == 'stockfish' else None)
    if engine:
        engine.configure({k: v for k, v in {'Threads': 1, 'Hash': 64}.items()
                          if k in engine.options})
    from backend.agents.search_agent import SearchAgent
    from backend.core.env import ChessEnv
    teacher_agent = SearchAgent(3)
    rng = random.Random(seed)
    rows = 0
    summaries = []
    try:
        with path.open('x', encoding='utf-8') as stream:
            for game_id in range(games):
                board = chess.Board()
                opening = []
                for _ in range(rng.randint(0, min(6, max_plies - 1))):
                    move = rng.choice(list(board.legal_moves))
                    opening.append(move.uci())
                    board.push(move)
                    if board.outcome(claim_draw=True):
                        break
                game_rows = 0
                while len(board.move_stack) < max_plies and not board.outcome(claim_draw=True):
                    if engine:
                        info = engine.analyse(board, chess.engine.Limit(time=teacher_ms / 1000))
                        proposed = info['pv'][0]
                        cp = info['score'].pov(board.turn).score(mate_score=100_000)
                    else:
                        # Preserve history; disable the UI's material adjudication.
                        env = ChessEnv(max_ply=100_000)
                        env.board = board.copy(stack=True)
                        env.ply = len(board.move_stack)
                        uci, info = teacher_agent.select_action(env, teacher_ms)
                        proposed = chess.Move.from_uci(uci)
                        cp = info['eval_score'] * 100 * (1 if board.turn else -1)
                    played = (rng.choice(list(board.legal_moves))
                              if rng.random() < exploration else proposed)
                    row = {
                        'game_id': f'g{game_id:06d}', 'fen': board.fen(),
                        'repetition': repetition_feature(board),
                        'target_uci': proposed.uci(), 'played_uci': played.uci(),
                        'value': math.tanh(float(cp) / 600), 'teacher': teacher,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    rows += 1
                    game_rows += 1
                    board.push(played)
                outcome = board.outcome(claim_draw=True)
                summaries.append({
                    'game_id': f'g{game_id:06d}', 'opening': opening, 'samples': game_rows,
                    'result': outcome.result() if outcome else None,
                    'termination': outcome.termination.name if outcome else 'truncated',
                })
    finally:
        if engine:
            engine.quit()
    metadata = {
        'schema': 1, 'encoder': ENCODER_VERSION, 'games': games, 'rows': rows,
        'seed': seed, 'teacher': teacher, 'teacher_ms': teacher_ms,
        'max_plies': max_plies, 'exploration': exploration,
        'value_target': 'tanh(teacher_cp_side_to_move / 600)',
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'game_summary': summaries,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def split_records(records, seed=42):
    game_ids = sorted({r['game_id'] for r in records})
    if len(game_ids) < 3:
        raise ValueError('Cần ít nhất ba ván để chia train/validation/test.')
    random.Random(seed).shuffle(game_ids)
    n_train = max(1, min(len(game_ids) - 2, int(len(game_ids) * .7)))
    n_val = max(1, int(len(game_ids) * .15))
    groups = [set(game_ids[:n_train]), set(game_ids[n_train:n_train + n_val]),
              set(game_ids[n_train + n_val:])]
    folds = []
    previous_keys = set()
    removed = []
    for group in groups:
        candidates = [r for r in records if r['game_id'] in group]
        kept = [r for r in candidates if position_key(r['fen']) not in previous_keys]
        removed.append(len(candidates) - len(kept))
        if not kept:
            raise ValueError('Một tập rỗng sau lọc trùng. Tăng số ván và độ đa dạng dữ liệu.')
        folds.append(kept)
        previous_keys.update(position_key(r['fen']) for r in candidates)
    manifest = {
        name: {'games': sorted(group), 'rows': len(fold), 'removed_overlap': count}
        for name, group, fold, count in zip(('train', 'validation', 'test'), groups, folds, removed)
    }
    return folds, manifest


class ChessDataset(Dataset):
    def __init__(self, records):
        self.samples = []
        for row in records:
            board = chess.Board(row['fen'])
            move = chess.Move.from_uci(row['target_uci'])
            value = float(row['value'])
            repetition = float(row['repetition'])
            if (not board.is_valid() or move not in board.legal_moves
                    or not math.isfinite(value) or not -1 <= value <= 1
                    or repetition not in (0, .5, 1)):
                raise ValueError(f'Mẫu dữ liệu không hợp lệ: {row}')
            self.samples.append((
                encode_board(board, repetition), move_id(move), value,
                [move_id(m) for m in board.legal_moves],
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(samples):
    x, targets, values, legal = zip(*samples)
    mask = torch.zeros((len(samples), ACTION_SIZE), dtype=torch.bool)
    for i, ids in enumerate(legal):
        mask[i, ids] = True
    return (torch.from_numpy(np.stack(x)), torch.tensor(targets, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32), mask)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--max-plies', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--teacher', choices=['alphabeta', 'stockfish'], default='alphabeta')
    p.add_argument('--teacher-ms', type=float, default=30)
    p.add_argument('--exploration', type=float, default=.15)
    p.add_argument('--engine-path')
    args = p.parse_args()
    info = generate(args.out, args.games, args.max_plies, args.seed, args.teacher,
                    args.teacher_ms, args.exploration, args.engine_path)
    print(json.dumps({k: v for k, v in info.items() if k != 'game_summary'}, indent=2))


if __name__ == '__main__':
    main()
