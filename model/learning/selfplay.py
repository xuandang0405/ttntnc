"""TD(0) fine-tuning: truncated games bootstrap; they are not labelled draws."""
import argparse
import copy
import json
import random
import chess
import torch
from torch.nn import functional as F
from model.learning.agents import terminal_score, value_moves
from model.learning.common import fresh_directory, setup, write_json
from model.learning.encoding import encode_board
from model.learning.models import load_model, save_model


def td_target(next_value, terminal_reward=None, gamma=1.):
    if terminal_reward is not None:
        return torch.full_like(next_value, terminal_reward)
    return -gamma * next_value.detach()


def selfplay(args):
    if args.episodes < 1 or args.max_plies < 1 or args.target_sync < 1:
        raise ValueError('Số tập, số ply và target_sync phải dương.')
    if not 0 <= args.epsilon <= 1 or not 0 <= args.gamma <= 1 or args.lr <= 0:
        raise ValueError('Thông số TD không hợp lệ.')
    device = setup(args.seed, args.device, args.threads)
    model, _ = load_model(args.init, device)
    target_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    out = fresh_directory(args.out)
    write_json(out / 'run.json', {**vars(args), 'device_used': device, 'method': 'TD(0)-value'})
    updates = 0
    with (out / 'episodes.jsonl').open('w', encoding='utf-8') as stream:
        for episode in range(args.episodes):
            board = chess.Board()
            total_loss = 0.
            count = 0
            for _ in range(args.max_plies):
                if board.outcome(claim_draw=True):
                    break
                x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
                mover = board.turn
                if rng.random() < args.epsilon:
                    move = rng.choice(list(board.legal_moves))
                else:
                    scores = value_moves(model, board, device)
                    move = max(scores, key=lambda m: (scores[m], m.uci()))
                board.push(move)
                reward = terminal_score(board, mover)
                with torch.no_grad():
                    nx = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
                    wanted = td_target(target_model.value(nx), reward, args.gamma)
                model.train()
                loss = F.mse_loss(model.value(x), wanted)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total_loss += loss.item()
                count += 1
                updates += 1
                if updates % args.target_sync == 0:
                    target_model.load_state_dict(model.state_dict())
                if reward is not None:
                    break
            outcome = board.outcome(claim_draw=True)
            record = {'episode': episode + 1, 'plies': count,
                      'loss': total_loss / max(count, 1),
                      'result': outcome.result() if outcome else None,
                      'termination': outcome.termination.name if outcome else 'truncated'}
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print(json.dumps(record), flush=True)
    save_model(out / 'td-final.pt', model, method='TD(0)-value', policy_trained=False,
               source_checkpoint=str(args.init), seed=args.seed, updates=updates,
               value_target='terminal WDL or negative target-network next value')
    return {'out': str(out), 'updates': updates}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--gamma', type=float, default=1.)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--target-sync', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    print(json.dumps(selfplay(p.parse_args()), indent=2))


if __name__ == '__main__':
    main()
