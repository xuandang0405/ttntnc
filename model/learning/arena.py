"""Paired-color evaluation; move-limit truncations stay separate from draws."""
import argparse
import json
import random
import chess
import chess.pgn
import numpy as np
from backend.agents.level1_random import RandomAgent
from backend.agents.search_agent import SearchAgent
from backend.core.env import ChessEnv
from model.learning.agents import LearnedAlphaBetaAgent, NeuralAgent, SuperNightmareAgent
from model.learning.common import fresh_directory, setup, write_json


def arena(args):
    if args.games < 2 or args.games % 2 or args.max_plies < 1 or args.time_ms <= 0:
        raise ValueError('Số ván phải chẵn và >= 2; max_plies, time_ms phải dương.')
    device = setup(args.seed, args.device, args.threads)
    if args.agent in ('super', 'nightmare'):
        challenger = SuperNightmareAgent(args.checkpoint, depth=args.depth, max_qdepth=4, device=device)
    elif args.agent == 'search':
        challenger = LearnedAlphaBetaAgent(args.checkpoint, args.depth, device)
    else:
        challenger = NeuralAgent(args.checkpoint, args.agent, device)
    out = fresh_directory(args.out)
    results = {'wins': 0, 'draws': 0, 'losses': 0, 'truncated': 0}
    latencies = {'challenger': [], 'opponent': []}
    overruns = {'challenger': 0, 'opponent': 0}
    with (out / 'games.jsonl').open('w', encoding='utf-8') as records, \
            (out / 'games.pgn').open('w', encoding='utf-8') as pgn:
        for index in range(args.games):
            pair = index // 2
            rng = random.Random(args.seed + pair)
            board = chess.Board()
            opening = []
            for _ in range(min(4, args.max_plies - 1)):
                move = rng.choice(list(board.legal_moves))
                opening.append(move.uci())
                board.push(move)
            env = ChessEnv(max_ply=100_000)
            env.board, env.ply = board, len(board.move_stack)
            challenger_color = chess.WHITE if index % 2 == 0 else chess.BLACK
            opponent = RandomAgent(args.seed + pair) if args.opponent == 'random' else SearchAgent(3)
            trace = []
            for _ in range(args.max_plies - env.ply):
                if env.board.outcome(claim_draw=True):
                    break
                is_challenger = env.board.turn == challenger_color
                owner = 'challenger' if is_challenger else 'opponent'
                agent = challenger if is_challenger else opponent
                move, telemetry = agent.select_action(env, args.time_ms)
                if move not in env.legal_actions():
                    raise RuntimeError(f'Tác tử trả nước bất hợp lệ: {move}')
                latency = telemetry['latency_ms']
                latencies[owner].append(latency)
                overruns[owner] += latency > args.time_ms
                trace.append({'owner': owner, 'uci': move, 'latency_ms': latency})
                env.step(move)
            outcome = env.board.outcome(claim_draw=True)
            if outcome is None:
                result = 'truncated'
            elif outcome.winner is None:
                result = 'draws'
            else:
                result = 'wins' if outcome.winner == challenger_color else 'losses'
            results[result] += 1
            record = {'game': index + 1, 'pair': pair, 'seed': args.seed + pair,
                      'challenger_color': 'white' if challenger_color else 'black',
                      'opening': opening, 'result': result,
                      'termination': outcome.termination.name if outcome else 'truncated', 'moves': trace}
            records.write(json.dumps(record) + '\n')
            records.flush()
            game = chess.pgn.Game.from_board(env.board)
            game.headers['Result'] = outcome.result() if outcome else '*'
            game.headers['White'] = 'challenger' if challenger_color else args.opponent
            game.headers['Black'] = args.opponent if challenger_color else 'challenger'
            pgn.write(str(game) + '\n\n')
    finished = results['wins'] + results['draws'] + results['losses']
    summary = {**results, 'finished': finished,
               'completed_score': (results['wins'] + .5 * results['draws']) / finished if finished else None,
               'latency': {owner: {'moves': len(values), 'mean_ms': float(np.mean(values)) if values else None,
                                  'p95_ms': float(np.percentile(values, 95)) if values else None,
                                  'overruns': overruns[owner]} for owner, values in latencies.items()},
               'config': vars(args), 'device_used': device}
    write_json(out / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--agent', choices=['policy', 'value', 'search', 'super', 'nightmare'], default='search')
    p.add_argument('--opponent', choices=['random', 'alphabeta'], default='random')
    p.add_argument('--games', type=int, default=20)
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--time-ms', type=float, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    print(json.dumps(arena(p.parse_args()), indent=2))


if __name__ == '__main__':
    main()
