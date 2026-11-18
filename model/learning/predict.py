"""Inspect one trained decision without starting the web server."""
import argparse
import json
import chess
from backend.core.env import ChessEnv
from model.learning.agents import LearnedAlphaBetaAgent, NeuralAgent
from model.learning.common import setup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--fen', default=chess.STARTING_FEN)
    p.add_argument('--agent', choices=['policy', 'value', 'search'], default='policy')
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--time-ms', type=float, default=100)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    device = setup(42, args.device, args.threads)
    env = ChessEnv(max_ply=100_000, fen=args.fen)
    agent = (LearnedAlphaBetaAgent(args.checkpoint, args.depth, device)
             if args.agent == 'search' else NeuralAgent(args.checkpoint, args.agent, device))
    action, telemetry = agent.select_action(env, args.time_ms)
    print(json.dumps({'uci': action, 'telemetry': telemetry, 'device': device}, indent=2))


if __name__ == '__main__':
    main()
