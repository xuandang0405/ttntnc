"""Stockfish teacher: generate labelled positions, then pretrain a model on them.

  python pretrain_stockfish.py generate --out data/sf.npz --games 400 --workers 4
  python pretrain_stockfish.py train --data data/sf.npz --algorithm AlphaZero --color both

The saved checkpoints (white_<algo>.pt / black_<algo>.pt) load in the app like any other
model, and training can then continue with self-play (optionally with the Stockfish coach:
CHESS_AI_STOCKFISH_COACH=1).
"""
import argparse
import os
import time

import chess
import torch

from stockfish.stockfish_config import (TEACHER_BATCH, TEACHER_EPOCHS, TEACHER_MAX_PLIES, TEACHER_MULTIPV,
                              TEACHER_NODES)


def cmd_generate(args):
    from stockfish.stockfish_teacher import generate_dataset
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    started = time.time()
    n = generate_dataset(args.out, args.games, args.workers, args.nodes, args.multipv,
                         args.max_plies, args.seed, args.stockfish)
    print(f"saved {n} labelled positions -> {args.out}  ({time.time() - started:.0f}s)")


def cmd_train(args):
    from algorithms.algorithms import load_agent, make_agent
    from stockfish.stockfish_teacher import TeacherData, pretrain
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    colors = {"white": [chess.WHITE], "black": [chess.BLACK], "both": [chess.WHITE, chess.BLACK]}[args.color]
    if args.init and len(colors) > 1:
        raise SystemExit("--init continues one checkpoint, so use --color white or --color black")
    data = TeacherData.load(args.data)
    print(f"{data.n} positions, device {device}")

    first = (load_agent(args.init, colors[0], device) if args.init
             else make_agent(args.algorithm, colors[0], device))
    if args.init:
        print(f"continuing from {args.init} ({first.label})")
    pretrain(first, data, args.epochs, args.batch, args.lr)

    out = args.out or os.path.join("checkpoints", time.strftime("stockfish_pretrain_%Y%m%d_%H%M%S"))
    paths = [first.save(out)]
    for color in colors[1:]:                         # same weights for the other colour
        other = make_agent(args.algorithm, color, device)
        other.net.load_state_dict(first.net.state_dict())
        other.opt.load_state_dict(first.opt.state_dict())
        other.strategy.load(first.strategy.state())
        other.load_extra(first.extra_state())
        paths.append(other.save(out))
    for p in paths:
        print("saved", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="play Stockfish-guided games and label every position")
    g.add_argument("--out", required=True)
    g.add_argument("--games", type=int, default=200)
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    g.add_argument("--nodes", type=int, default=TEACHER_NODES, help="Stockfish nodes per position")
    g.add_argument("--multipv", type=int, default=TEACHER_MULTIPV)
    g.add_argument("--max-plies", type=int, default=TEACHER_MAX_PLIES)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--stockfish", help="path to the Stockfish executable (or set STOCKFISH_PATH)")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="supervised pretraining on a labelled dataset")
    t.add_argument("--data", required=True)
    t.add_argument("--algorithm", default="AlphaZero", choices=["A2C", "PPO", "DQN", "AlphaZero"])
    t.add_argument("--color", default="both", choices=["white", "black", "both"])
    t.add_argument("--init", help="continue from an existing checkpoint instead of a fresh network")
    t.add_argument("--epochs", type=int, default=TEACHER_EPOCHS)
    t.add_argument("--batch", type=int, default=TEACHER_BATCH)
    t.add_argument("--lr", type=float, default=None, help="learning rate for this run (default: agent's)")
    t.add_argument("--device", default="auto")
    t.add_argument("--out", help="output directory (default checkpoints/stockfish_pretrain_<time>)")
    t.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
