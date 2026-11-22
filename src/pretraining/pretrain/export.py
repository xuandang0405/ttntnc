from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algorithms", nargs="+", default=["alphazero"])
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import chess
    from algorithms.algorithms import make_agent          # your project
    state = torch.load(a.ckpt, map_location="cpu")["net"]
    for algo in a.algorithms:
        for color in (chess.WHITE, chess.BLACK):
            agent = make_agent(algo, color, torch.device(a.device))
            agent.net.load_state_dict(state)
            agent.save(a.out)
            print(f"saved {algo} / {'white' if color else 'black'}")


if __name__ == "__main__":
    main()
