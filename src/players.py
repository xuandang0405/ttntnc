import os
import random

from algorithms import load_agent
from model import device
 
class Player:
    is_human = False
    name = "Player"
 
    def choose_move(self, board):
        raise NotImplementedError


class HumanPlayer(Player):
    is_human = True
    name = "Human"


class RandomPlayer(Player):
    name = "Random mover"

    def choose_move(self, board):
        return random.choice(list(board.legal_moves))


class ModelPlayer(Player):
    def __init__(self, color, path, greedy=False):
        self.color, self.greedy = color, greedy
        self.agent = load_agent(path, color, device)
        self.name = f"{self.agent.label} ({os.path.basename(path)})"

    def choose_move(self, board):
        return self.agent.select_move(board, self.greedy)


def make_player(spec, color, greedy=False):
    if spec == "Human":
        return HumanPlayer()
    if spec == "Random mover":
        return RandomPlayer()
    if spec.startswith("file:"):
        return ModelPlayer(color, spec[5:], greedy)
    raise ValueError(f"unknown player spec {spec!r}")