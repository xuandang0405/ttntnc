import random
import torch

from agent import load_policy
from encoding import encode, legal_moves_map
from model import device

PLAYER_CHOICES = ["White model", "Black model", "Random mover"]

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
    def __init__(self, color, source, greedy=False):
        self.color, self.greedy = color, greedy
        self.net, self.trained = load_policy(source)
        self.name = f"{source} model" + ("" if self.trained else " (untrained)")

    @torch.no_grad()
    def choose_move(self, board):
        moves = legal_moves_map(board, self.color)
        idx = list(moves.keys())
        logits, _ = self.net(encode(board, self.color).to(device))
        legal_logits = logits[0, idx]
        if self.greedy:
            i = legal_logits.argmax().item()
        else:
            i = torch.distributions.Categorical(logits=legal_logits).sample().item()
        return moves[idx[i]]


def make_player(spec, color, greedy=False):
    if spec == "Human":
        return HumanPlayer()
    if spec == "Random mover":
        return RandomPlayer()
    return ModelPlayer(color, spec.split()[0], greedy)