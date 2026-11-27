import random
 
from algorithms.algorithms import load_play_agent
from model import device
from model_presets import model_run_name
 
 
class Player:
    is_human = False
    name = "Player"
 
    def choose_move(self, board):
        raise NotImplementedError
 
    def choose_moves(self, boards):
        return [self.choose_move(board) for board in boards]

 
class HumanPlayer(Player):
    is_human = True
    name = "Human"
 
 
class RandomPlayer(Player):
    name = "Random mover"
 
    def choose_move(self, board):
        return random.choice(list(board.legal_moves))
 
 
class ModelPlayer(Player):
 
    def __init__(self, color, path, greedy=False, lookahead=False):
        self.color, self.greedy = color, greedy
        self.agent = load_play_agent(path, device)
        self.lookahead = lookahead and self.agent.supports_lookahead()
        self.name = f"{self.agent.label} ({model_run_name(path)})"
        if lookahead:
            self.name += " + lookahead" if self.lookahead else " (no lookahead)"
 
    def choose_move(self, board):
        return self.agent.choose_move(board, self.greedy, self.lookahead)
 
    def choose_moves(self, boards):
        if self.lookahead and self.agent.supports_lookahead():
            return [self.choose_move(board) for board in boards]
        return self.agent.select_moves(boards, self.greedy)

 
def make_player(spec, color, greedy=False, lookahead=False):
    if spec == "Human":
        return HumanPlayer()
    if spec == "Random mover":
        return RandomPlayer()
    if spec.startswith("file:"):
        return ModelPlayer(color, spec[5:], greedy, lookahead)
    raise ValueError(f"unknown player spec {spec!r}")
