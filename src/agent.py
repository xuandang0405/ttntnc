import os
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from checkpoints import latest_checkpoint
from config import GAMMA, LR, SHAPING
from encoding import encode, legal_moves_map, material_balance
from model import PolicyValueNet, device
 
 
def models_available():
    return {n: latest_checkpoint(n) is not None for n in ("White", "Black")}
 
 
def load_policy(name):
    net = PolicyValueNet().to(device)
    path = latest_checkpoint(name)      # newest experiment
    trained = path is not None
    if trained:
        net.load_state_dict(torch.load(path, map_location=device)["net"])
    net.eval()
    return net, trained
 
 
class Agent:
    def __init__(self, color, name):
        self.color, self.name = color, name
        self.net = PolicyValueNet().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.last_loss = 0.0
        self.loaded_from = None
        self.reset_episode()
 
    def reset_episode(self):
        self.logps, self.values, self.entropies, self.balances = [], [], [], []
 
    def act(self, board):
        moves = legal_moves_map(board, self.color)
        idx = list(moves.keys())
        logits, value = self.net(encode(board, self.color).to(device))
        dist = torch.distributions.Categorical(logits=logits[0, idx])  # legal moves only
        choice = dist.sample()
        self.logps.append(dist.log_prob(choice))
        self.entropies.append(dist.entropy())
        self.values.append(value[0])
        self.balances.append(material_balance(board, self.color))
        return moves[idx[choice.item()]]
 
    def learn(self, final_balance, terminal_reward):
        n = len(self.logps)
        if n == 0:
            return
        bals = self.balances + [final_balance]
        rewards = [SHAPING * (bals[i + 1] - bals[i]) for i in range(n)]
        rewards[-1] += terminal_reward
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + GAMMA * g
            returns.append(g)
        returns = torch.tensor(returns[::-1], dtype=torch.float32, device=device)
 
        logps = torch.stack(self.logps)
        values = torch.stack(self.values)
        ent = torch.stack(self.entropies).mean()
        adv = (returns - values).detach()
        if n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        loss = (-(logps * adv).mean()
                + 0.5 * F.mse_loss(values, returns.clamp(-1.5, 1.5))
                - 0.01 * ent)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        self.last_loss = loss.item()
        self.reset_episode()
 
    # checkpoints
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name.lower()}.pt")
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict()}, path)
        return path
 
    def load(self):
        path = latest_checkpoint(self.name)
        if path is None:
            return False
        ck = torch.load(path, map_location=device)
        self.net.load_state_dict(ck["net"])
        self.opt.load_state_dict(ck["opt"])
        self.loaded_from = path
        return True