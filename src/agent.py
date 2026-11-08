import os
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
 
from checkpoints import latest_checkpoint
from config import GAMMA, LR, SHAPING
from encoding import encode, legal_moves_map, material_balance
from model import PolicyValueNet, device
 
MAX_GRAD_NORM = 1.0
ENTROPY_COEF = 0.01


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
        self.metrics = None
        self.reset_episode()
 
    def reset_episode(self):
        self.logps, self.values, self.entropies, self.balances = [], [], [], []
 
    def act(self, board):
        moves = legal_moves_map(board, self.color)
        idx = list(moves.keys())
        logits, value = self.net(encode(board, self.color).to(device))
        dist = torch.distributions.Categorical(logits=logits[0, idx])
        choice = dist.sample()
        self.logps.append(dist.log_prob(choice))
        self.entropies.append(dist.entropy())
        self.values.append(value[0])
        self.balances.append(material_balance(board, self.color))
        return moves[idx[choice.item()]]
 
    def learn(self, final_balance, terminal):
        n = len(self.logps)
        if n == 0:
            return
        bals = self.balances + [final_balance]
        rewards = [SHAPING * (bals[i + 1] - bals[i]) for i in range(n)]
        material = sum(rewards)
        rewards[-1] += sum(terminal.values())
        self._log_episode(rewards, material, terminal)
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + GAMMA * g
            returns.append(g)
        returns = torch.tensor(returns[::-1], dtype=torch.float32, device=device)
        self._optimize(torch.stack(self.logps), torch.stack(self.values), returns,
                       torch.stack(self.entropies).mean(), ENTROPY_COEF)
        self.reset_episode()
 
    # shared by the concurrent agent
    def _log_episode(self, rewards, material, terminal):
        if self.metrics:
            self.metrics.record_episode(self.name, {
                "reward": sum(rewards), "steps": len(rewards), "material": material,
                "terminal": sum(terminal.values()), "draw": terminal.get("draw", 0.0),
                "repetition": terminal.get("repetition", 0.0),
                "pos": sum(r > 0 for r in rewards), "neg": sum(r < 0 for r in rewards)})
 
    def _optimize(self, logps, values, returns, entropy, entropy_coef):
        raw_adv = (returns - values).detach()
        adv = raw_adv
        if len(raw_adv) > 1:
            adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-6)
        actor_loss = -(logps * adv).mean()
        critic_loss = 0.5 * F.mse_loss(values, returns.clamp(-1.5, 1.5))
        loss = actor_loss + critic_loss - entropy_coef * entropy
 
        before = parameters_to_vector(self.net.parameters()).detach().clone()
        self.opt.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM).item()
        self.opt.step()
        update_norm = (parameters_to_vector(self.net.parameters()).detach() - before).norm().item()
 
        ret, val = returns.detach(), values.detach()
        var_ret = ret.var(unbiased=False).item()
        explained = 0.0 if var_ret < 1e-8 else 1.0 - (ret - val).var(unbiased=False).item() / var_ret
        self.last_loss = loss.item()
        if self.metrics:
            self.metrics.record_update(self.name, {
                "actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "total_loss": self.last_loss, "policy_entropy": entropy.item(),
                "mean_advantage": raw_adv.mean().item(), "std_advantage": raw_adv.std(unbiased=False).item(),
                "mean_return": ret.mean().item(), "mean_value": val.mean().item(),
                "value_explained_variance": explained, "gradient_norm": grad_norm,
                "gradient_clip_fraction": 1.0 if grad_norm > MAX_GRAD_NORM else 0.0,
                "weight_update_norm": update_norm})
 
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