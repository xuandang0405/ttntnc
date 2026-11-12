import os
 
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
 
from algo_config import MAX_GRAD_NORM
from config import LR, SHAPING
from encoding import encode, legal_moves_map
from model import PolicyValueNet


class BaseAgent:
    algorithm = "base"    # lower-case key used in file names, e.g. white_ppo.pt
    label = "Base"        # name shown in the UI
 
    def __init__(self, color, name, device, episodes_per_update=1):
        self.color, self.name, self.device = color, name, device
        self.episodes_per_update = episodes_per_update
        self.net = PolicyValueNet().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.last_loss = 0.0
        self.loaded_from = None
        self.metrics = None      # TrainingMetrics, attached by the game
        self.eps = {}            # game id -> in-progress episode
        self.buffer = []         # finished episodes waiting for an update
 
    def act_batch(self, game_ids, boards):
        raise NotImplementedError
 
    def finish_episode(self, game_id, final_balance, terminal):
        raise NotImplementedError
 
    def select_move(self, board, greedy=False):
        raise NotImplementedError
 
    def hyperparameters(self):
        return {}
 
    def extra_state(self):       # saved in the checkpoint next to the weights
        return {}
 
    def load_extra(self, extra):
        pass
 
    def reset_episode(self):
        self.eps.clear()
 
    def act(self, board):
        return self.act_batch([0], [board])[0]
 
    def learn(self, final_balance, terminal):
        self.finish_episode(0, final_balance, terminal)
 
    def encode_boards(self, boards):
        return torch.cat([encode(b, self.color) for b in boards])
 
    def legal_batch(self, boards):
        maps = [legal_moves_map(b, self.color) for b in boards]
        mask = np.zeros((len(boards), 4096), dtype=bool)
        for i, m in enumerate(maps):
            mask[i, list(m.keys())] = True
        return maps, mask
 
    @staticmethod
    def mask_from_legal(legal_lists):
        mask = np.zeros((len(legal_lists), 4096), dtype=bool)
        for r, legal in enumerate(legal_lists):
            mask[r, legal] = True
        return mask
 
    @torch.no_grad()
    def policy_move(self, board, greedy):
        color = board.turn
        moves = legal_moves_map(board, color)
        idx = list(moves.keys())
        logits, _ = self.net(encode(board, color).to(self.device))
        legal_logits = logits[0, idx]
        if greedy:
            i = int(legal_logits.argmax())
        else:
            i = int(torch.distributions.Categorical(logits=legal_logits).sample())
        return moves[idx[i]]
 
    # rewards / metrics
    def _rewards(self, bal, final_balance, terminal):
        bals = list(bal) + [final_balance]
        rewards = [SHAPING * (bals[i + 1] - bals[i]) for i in range(len(bal))]
        material = sum(rewards)
        rewards[-1] += sum(terminal.values())
        if self.metrics:
            self.metrics.record_episode(self.name, {
                "reward": sum(rewards), "steps": len(rewards), "material": material,
                "terminal": sum(terminal.values()), "draw": terminal.get("draw", 0.0),
                "repetition": terminal.get("repetition", 0.0),
                "pos": sum(r > 0 for r in rewards), "neg": sum(r < 0 for r in rewards)})
        return rewards
 
    def _apply(self, loss):
        before = parameters_to_vector(self.net.parameters()).detach().clone()
        self.opt.zero_grad()
        loss.backward()
        grad_norm = float(nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM))
        self.opt.step()
        update_norm = float((parameters_to_vector(self.net.parameters()).detach() - before).norm())
        return grad_norm, update_norm
 
    def _record(self, actor_loss, critic_loss, total_loss, entropy, adv, ret, val, grad_norm, update_norm):
        ret, val, adv = ret.detach().float(), val.detach().float(), adv.detach().float()
        var_ret = float(ret.var(unbiased=False)) if len(ret) > 1 else 0.0
        explained = 0.0 if var_ret < 1e-8 else 1.0 - float((ret - val).var(unbiased=False)) / var_ret
        self.last_loss = float(total_loss)
        if self.metrics:
            self.metrics.record_update(self.name, {
                "actor_loss": float(actor_loss), "critic_loss": float(critic_loss),
                "total_loss": self.last_loss, "policy_entropy": float(entropy),
                "mean_advantage": float(adv.mean()), "std_advantage": float(adv.std(unbiased=False)),
                "mean_return": float(ret.mean()), "mean_value": float(val.mean()),
                "value_explained_variance": explained, "gradient_norm": grad_norm,
                "gradient_clip_fraction": 1.0 if grad_norm > MAX_GRAD_NORM else 0.0,
                "weight_update_norm": update_norm})
 
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name.lower()}_{self.algorithm}.pt")
        torch.save({"format": 2, "algorithm": self.algorithm, "name": self.name,
                    "net": self.net.state_dict(), "opt": self.opt.state_dict(),
                    "extra": self.extra_state()}, path)
        return path
 
    def load_state(self, ck):
        self.net.load_state_dict(ck["net"])
        try:
            self.opt.load_state_dict(ck["opt"])
        except Exception:
            pass                       # keep a fresh optimizer if the saved one doesn't fit
        self.load_extra(ck.get("extra", {}))