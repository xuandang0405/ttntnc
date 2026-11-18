import os
 
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
 
from algorithms.algo_config import MAX_GRAD_NORM
from config import LR, SHAPING
from encoding import ACTION_SIZE, HISTORY_LENGTH, INPUT_CHANNELS, encode, legal_moves_map
from model import PolicyValueNet
from strategy import StrategyLearner, lookahead_move
 

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
        self.strategy = StrategyLearner(self)   # opponent model + phase head (auxiliary learning)
 
    def act_batch(self, game_ids, boards):
        raise NotImplementedError
 
    def finish_episode(self, game_id, final_balance, terminal):
        raise NotImplementedError

    @torch.inference_mode()
    def select_move(self, board, greedy=False):
        raise NotImplementedError
 
    def hyperparameters(self):
        return {}
 
    def extra_state(self):       # saved in the checkpoint next to the weights
        return {}
 
    def load_extra(self, extra):
        pass
 
    # play-time move choice
    def supports_lookahead(self):
        return self.algorithm != "alphazero" and self.strategy.ready
 
    def choose_move(self, board, greedy=False, lookahead=False):
        if lookahead and self.supports_lookahead():
            return lookahead_move(self, board, greedy)
        return self.select_move(board, greedy)
 
    # single-game adapters
    def reset_episode(self):
        self.eps.clear()           # single-game mode: forget a half-played game
 
    def act(self, board):
        return self.act_batch([0], [board])[0]
 
    def learn(self, final_balance, terminal):
        self.finish_episode(0, final_balance, terminal)
 
    # batching helpers
    def encode_boards(self, boards):
        return torch.cat([encode(b, self.color) for b in boards])
 
    def legal_batch(self, boards):
        maps = [legal_moves_map(b, self.color) for b in boards]
        mask = np.zeros((len(boards), ACTION_SIZE), dtype=bool)
        for i, m in enumerate(maps):
            mask[i, list(m.keys())] = True
        return maps, mask
 
    @staticmethod
    def mask_from_legal(legal_lists):
        mask = np.zeros((len(legal_lists), ACTION_SIZE), dtype=bool)
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
 
    # checkpoints: <colour>_<algorithm>.pt
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name.lower()}_{self.algorithm}.pt")
        torch.save({"format": 3, "algorithm": self.algorithm, "name": self.name,
                    "encoding": {"input_channels": INPUT_CHANNELS, "action_size": ACTION_SIZE, "history_length": HISTORY_LENGTH},
                    "net": self.net.state_dict(), "opt": self.opt.state_dict(),
                    "extra": {**self.extra_state(), "strategy": self.strategy.state()}}, path)
        return path
 
    def load_state(self, ck):
        try:
            missing, unexpected = self.net.load_state_dict(ck["net"], strict=False)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint uses an incompatible network/encoding. The new 104-plane "
                "Residual CNN and 20,480-action policy require fresh training; old "
                "weights cannot be loaded directly."
            ) from exc
        if unexpected or any(not k.startswith(("opp.", "phase.")) for k in missing):
            raise ValueError("checkpoint does not match this network")   # (older files lack the strategy heads)
        try:
            self.opt.load_state_dict(ck["opt"])
        except Exception:
            pass                       # keep a fresh optimizer if the saved one doesn't fit
        extra = ck.get("extra", {})
        self.strategy.load(extra.get("strategy"))
        self.load_extra(extra)