import numpy as np
import torch
import torch.nn.functional as F

from algo_base import BaseAgent
from algo_config import ENTROPY_COEF
from config import GAMMA
from encoding import material_balance


class A2CAgent(BaseAgent):
    algorithm = "a2c"
    label = "A2C"

    def hyperparameters(self):
        return {"gamma": GAMMA, "entropy_coef": ENTROPY_COEF, "episodes_per_update": self.episodes_per_update}

    @torch.no_grad()
    def act_batch(self, game_ids, boards):
        maps, mask = self.legal_batch(boards)
        states = self.encode_boards(boards)
        logits, _ = self.net(states.to(self.device))
        logits = logits.masked_fill(~torch.from_numpy(mask).to(self.device), float("-inf"))
        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        st = states.numpy().astype(np.uint8)
        moves = []
        for i, (g, board) in enumerate(zip(game_ids, boards)):
            a = int(actions[i])
            ep = self.eps.setdefault(g, {"s": [], "a": [], "legal": [], "bal": []})
            ep["s"].append(st[i])
            ep["a"].append(a)
            ep["legal"].append(np.fromiter(maps[i].keys(), dtype=np.int64))
            ep["bal"].append(material_balance(board, self.color))
            moves.append(maps[i][a])
        return moves

    def finish_episode(self, game_id, final_balance, terminal):
        ep = self.eps.pop(game_id, None)
        if not ep or not ep["a"]:
            return
        rewards = self._rewards(ep["bal"], final_balance, terminal)
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + GAMMA * g
            returns.append(g)
        self.buffer.append((np.stack(ep["s"]), np.array(ep["a"]), ep["legal"],
                            np.array(returns[::-1], dtype=np.float32)))
        if len(self.buffer) >= self.episodes_per_update:
            self.update()

    def update(self):
        # One actor-critic step over every queued game.
        if not self.buffer:
            return
        dev = self.device
        states = torch.from_numpy(np.concatenate([b[0] for b in self.buffer])).float().to(dev)
        actions = torch.from_numpy(np.concatenate([b[1] for b in self.buffer])).long().to(dev)
        returns = torch.from_numpy(np.concatenate([b[3] for b in self.buffer])).to(dev)
        mask = torch.from_numpy(self.mask_from_legal([l for b in self.buffer for l in b[2]])).to(dev)
        self.buffer = []

        logits, values = self.net(states)
        logp_all = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        logps = logp_all.gather(1, actions[:, None]).squeeze(1)
        safe = logp_all.masked_fill(~mask, 0.0)
        entropy = -(safe.exp() * safe).sum(-1).mean()

        raw_adv = (returns - values).detach()
        adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-6) if len(raw_adv) > 1 else raw_adv
        actor_loss = -(logps * adv).mean()
        critic_loss = 0.5 * F.mse_loss(values, returns.clamp(-1.5, 1.5))
        loss = actor_loss + critic_loss - ENTROPY_COEF * entropy
        grad_norm, update_norm = self._apply(loss)
        self._record(actor_loss, critic_loss, loss, entropy, raw_adv, returns, values, grad_norm, update_norm)

    def select_move(self, board, greedy=False):
        return self.policy_move(board, greedy)