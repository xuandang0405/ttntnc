import numpy as np
import torch
import torch.nn.functional as F

from algo_base import BaseAgent
from algo_config import ENTROPY_COEF, GAE_LAMBDA, PPO_CLIP, PPO_EPOCHS, PPO_MINIBATCH
from config import GAMMA
from encoding import material_balance


class PPOAgent(BaseAgent):
    algorithm = "ppo"
    label = "PPO"

    def hyperparameters(self):
        return {"gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "clip": PPO_CLIP, "epochs": PPO_EPOCHS,
                "minibatch": PPO_MINIBATCH, "entropy_coef": ENTROPY_COEF,
                "episodes_per_update": self.episodes_per_update}

    @torch.no_grad()
    def act_batch(self, game_ids, boards):
        maps, mask = self.legal_batch(boards)
        states = self.encode_boards(boards)
        logits, values = self.net(states.to(self.device))
        logits = logits.masked_fill(~torch.from_numpy(mask).to(self.device), float("-inf"))
        logp_all = F.log_softmax(logits, dim=-1)
        actions = torch.multinomial(logp_all.exp(), 1).squeeze(1)
        logps = logp_all.gather(1, actions[:, None]).squeeze(1)
        actions, logps, values = actions.cpu().numpy(), logps.cpu().numpy(), values.cpu().numpy()
        st = states.numpy().astype(np.uint8)
        moves = []
        for i, (g, board) in enumerate(zip(game_ids, boards)):
            a = int(actions[i])
            ep = self.eps.setdefault(g, {"s": [], "a": [], "logp": [], "v": [], "legal": [], "bal": []})
            ep["s"].append(st[i])
            ep["a"].append(a)
            ep["logp"].append(float(logps[i]))
            ep["v"].append(float(values[i]))
            ep["legal"].append(np.fromiter(maps[i].keys(), dtype=np.int64))
            ep["bal"].append(material_balance(board, self.color))
            moves.append(maps[i][a])
        return moves

    def finish_episode(self, game_id, final_balance, terminal):
        ep = self.eps.pop(game_id, None)
        if not ep or not ep["a"]:
            return
        rewards = self._rewards(ep["bal"], final_balance, terminal)
        v = np.array(ep["v"], dtype=np.float64)
        n = len(v)
        adv, last = np.zeros(n), 0.0
        for t in reversed(range(n)):                       # generalised advantage estimation
            next_v = v[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + GAMMA * next_v - v[t]
            last = delta + GAMMA * GAE_LAMBDA * last
            adv[t] = last
        self.buffer.append((np.stack(ep["s"]), np.array(ep["a"]), np.array(ep["logp"], dtype=np.float32),
                            adv.astype(np.float32), (adv + v).astype(np.float32), ep["legal"]))
        if len(self.buffer) >= self.episodes_per_update:
            self.update()

    def update(self):
        if not self.buffer:
            return
        dev = self.device
        legal = [l for b in self.buffer for l in b[5]]
        states = torch.from_numpy(np.concatenate([b[0] for b in self.buffer])).float().to(dev)
        actions = torch.from_numpy(np.concatenate([b[1] for b in self.buffer])).long().to(dev)
        old_logp = torch.from_numpy(np.concatenate([b[2] for b in self.buffer])).to(dev)
        raw_adv = torch.from_numpy(np.concatenate([b[3] for b in self.buffer])).to(dev)
        returns = torch.from_numpy(np.concatenate([b[4] for b in self.buffer])).to(dev)
        self.buffer = []
        n = len(actions)
        adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8) if n > 1 else raw_adv

        for _ in range(PPO_EPOCHS):
            perm = np.random.permutation(n)
            for start in range(0, n, PPO_MINIBATCH):
                idx = perm[start:start + PPO_MINIBATCH]
                it = torch.from_numpy(idx).to(dev)
                mask = torch.from_numpy(self.mask_from_legal([legal[i] for i in idx])).to(dev)
 
                logits, values = self.net(states[it])
                logp_all = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
                logps = logp_all.gather(1, actions[it][:, None]).squeeze(1)
                safe = logp_all.masked_fill(~mask, 0.0)
                entropy = -(safe.exp() * safe).sum(-1).mean()
 
                ratio = torch.exp(logps - old_logp[it])
                a = adv[it]
                surrogate = torch.min(ratio * a, torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * a)
                actor_loss = -surrogate.mean()
                critic_loss = 0.5 * F.mse_loss(values, returns[it].clamp(-1.5, 1.5))
                loss = actor_loss + critic_loss - ENTROPY_COEF * entropy
                grad_norm, update_norm = self._apply(loss)
                self._record(actor_loss, critic_loss, loss, entropy, raw_adv[it], returns[it], values,
                             grad_norm, update_norm)

    def select_move(self, board, greedy=False):
        return self.policy_move(board, greedy)