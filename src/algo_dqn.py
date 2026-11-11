import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from algo_base import BaseAgent
from algo_config import (DQN_BATCH, DQN_BUFFER, DQN_EPS_DECAY_STEPS, DQN_EPS_END, DQN_EPS_START,
                         DQN_PLAY_TEMPERATURE, DQN_TARGET_SYNC, DQN_TRAIN_EVERY, DQN_WARMUP)
from config import GAMMA
from encoding import legal_moves_map, encode, material_balance


class DQNAgent(BaseAgent):
    algorithm = "dqn"
    label = "DQN"
 
    def __init__(self, color, name, device, episodes_per_update=1):
        super().__init__(color, name, device, episodes_per_update)
        self.target = copy.deepcopy(self.net)
        self.replay, self.replay_pos = [], 0
        self.act_steps = 0          # moves chosen so far (drives epsilon)
        self.n_updates = 0          # gradient steps taken
 
    def hyperparameters(self):
        return {"gamma": GAMMA, "buffer": DQN_BUFFER, "batch": DQN_BATCH, "warmup": DQN_WARMUP,
                "target_sync": DQN_TARGET_SYNC, "train_every": DQN_TRAIN_EVERY,
                "eps_start": DQN_EPS_START, "eps_end": DQN_EPS_END, "eps_decay_steps": DQN_EPS_DECAY_STEPS}
 
    def extra_state(self):
        return {"act_steps": self.act_steps, "n_updates": self.n_updates}
 
    def load_extra(self, extra):
        self.act_steps = int(extra.get("act_steps", 0))
        self.n_updates = int(extra.get("n_updates", 0))
        self.target.load_state_dict(self.net.state_dict())

    @property
    def epsilon(self):
        frac = min(1.0, self.act_steps / DQN_EPS_DECAY_STEPS)
        return DQN_EPS_START + (DQN_EPS_END - DQN_EPS_START) * frac

    @torch.no_grad()
    def act_batch(self, game_ids, boards):
        maps, mask = self.legal_batch(boards)
        states = self.encode_boards(boards)
        q, _ = self.net(states.to(self.device))
        q = q.masked_fill(~torch.from_numpy(mask).to(self.device), float("-inf"))
        greedy = q.argmax(dim=1).cpu().numpy()
        st = states.numpy().astype(np.uint8)
        eps = self.epsilon
        moves = []
        for i, (g, board) in enumerate(zip(game_ids, boards)):
            keys = list(maps[i].keys())
            a = random.choice(keys) if random.random() < eps else int(greedy[i])
            ep = self.eps.setdefault(g, {"s": [], "a": [], "legal": [], "bal": []})
            ep["s"].append(st[i])
            ep["a"].append(a)
            ep["legal"].append(np.array(keys, dtype=np.int64))
            ep["bal"].append(material_balance(board, self.color))
            moves.append(maps[i][a])
        self.act_steps += len(boards)
        return moves
 
    def finish_episode(self, game_id, final_balance, terminal):
        ep = self.eps.pop(game_id, None)
        if not ep or not ep["a"]:
            return
        rewards = self._rewards(ep["bal"], final_balance, terminal)
        n = len(ep["a"])
        empty = np.zeros_like(ep["s"][0])
        for t in range(n):                       # (s_t, a_t, r_t, s_{t+1}) - next state is our next turn
            last = t == n - 1
            item = (ep["s"][t], ep["a"][t], float(rewards[t]),
                    empty if last else ep["s"][t + 1],
                    np.zeros(0, dtype=np.int64) if last else ep["legal"][t + 1],
                    last, ep["legal"][t])
            if len(self.replay) < DQN_BUFFER:
                self.replay.append(item)
            else:
                self.replay[self.replay_pos] = item
                self.replay_pos = (self.replay_pos + 1) % DQN_BUFFER
        if len(self.replay) >= DQN_WARMUP:
            for _ in range(max(1, n // DQN_TRAIN_EVERY)):
                self.update()
 
    def update(self):
        dev = self.device
        batch = random.sample(self.replay, min(DQN_BATCH, len(self.replay)))
        s = torch.from_numpy(np.stack([t[0] for t in batch])).float().to(dev)
        a = torch.tensor([t[1] for t in batch], dtype=torch.long, device=dev)
        r = torch.tensor([t[2] for t in batch], dtype=torch.float32, device=dev)
        s2 = torch.from_numpy(np.stack([t[3] for t in batch])).float().to(dev)
        done = torch.tensor([float(t[5]) for t in batch], dtype=torch.float32, device=dev)
        mask2 = torch.from_numpy(self.mask_from_legal([t[4] for t in batch])).to(dev)
        mask1 = torch.from_numpy(self.mask_from_legal([t[6] for t in batch])).to(dev)
 
        q_all, _ = self.net(s)
        q = q_all.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            best = self.net(s2)[0].masked_fill(~mask2, float("-inf")).argmax(dim=1, keepdim=True)
            target = r + GAMMA * (1.0 - done) * self.target(s2)[0].gather(1, best).squeeze(1)
        loss = F.smooth_l1_loss(q, target)
        grad_norm, update_norm = self._apply(loss)
 
        with torch.no_grad():                    # diagnostics: how peaked is softmax(Q / T) over legal moves
            p = torch.softmax((q_all.detach() / DQN_PLAY_TEMPERATURE).masked_fill(~mask1, float("-inf")), dim=-1)
            entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()
        self._record(0.0, loss, loss, entropy, target - q.detach(), target, q, grad_norm, update_norm)
 
        self.n_updates += 1
        if self.n_updates % DQN_TARGET_SYNC == 0:
            self.target.load_state_dict(self.net.state_dict())
 
    @torch.no_grad()
    def select_move(self, board, greedy=False):
        color = board.turn
        moves = legal_moves_map(board, color)
        idx = list(moves.keys())
        q, _ = self.net(encode(board, color).to(self.device))
        q = q[0, idx]
        if greedy:
            i = int(q.argmax())
        else:
            i = int(torch.distributions.Categorical(logits=q / DQN_PLAY_TEMPERATURE).sample())
        return moves[idx[i]]