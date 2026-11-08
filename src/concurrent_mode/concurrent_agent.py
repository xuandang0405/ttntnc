import numpy as np
import torch
import torch.nn.functional as F
 
from agent import Agent
from concurrent_mode.concurrent_config import ENTROPY_COEF, EPISODES_PER_UPDATE, USE_CUDA
from config import GAMMA, LR, SHAPING
from encoding import encode, legal_moves_map, material_balance
 
device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

class ConcurrentAgent(Agent):
    def __init__(self, color, name):
        super().__init__(color, name)
        self.net.to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)   # rebuilt for the new device
        self.episodes = {}   # game slot id -> in-progress trajectory
        self.buffer = []
 
    # acting: one batched forward pass for all games where it is this agent's turn
    @torch.no_grad()
    def act_batch(self, game_ids, boards):
        maps = [legal_moves_map(b, self.color) for b in boards]
        states = torch.cat([encode(b, self.color) for b in boards])          # (B,16,8,8) on CPU
        mask = np.zeros((len(boards), 4096), dtype=bool)
        for i, m in enumerate(maps):
            mask[i, list(m.keys())] = True
 
        logits, _ = self.net(states.to(device))
        logits = logits.masked_fill(~torch.from_numpy(mask).to(device), float("-inf"))
        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
 
        moves = []
        for i, (g, board) in enumerate(zip(game_ids, boards)):
            a = int(actions[i])
            ep = self.episodes.setdefault(g, {"s": [], "a": [], "legal": [], "bal": []})
            ep["s"].append(states[i].clone())
            ep["a"].append(a)
            ep["legal"].append(np.fromiter(maps[i].keys(), dtype=np.int64))
            ep["bal"].append(material_balance(board, self.color))
            moves.append(maps[i][a])
        return moves
 
    # learning
    def finish_episode(self, game_id, final_balance, terminal):
        ep = self.episodes.pop(game_id, None)
        if not ep or not ep["a"]:
            return
        n = len(ep["a"])
        bals = ep["bal"] + [final_balance]
        rewards = [SHAPING * (bals[i + 1] - bals[i]) for i in range(n)]
        material = sum(rewards)
        rewards[-1] += sum(terminal.values())
        self._log_episode(rewards, material, terminal)
        returns, g = [], 0.0
        for r in reversed(rewards):
            g = r + GAMMA * g
            returns.append(g)
        self.buffer.append((torch.stack(ep["s"]), np.array(ep["a"]), ep["legal"],
                            np.array(returns[::-1], dtype=np.float32)))
        if len(self.buffer) >= EPISODES_PER_UPDATE:
            self.update()
 
    def update(self):
        if not self.buffer:
            return
        states = torch.cat([b[0] for b in self.buffer]).to(device)
        actions = torch.from_numpy(np.concatenate([b[1] for b in self.buffer])).long().to(device)
        returns = torch.from_numpy(np.concatenate([b[3] for b in self.buffer])).to(device)
        mask = np.zeros((len(returns), 4096), dtype=bool)
        row = 0
        for b in self.buffer:
            for legal in b[2]:
                mask[row, legal] = True
                row += 1
        mask = torch.from_numpy(mask).to(device)
        self.buffer = []
 
        logits, values = self.net(states)
        logp_all = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        logps = logp_all.gather(1, actions[:, None]).squeeze(1)
        safe = logp_all.masked_fill(~mask, 0.0)
        entropy = -(safe.exp() * safe).sum(-1).mean()
 
        self._optimize(logps, values, returns, entropy, ENTROPY_COEF)