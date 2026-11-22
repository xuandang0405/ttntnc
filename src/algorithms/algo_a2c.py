import numpy as np
import torch
import torch.nn.functional as F

from algorithms.algo_base import BaseAgent
from algorithms.algo_config import ENTROPY_COEF, USE_RUST_RL
from config import GAMMA
from encoding import ACTION_SIZE, INPUT_CHANNELS, legal_moves_map, material_balance


class A2CAgent(BaseAgent):
    algorithm = "a2c"
    label = "A2C"

    def __init__(self, color, name, device, episodes_per_update=1):
        super().__init__(color, name, device, episodes_per_update)
        self.core = None
        if USE_RUST_RL:             # sampling, discounted returns and legal masks run natively
            self.core = self.native_core(
                "A2cCore", self.state_dim(), ACTION_SIZE, GAMMA,
                int(np.random.randint(0, 2 ** 62, dtype=np.int64)))

    def reset_episode(self):
        super().reset_episode()
        if self.core is not None:
            self.core.clear_episodes()

    def hyperparameters(self):
        return {"gamma": GAMMA, "entropy_coef": ENTROPY_COEF, "episodes_per_update": self.episodes_per_update}

    @torch.no_grad()
    def act_batch(self, game_ids, boards):
        if self.core is not None:
            return self._act_batch_native(game_ids, boards)
        self.prefetch_balances(boards)
        with self._profile("action_prepare"):
            maps, mask = self.legal_batch(boards)
            states = self.encode_boards(boards)
        with self._profile("action_inference"):
            logits, _ = self.net(states.to(self.device))
        with self._profile("action_select"):
            logits = logits.masked_fill(~torch.from_numpy(mask).to(self.device), float("-inf"))
            actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            st = states.numpy().astype(np.float16)
            moves = []
            for i, (g, board) in enumerate(zip(game_ids, boards)):
                a = int(actions[i])
                ep = self.eps.setdefault(g, {"s": [], "a": [], "legal": [], "bal": []})
                ep["s"].append(st[i])
                ep["a"].append(a)
                ep["legal"].append(np.fromiter(maps[i].keys(), dtype=np.int64))
                ep["bal"].append(self.balance_of(board))
                moves.append(maps[i][a])
        return moves

    def _act_batch_native(self, game_ids, boards):
        self.prefetch_balances(boards)
        with self._profile("action_prepare"):
            maps = [legal_moves_map(b, self.color) for b in boards]
            legal = self.legal_padded([list(m) for m in maps])
            states = self.encode_boards(boards)
        with self._profile("action_inference"):
            logits, _ = self.net(states.to(self.device))
            logits = self.gather_legal(logits, legal)    # only legal logits cross to the CPU
        with self._profile("action_select"):
            ids = [int(g) for g in game_ids]
            actions = self.core.act(ids, self.flat_states(states), logits, legal)
            moves = []
            for i, (g, board) in enumerate(zip(ids, boards)):
                self.eps.setdefault(g, {"bal": []})["bal"].append(self.balance_of(board))
                moves.append(maps[i][int(actions[i])])
        return moves

    def _finish_native(self, game_id, final_balance, terminal):
        ep = self.eps.pop(game_id, None)
        if not ep or not ep["bal"]:
            return
        with self._profile("episode_prepare"):
            rewards = self._rewards(ep["bal"], final_balance, terminal)
            self.core.finish_episode(int(game_id), [float(r) for r in rewards])   # discounted returns natively
        if self.core.queued_episodes() >= self.episodes_per_update:
            self.update()

    def finish_episode(self, game_id, final_balance, terminal):
        if self.core is not None:
            return self._finish_native(game_id, final_balance, terminal)
        ep = self.eps.pop(game_id, None)
        if not ep or not ep["a"]:
            return
        with self._profile("episode_prepare"):
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
        dev = self.device
        if self.core is not None:
            batch = self.core.take_batch()
            if batch is None:
                return
            with self._profile("batch_prepare"):
                everything = np.arange(len(batch), dtype=np.int64)
                states, actions, _, _, _, returns, mask = (torch.from_numpy(x).to(dev) for x in batch.gather(everything))
                states = states.view(-1, INPUT_CHANNELS, 8, 8)
                actions = actions.long()
        else:
            if not self.buffer:
                return
            with self._profile("batch_prepare"):
                states = torch.from_numpy(np.concatenate([b[0] for b in self.buffer])).float().to(dev)
                actions = torch.from_numpy(np.concatenate([b[1] for b in self.buffer])).long().to(dev)
                returns = torch.from_numpy(np.concatenate([b[3] for b in self.buffer])).to(dev)
                mask = torch.from_numpy(self.mask_from_legal([l for b in self.buffer for l in b[2]])).to(dev)
                self.buffer = []

        with self._profile("train_forward"):
            logits, values = self.net(states)

        with self._profile("loss_compute"):
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