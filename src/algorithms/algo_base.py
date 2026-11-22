from contextlib import contextmanager
import os
import time
 
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
 
from algorithms.algo_config import MAX_GRAD_NORM
from config import LR, SHAPING
from encoding import ACTION_SIZE, HISTORY_LENGTH, INPUT_CHANNELS, encode, legal_moves_map, material_balance
from model import PolicyValueNet
from stockfish.stockfish_coach import get_coach
from strategy import StrategyLearner, lookahead_move

try:
    import algorithms_rs
except ImportError:
    algorithms_rs = None
 

class BaseAgent:
    algorithm = "base"    # lower-case key used in file names, e.g. white_ppo.pt
    label = "Base"        # name shown in the UI
    play_temperature = 1.0
 
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
        self.coach = get_coach()                # Stockfish reward-shaping coach (None unless CHESS_AI_STOCKFISH_COACH=1)
 
    def act_batch(self, game_ids, boards):
        raise NotImplementedError
 
    def finish_episode(self, game_id, final_balance, terminal):
        raise NotImplementedError

    @torch.inference_mode()
    def select_move(self, board, greedy=False):
        raise NotImplementedError
 
    @torch.inference_mode()
    def select_moves(self, boards, greedy=False):
        if not boards:
            return []

        was_training = self.net.training
        self.net.eval()
        try:
            maps = [legal_moves_map(board, self.color) for board in boards]
            states = self.encode_boards(boards).to(self.device)
            logits, _ = self.net(states)
            legal_indices = [list(moves) for moves in maps]
            max_legal = max(map(len, legal_indices))
            indices = torch.zeros(
                (len(boards), max_legal), dtype=torch.long, device=self.device)
            valid = torch.zeros_like(indices, dtype=torch.bool)
            for row, actions in enumerate(legal_indices):
                indices[row, :len(actions)] = torch.tensor(
                    actions, dtype=torch.long, device=self.device)
                valid[row, :len(actions)] = True
            logits = logits.gather(1, indices).masked_fill(~valid, float("-inf"))
            if greedy:
                selections = logits.argmax(dim=1)
            else:
                selections = torch.distributions.Categorical(
                    logits=logits / self.play_temperature).sample()
            return [moves[legal_indices[i][selection]]
                    for i, (moves, selection) in enumerate(zip(maps, selections.tolist()))]
        finally:
            self.net.train(was_training)

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
 
    # native (Rust) policy / replay cores
    def native_core(self, name, *args):
        if algorithms_rs is None or not hasattr(algorithms_rs, name):
            raise RuntimeError(
                f"Rust RL was requested, but the algorithms_rs extension with {name} is not installed. "
                "Build it with `python -m pip install .` from src/algorithms-rs, or set CHESS_AI_RUST_RL=0.")
        return getattr(algorithms_rs, name)(*args)

    @staticmethod
    def legal_padded(legal_lists):
        out = np.full((len(legal_lists), max(map(len, legal_lists))), -1, dtype=np.int32)
        for i, legal in enumerate(legal_lists):
            out[i, :len(legal)] = legal
        return out

    @staticmethod
    def gather_legal(scores, legal):
        idx = torch.from_numpy(legal).to(scores.device).long().clamp_(min=0)
        return np.ascontiguousarray(scores.gather(1, idx).float().cpu().numpy())

    @staticmethod
    def flat_states(states):
        return np.ascontiguousarray(states.numpy(), dtype=np.float32).reshape(len(states), -1)

    @staticmethod
    def state_dim():
        return INPUT_CHANNELS * 8 * 8

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
    # The shaping potential at each of our own decision points: material balance, or Stockfish's
    # evaluation (in pawns) when the coach is on.
    def balance_of(self, board):
        if self.coach is not None:
            return self.coach.potential(board, self.color)
        return material_balance(board, self.color)

    def prefetch_balances(self, boards):      # start the Stockfish evaluations early; they overlap with inference
        if self.coach is not None:
            self.coach.prefetch(boards)

    def _final_balance(self, final_balance):
        # A finished game has potential 0 (the terminal reward carries the result), which keeps
        # potential-based shaping consistent; material mode keeps the old behaviour.
        return 0.0 if self.coach is not None else final_balance

    def _rewards(self, bal, final_balance, terminal):
        bals = list(bal) + [self._final_balance(final_balance)]
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
 
    @contextmanager
    def _profile(self, part):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._record_profile_time(part, time.perf_counter() - started)

    def _record_profile_time(self, part, elapsed_seconds):
        if self.metrics is not None:
            self.metrics.record_timing(self.name, part, elapsed_seconds)

    def _apply(self, loss):
        with self._profile("backward"):
            before = parameters_to_vector(self.net.parameters()).detach().clone()
            self.opt.zero_grad()
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM))
        with self._profile("optimizer_step"):
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
                "Checkpoint uses an incompatible network/encoding. The new 56-plane "
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