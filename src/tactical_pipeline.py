# import random
# from typing import Dict, List, Optional

# import chess
# import numpy as np
# import torch
# import torch.nn.functional as F

# from encoding import ACTION_SIZE, encode, legal_moves_map, move_to_index
# from strategy_config import (
#     TACTICAL_BATCH,
#     TACTICAL_PIPELINE_ENABLED,
#     TACTICAL_TRAIN_EVERY_GAMES,
#     TACTICAL_WEIGHT,
# )
# from tactics_library import (
#     PATTERNS_BY_CATEGORY,
#     TacticalPattern,
#     VERIFIED_PATTERNS,
#     instantiate_pattern,
# )


# def _legal_mask(legal_lists: List[List[int]]) -> np.ndarray:
#     mask = np.zeros((len(legal_lists), ACTION_SIZE), dtype=bool)
#     for row, legal in enumerate(legal_lists):
#         mask[row, legal] = True
#     return mask


# class TacticalTrainer:

#     def __init__(self, agent):
#         self.agent = agent
#         self._games = 0
#         self.drill_steps = 0
#         self.rng = random.Random(42 if agent.color == chess.WHITE else 1337)
#         # Track exponential moving accuracy per pattern_id to prioritize harder motifs
#         self.mastery: Dict[str, float] = {p.pattern_id: 0.5 for p in VERIFIED_PATTERNS}

#     def state(self) -> dict:
#         return {
#             "drill_steps": self.drill_steps,
#             "mastery": dict(self.mastery),
#         }

#     def load(self, state: Optional[dict]) -> None:
#         if not state:
#             return
#         self.drill_steps = int(state.get("drill_steps", 0))
#         saved_mastery = state.get("mastery")
#         if isinstance(saved_mastery, dict):
#             for k, v in saved_mastery.items():
#                 if k in self.mastery:
#                     self.mastery[k] = float(v)

#     def _sample_patterns(self, k: int) -> List[TacticalPattern]:
#         if not VERIFIED_PATTERNS:
#             return []
#         # Inverse-mastery weighting: patterns the agent fails get sampled more often
#         weights = [max(0.15, 1.15 - self.mastery.get(p.pattern_id, 0.5)) for p in VERIFIED_PATTERNS]
#         return self.rng.choices(VERIFIED_PATTERNS, weights=weights, k=k)

#     def train_after_game(self) -> None:
#         if not TACTICAL_PIPELINE_ENABLED or not VERIFIED_PATTERNS:
#             return
#         self._games += 1
#         if self._games < TACTICAL_TRAIN_EVERY_GAMES:
#             return
#         self._games = 0
#         self.train_batch(TACTICAL_BATCH)

#     def train_batch(self, batch_size: int = TACTICAL_BATCH) -> Optional[dict]:
#         patterns = self._sample_patterns(batch_size)
#         if not patterns:
#             return None

#         dev = self.agent.device
#         states = []
#         target_policies = np.zeros((len(patterns), ACTION_SIZE), dtype=np.float32)
#         target_values = []
#         target_phases = []
#         legal_lists = []
#         pattern_refs: List[TacticalPattern] = []
#         best_idx_sets: List[set] = []

#         for row, pat in enumerate(patterns):
#             board, best_moves, target_val, phase = instantiate_pattern(pat, rng=self.rng, allow_mirror=True)
#             color = board.turn
#             lmap = legal_moves_map(board, color)
#             if not lmap:
#                 continue
#             best_indices = [move_to_index(m, color) for m in best_moves if move_to_index(m, color) in lmap]
#             if not best_indices:
#                 continue

#             states.append(encode(board, color)[0].numpy())
#             weight_each = 1.0 / len(best_indices)
#             for idx in best_indices:
#                 target_policies[row, idx] = weight_each
#             target_values.append(float(target_val))
#             target_phases.append(int(phase))
#             legal_lists.append(list(lmap.keys()))
#             pattern_refs.append(pat)
#             best_idx_sets.append(set(best_indices))

#         if not states:
#             return None

#         n = len(states)
#         s_t = torch.from_numpy(np.stack(states)).float().to(dev)
#         pi_t = torch.from_numpy(target_policies[:n]).to(dev)
#         val_t = torch.tensor(target_values, dtype=torch.float32, device=dev)
#         ph_t = torch.tensor(target_phases, dtype=torch.long, device=dev)
#         mask_t = torch.from_numpy(_legal_mask(legal_lists)).to(dev)

#         logits, values = self.agent.net(s_t)
#         masked_logits = logits.masked_fill(~mask_t, float("-inf"))
#         logp = F.log_softmax(masked_logits, dim=-1)
#         safe_logp = logp.masked_fill(~mask_t, 0.0)
#         policy_loss = -(pi_t * safe_logp).sum(dim=-1).mean()

#         # Value calibration: only train value head on algorithms that use it (PPO/A2C/AlphaZero)
#         if self.agent.algorithm != "dqn":
#             value_loss = F.mse_loss(values, val_t)
#         else:
#             value_loss = torch.tensor(0.0, device=dev)

#         _, phase_logits = self.agent.net.forward_aux(s_t)
#         phase_loss = F.cross_entropy(phase_logits, ph_t)

#         total_loss = TACTICAL_WEIGHT * (policy_loss + 0.5 * value_loss + 0.25 * phase_loss)
#         self.agent._apply(total_loss)
#         self.drill_steps += 1

#         # Update per-pattern mastery estimate using current top-1 prediction
#         with torch.no_grad():
#             preds = masked_logits.argmax(dim=-1).cpu().tolist()
#             solved = 0
#             for pat, pred_idx, valid_set in zip(pattern_refs, preds, best_idx_sets):
#                 hit = 1.0 if pred_idx in valid_set else 0.0
#                 solved += int(hit)
#                 old = self.mastery.get(pat.pattern_id, 0.5)
#                 self.mastery[pat.pattern_id] = 0.85 * old + 0.15 * hit

#         stats = {
#             "tactical_acc": solved / max(1, n),
#             "tactical_loss": float(total_loss),
#         }
#         if self.agent.metrics:
#             self.agent.metrics.record_strategy(self.agent.name, stats)
#         return stats


# @torch.no_grad()
# def evaluate_tactical_benchmark(agent) -> dict:
#     by_cat_hits = {"attack": 0, "defense": 0, "endgame": 0}
#     by_cat_total = {"attack": 0, "defense": 0, "endgame": 0}
#     motif_hits: Dict[str, int] = {}
#     motif_total: Dict[str, int] = {}

#     for pat in VERIFIED_PATTERNS:
#         board = chess.Board(pat.fen)
#         expected = set(pat.best_moves)
#         # Use greedy policy move for fast deterministic evaluation (or agent.select_move if non-AZ)
#         if agent.algorithm == "alphazero":
#             move = agent.policy_move(board, greedy=True)
#         else:
#             move = agent.select_move(board, greedy=True)
#         ok = int(move.uci() in expected)

#         by_cat_hits[pat.category] += ok
#         by_cat_total[pat.category] += 1
#         motif_hits[pat.motif] = motif_hits.get(pat.motif, 0) + ok
#         motif_total[pat.motif] = motif_total.get(pat.motif, 0) + 1

#     total_hits = sum(by_cat_hits.values())
#     total_count = max(1, sum(by_cat_total.values()))

#     return {
#         "overall": round(total_hits / total_count, 3),
#         "attack": round(by_cat_hits["attack"] / max(1, by_cat_total["attack"]), 3),
#         "defense": round(by_cat_hits["defense"] / max(1, by_cat_total["defense"]), 3),
#         "endgame": round(by_cat_hits["endgame"] / max(1, by_cat_total["endgame"]), 3),
#         "motifs": {
#             m: round(motif_hits[m] / max(1, motif_total[m]), 3)
#             for m in sorted(motif_total)
#         },
#         "counts": dict(by_cat_total),
#     }
