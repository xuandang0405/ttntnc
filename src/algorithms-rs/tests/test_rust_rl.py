"""Parity tests: native DQN / PPO / A2C cores vs. the reference formulas in
algorithms/algo_{dqn,ppo,a2c}.py. Needs only numpy + the built `algorithms_rs` extension."""
import unittest

import numpy as np

import algorithms_rs as rs

S, A = 6, 20
GAMMA, LAM = 0.97, 0.95


def pad(legal_lists):
    width = max(map(len, legal_lists))
    out = np.full((len(legal_lists), width), -1, dtype=np.int32)
    for i, l in enumerate(legal_lists):
        out[i, :len(l)] = l
    return out


def ref_gae(rewards, v, gamma=GAMMA, lam=LAM):          # algo_ppo.finish_episode
    v = np.array(v, dtype=np.float64)
    n = len(v)
    adv, last = np.zeros(n), 0.0
    for t in reversed(range(n)):
        next_v = v[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_v - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv.astype(np.float32), (adv + v).astype(np.float32)


def ref_returns(rewards, gamma=GAMMA):                  # algo_a2c.finish_episode
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return np.array(out[::-1], dtype=np.float32)


def aligned(full, legal):
    """Gather full (n, A) scores into the (n, L) layout aligned with the padded legal matrix."""
    width = max(map(len, legal))
    out = np.zeros((len(legal), width), np.float32)
    for i, l in enumerate(legal):
        out[i, :len(l)] = full[i, l]
    return out


def make_game(rng, n):
    legal = [np.sort(rng.choice(A, size=rng.integers(2, 9), replace=False)).astype(np.int32) for _ in range(n)]
    return rng.normal(size=(n, S)).astype(np.float32), rng.normal(size=(n, A)).astype(np.float32), legal


class PpoParity(unittest.TestCase):
    def test_gae_matches_python_reference_and_batch_is_normalised(self):
        rng = np.random.default_rng(0)
        core = rs.PpoCore(S, A, GAMMA, LAM, 1)
        lengths, expect_adv, expect_ret, all_states, all_legal = [7, 4], [], [], [], []
        values_by_game, rewards_by_game = {}, {}
        for g, n in enumerate(lengths):
            states, logits, legal = make_game(rng, n)
            vals = rng.normal(size=n).astype(np.float32)
            acts, logps = [], []
            for t in range(n):      # one decision at a time, as in a live game
                a, lp = core.act([g], states[t:t + 1], aligned(logits[t:t + 1], [legal[t]]), vals[t:t + 1], pad([legal[t]]))
                acts.append(int(a[0])); logps.append(float(lp[0]))
                self.assertIn(int(a[0]), legal[t])
                ref = logits[t, legal[t]]
                ref = ref - ref.max(); ref = ref - np.log(np.exp(ref).sum())
                self.assertAlmostEqual(float(lp[0]), float(ref[list(legal[t]).index(a[0])]), places=5)
            rewards = rng.normal(size=n)
            values_by_game[g], rewards_by_game[g] = vals, rewards
            self.assertEqual(core.finish_episode(g, rewards.tolist()), n)
            adv, ret = ref_gae(rewards, vals)
            expect_adv.append(adv); expect_ret.append(ret)
            all_states.append(states); all_legal += legal
        self.assertEqual(core.queued_episodes(), 2)
        batch = core.take_batch()
        self.assertEqual(len(batch), sum(lengths))
        self.assertIsNone(core.take_batch())
        raw = np.concatenate(expect_adv)
        idx = np.arange(len(batch), dtype=np.int64)
        s, a, old_lp, adv, raw_adv, ret, mask = batch.gather(idx)
        np.testing.assert_allclose(raw_adv, raw, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ret, np.concatenate(expect_ret), rtol=1e-5, atol=1e-6)
        torch_style = (raw - raw.mean()) / (raw.std(ddof=1) + 1e-8)       # torch .std() is unbiased
        np.testing.assert_allclose(adv, torch_style, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(s, np.concatenate(all_states).astype(np.float16).astype(np.float32))
        for row, l in enumerate(all_legal):
            self.assertEqual(sorted(np.flatnonzero(mask[row])), sorted(l.tolist()))
        self.assertTrue(np.all(mask[np.arange(len(a)), a]))                # chosen action always legal

    def test_epochs_partition_batch(self):
        rng = np.random.default_rng(1)
        core = rs.PpoCore(S, A, GAMMA, LAM, 2)
        states, logits, legal = make_game(rng, 10)
        core.act([0] * 10, states, aligned(logits, legal), np.zeros(10, np.float32), pad(legal))
        core.finish_episode(0, [0.0] * 9 + [1.0])
        batch = core.take_batch()
        e1, e2 = batch.epoch(4), batch.epoch(4)
        self.assertEqual([len(c) for c in e1], [4, 4, 2])
        self.assertEqual(sorted(np.concatenate(e1)), list(range(10)))
        self.assertNotEqual(list(np.concatenate(e1)), list(np.concatenate(e2)))   # reshuffled each epoch

    def test_bad_inputs_raise(self):
        core = rs.PpoCore(S, A, GAMMA, LAM, 3)
        z = np.zeros((1, S), np.float32)
        l2 = np.zeros((1, 2), np.float32)
        with self.assertRaises(ValueError):       # row with no legal actions
            core.act([0], z, l2, np.zeros(1, np.float32), np.full((1, 2), -1, np.int32))
        with self.assertRaises(ValueError):       # wrong state width
            core.act([0], np.zeros((1, S + 1), np.float32), l2, np.zeros(1, np.float32), pad([[1, 2]]))
        with self.assertRaises(ValueError):       # action index outside the action space
            core.act([0], z, l2, np.zeros(1, np.float32), pad([[1, A]]))
        core.act([0], z, l2, np.zeros(1, np.float32), pad([[1, 2]]))
        with self.assertRaises(ValueError):
            core.finish_episode(0, [0.0, 0.0])           # wrong reward count


class A2cParity(unittest.TestCase):
    def test_returns_and_masks(self):
        rng = np.random.default_rng(4)
        core = rs.A2cCore(S, A, GAMMA, 5)
        states, logits, legal = make_game(rng, 6)
        for t in range(6):
            core.act([9], states[t:t + 1], aligned(logits[t:t + 1], [legal[t]]), pad([legal[t]]))
        rewards = rng.normal(size=6)
        core.finish_episode(9, rewards.tolist())
        b = core.take_batch()
        _, _, _, adv, raw_adv, ret, mask = b.gather(np.arange(6, dtype=np.int64))
        np.testing.assert_allclose(ret, ref_returns(rewards), rtol=1e-5, atol=1e-6)
        self.assertEqual(len(adv), 0)
        self.assertEqual(mask.shape, (6, A))

    def test_policy_samples_match_softmax(self):
        core = rs.A2cCore(S, A, GAMMA, 6)
        logits = np.array([[np.log(3.0), 0.0]], np.float32)                      # legal [3, 8] -> p = .75 / .25
        n = 4000
        picks = [int(core.act([i], np.zeros((1, S), np.float32), logits, pad([[3, 8]]))[0]) for i in range(n)]
        self.assertAlmostEqual(picks.count(3) / n, 0.75, delta=0.03)


class DqnParity(unittest.TestCase):
    def make(self, **kw):
        cfg = dict(state_dim=S, action_size=A, capacity=50, batch=8, warmup=5, train_every=4,
                   eps_start=1.0, eps_end=0.05, eps_decay_steps=100.0, seed=0)
        cfg.update(kw)
        return rs.DqnCore(**cfg)

    def test_epsilon_schedule_matches_python(self):
        core = self.make()
        for steps in (0, 25, 100, 5000):
            core.act_steps = steps
            frac = min(1.0, steps / 100)
            self.assertAlmostEqual(core.epsilon, 1.0 + (0.05 - 1.0) * frac, places=12)

    def test_greedy_picks_best_legal_q_and_counts_steps(self):
        core = self.make(eps_start=0.0, eps_end=0.0)
        # legal rows [1, 4] and [7, 9, 2]; padded entries (the 0.0 / 99.0 tails) must be ignored
        q = np.array([[3.0, 1.0, 99.0], [2.0, 0.5, 5.0]], np.float32)
        acts = core.act([0, 1], np.zeros((2, S), np.float32), q, pad([[1, 4], [7, 9, 2]]))
        self.assertEqual(list(acts), [1, 2])
        self.assertEqual(core.act_steps, 2)

    def test_full_exploration_stays_legal_and_uniform(self):
        core = self.make(eps_start=1.0, eps_end=1.0)
        counts = {2: 0, 6: 0, 11: 0}
        for i in range(3000):
            a = int(core.act([i], np.zeros((1, S), np.float32), np.zeros((1, 3), np.float32), pad([[2, 6, 11]]))[0])
            counts[a] += 1
        self.assertTrue(all(abs(c / 3000 - 1 / 3) < 0.04 for c in counts.values()), counts)

    def test_transitions_replay_and_training_schedule(self):
        rng = np.random.default_rng(7)
        core = self.make(warmup=5, train_every=2, batch=64)
        states, _, legal = make_game(rng, 4)
        q = rng.normal(size=(4, A)).astype(np.float32)
        acts = [int(core.act([0], states[t:t + 1], aligned(q[t:t + 1], [legal[t]]), pad([legal[t]]))[0]) for t in range(4)]
        self.assertEqual(core.pending(0), 4)
        self.assertEqual(core.finish_episode(0, [0.1, 0.2, 0.3, 1.0]), 0)    # 4 < warmup: no training yet
        self.assertEqual(len(core), 4)
        for t in range(3):
            core.act([1], states[t:t + 1] + 100.0, aligned(q[t:t + 1], [legal[t]]), pad([legal[t]]))   # distinct states
        self.assertEqual(core.finish_episode(1, [0.0, 0.0, -1.0]), 1)         # len 7 >= warmup, max(1, 3//2)
        s, a, r, s2, done, m2, m1 = core.sample()
        self.assertEqual(len(s), 7)                                            # batch (64) capped at buffer size
        self.assertEqual(int(done.sum()), 2)
        for i in range(len(s)):
            self.assertTrue(m1[i, a[i]])
            if done[i]:
                self.assertFalse(m2[i].any()); self.assertFalse(s2[i].any())
            else:
                self.assertTrue(m2[i].any())
        # a non-terminal row's s2 must equal the next recorded state of the same game
        h = lambda x: x.astype(np.float16).astype(np.float32)
        for i in range(len(s)):
            if not done[i] and np.allclose(s[i], h(states[0])):
                np.testing.assert_allclose(s2[i], h(states[1]))
                self.assertAlmostEqual(float(r[i]), 0.1, places=6)

    def test_capacity_ring(self):
        core = self.make(capacity=5, warmup=1)
        for g in range(4):
            core.act([g, g], np.zeros((2, S), np.float32), np.zeros((2, 2), np.float32), pad([[1, 2], [3, 4]]))
            core.finish_episode(g, [0.0, 1.0])
        self.assertEqual(len(core), 5)

    def test_clear_episodes(self):
        core = self.make()
        core.act([3], np.zeros((1, S), np.float32), np.zeros((1, 2), np.float32), pad([[0, 1]]))
        core.clear_episodes()
        self.assertEqual(core.pending(3), 0)
        self.assertEqual(core.finish_episode(3, []), 0)


if __name__ == "__main__":
    unittest.main()
