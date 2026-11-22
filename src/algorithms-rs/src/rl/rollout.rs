//! On-policy episode storage for PPO and A2C: per-game bookkeeping, return / GAE computation when
//! an episode finishes, and flattened batches with minibatch + legal-mask assembly.
use super::ops::{discounted_returns, gae, standardize};
use super::replay::{to_f16, Legal, State};
use half::f16;
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    A2c,
    Ppo,
}

#[derive(Default)]
struct Episode {
    states: Vec<State>,
    actions: Vec<i64>,
    logp: Vec<f32>,
    values: Vec<f32>,
    legal: Vec<Legal>,
}

struct Finished {
    ep: Episode,
    adv: Vec<f32>, // PPO only (empty for A2C)
    ret: Vec<f32>,
}

pub struct Rollout {
    pub kind: Kind,
    pub gamma: f64,
    pub lambda: f64,
    episodes: HashMap<i64, Episode>,
    done: Vec<Finished>,
}

/// Everything from the queued episodes, concatenated.
pub struct Batch {
    pub state_dim: usize,
    pub action_size: usize,
    states: Vec<f16>,
    pub actions: Vec<i64>,
    pub old_logp: Vec<f32>,
    pub raw_adv: Vec<f32>,
    pub adv: Vec<f32>, // standardised raw_adv (PPO)
    pub returns: Vec<f32>,
    legal: Vec<Legal>,
}

pub struct MiniBatch {
    pub states: Vec<f32>,
    pub actions: Vec<i64>,
    pub old_logp: Vec<f32>,
    pub adv: Vec<f32>,
    pub raw_adv: Vec<f32>,
    pub returns: Vec<f32>,
    pub mask: Vec<bool>,
}

impl Rollout {
    pub fn new(kind: Kind, gamma: f64, lambda: f64) -> Self {
        Self { kind, gamma, lambda, episodes: HashMap::new(), done: Vec::new() }
    }

    pub fn record(&mut self, game: i64, state: &[f32], action: i64, logp: f32, value: f32, legal: &[i32]) {
        let ep = self.episodes.entry(game).or_default();
        ep.states.push(to_f16(state));
        ep.actions.push(action);
        ep.logp.push(logp);
        ep.values.push(value);
        ep.legal.push(legal.to_vec().into());
    }

    pub fn pending(&self, game: i64) -> usize {
        self.episodes.get(&game).map_or(0, |e| e.actions.len())
    }

    pub fn discard(&mut self, game: i64) {
        self.episodes.remove(&game);
    }

    pub fn clear_episodes(&mut self) {
        self.episodes.clear();
    }

    pub fn queued_episodes(&self) -> usize {
        self.done.len()
    }

    pub fn queued_steps(&self) -> usize {
        self.done.iter().map(|f| f.ep.actions.len()).sum()
    }

    /// Close the episode of `game` using the shaped per-step `rewards`; returns its length.
    pub fn finish(&mut self, game: i64, rewards: &[f64]) -> Result<usize, String> {
        let Some(ep) = self.episodes.remove(&game) else { return Ok(0) };
        let n = ep.actions.len();
        if n == 0 {
            return Ok(0);
        }
        if rewards.len() != n {
            return Err(format!("expected {n} rewards for game {game}, got {}", rewards.len()));
        }
        let (adv, ret) = match self.kind {
            Kind::A2c => (Vec::new(), discounted_returns(rewards, self.gamma)),
            Kind::Ppo => {
                let v: Vec<f64> = ep.values.iter().map(|&x| x as f64).collect();
                gae(rewards, &v, self.gamma, self.lambda)
            }
        };
        self.done.push(Finished { ep, adv, ret });
        Ok(n)
    }

    /// Drain every queued episode into one flat batch. `state_dim`/`action_size` size the outputs.
    pub fn take(&mut self, state_dim: usize, action_size: usize) -> Option<Batch> {
        if self.done.is_empty() {
            return None;
        }
        let queued = std::mem::take(&mut self.done);
        let total: usize = queued.iter().map(|f| f.ep.actions.len()).sum();
        let mut b = Batch {
            state_dim,
            action_size,
            states: Vec::with_capacity(total * state_dim),
            actions: Vec::with_capacity(total),
            old_logp: Vec::with_capacity(total),
            raw_adv: Vec::with_capacity(total),
            adv: Vec::new(),
            returns: Vec::with_capacity(total),
            legal: Vec::with_capacity(total),
        };
        for f in queued {
            for s in &f.ep.states {
                debug_assert_eq!(s.len(), state_dim);
                b.states.extend_from_slice(s);
            }
            b.actions.extend(f.ep.actions);
            b.old_logp.extend(f.ep.logp);
            b.raw_adv.extend(f.adv);
            b.returns.extend(f.ret);
            b.legal.extend(f.ep.legal);
        }
        if self.kind == Kind::Ppo {
            b.adv = standardize(&b.raw_adv, 1e-8);
        }
        Some(b)
    }
}

impl Batch {
    pub fn len(&self) -> usize {
        self.actions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// A fresh shuffled pass over the batch, split into minibatches of at most `size`.
    pub fn epoch<R: Rng>(&self, size: usize, rng: &mut R) -> Vec<Vec<usize>> {
        let mut perm: Vec<usize> = (0..self.len()).collect();
        perm.shuffle(rng);
        perm.chunks(size.max(1)).map(|c| c.to_vec()).collect()
    }

    pub fn gather(&self, idx: &[usize]) -> MiniBatch {
        let (s, a) = (self.state_dim, self.action_size);
        let mut m = MiniBatch {
            states: vec![0f32; idx.len() * s],
            actions: Vec::with_capacity(idx.len()),
            old_logp: Vec::with_capacity(idx.len()),
            adv: Vec::with_capacity(idx.len()),
            raw_adv: Vec::with_capacity(idx.len()),
            returns: Vec::with_capacity(idx.len()),
            mask: vec![false; idx.len() * a],
        };
        for (row, &i) in idx.iter().enumerate() {
            let src = &self.states[i * s..(i + 1) * s];
            for (d, v) in m.states[row * s..(row + 1) * s].iter_mut().zip(src) {
                *d = v.to_f32();
            }
            for &act in self.legal[i].iter() {
                m.mask[row * a + act as usize] = true;
            }
            m.actions.push(self.actions[i]);
            m.old_logp.push(self.old_logp[i]);
            m.returns.push(self.returns[i]);
            if !self.raw_adv.is_empty() {
                m.raw_adv.push(self.raw_adv[i]);
                m.adv.push(self.adv[i]);
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn play(r: &mut Rollout, game: i64, n: usize) {
        for t in 0..n {
            r.record(game, &[game as f32, t as f32], t as i64, -0.5, 0.1 * t as f32, &[t as i32, 7]);
        }
    }

    #[test]
    fn a2c_stores_discounted_returns_across_interleaved_games() {
        let mut r = Rollout::new(Kind::A2c, 0.5, 0.95);
        play(&mut r, 1, 2);
        play(&mut r, 2, 1);
        assert_eq!(r.finish(1, &[0.0, 4.0]).unwrap(), 2);
        assert_eq!(r.finish(2, &[1.0]).unwrap(), 1);
        assert_eq!((r.queued_episodes(), r.queued_steps()), (2, 3));
        let b = r.take(2, 8).unwrap();
        assert_eq!(b.returns, vec![2.0, 4.0, 1.0]);
        assert!(b.adv.is_empty() && r.queued_episodes() == 0);
    }

    #[test]
    fn ppo_batch_has_standardised_advantages_and_masks() {
        let mut r = Rollout::new(Kind::Ppo, 0.99, 0.95);
        play(&mut r, 0, 4);
        r.finish(0, &[0.0, 0.0, 0.0, 1.0]).unwrap();
        let b = r.take(2, 8).unwrap();
        let mean: f32 = b.adv.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        let mb = b.gather(&[2, 0]);
        assert_eq!(mb.actions, vec![2, 0]);
        assert_eq!(mb.states, vec![0.0, 2.0, 0.0, 0.0]);
        let on = |row: usize| (0..8).filter(|&c| mb.mask[row * 8 + c]).collect::<Vec<_>>();
        assert_eq!((on(0), on(1)), (vec![2, 7], vec![0, 7]));
        assert_eq!(mb.raw_adv.len(), 2);
    }

    #[test]
    fn epochs_cover_every_index_once() {
        let mut r = Rollout::new(Kind::Ppo, 0.99, 0.95);
        play(&mut r, 0, 7);
        r.finish(0, &[0.0; 7]).unwrap();
        let b = r.take(2, 8).unwrap();
        let mut rng = StdRng::seed_from_u64(3);
        let chunks = b.epoch(3, &mut rng);
        assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), vec![3, 3, 1]);
        let mut all: Vec<usize> = chunks.concat();
        all.sort();
        assert_eq!(all, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn unknown_game_and_bad_reward_len() {
        let mut r = Rollout::new(Kind::A2c, 0.9, 0.95);
        assert_eq!(r.finish(42, &[]).unwrap(), 0);
        play(&mut r, 1, 2);
        assert!(r.finish(1, &[1.0]).is_err());
    }
}
