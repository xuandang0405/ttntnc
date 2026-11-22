//! Uniform replay ring buffer for DQN. States are stored as f16 and shared (`Arc`) between the
//! transition that produced them and the one that uses them as `s'`, so nothing is copied twice.
use half::f16;
use rand::{seq::index::sample, Rng};
use std::collections::HashMap;
use std::sync::Arc;

pub type State = Arc<[f16]>;
pub type Legal = Arc<[i32]>;

#[derive(Clone)]
pub struct Transition {
    pub s: State,
    pub a: i64,
    pub r: f32,
    pub s2: Option<State>,    // None on the terminal step
    pub legal2: Option<Legal>,
    pub legal1: Legal,
}

impl Transition {
    pub fn done(&self) -> bool {
        self.s2.is_none()
    }
}

#[derive(Default)]
struct Episode {
    states: Vec<State>,
    actions: Vec<i64>,
    legal: Vec<Legal>,
}

pub struct ReplayBuffer {
    capacity: usize,
    items: Vec<Transition>,
    pos: usize,
    episodes: HashMap<i64, Episode>,
}

pub fn to_f16(x: &[f32]) -> State {
    x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>().into()
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { capacity, items: Vec::new(), pos: 0, episodes: HashMap::new() }
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Record one decision of game `game`.
    pub fn record(&mut self, game: i64, state: &[f32], action: i64, legal: &[i32]) {
        let ep = self.episodes.entry(game).or_default();
        ep.states.push(to_f16(state));
        ep.actions.push(action);
        ep.legal.push(legal.to_vec().into());
    }

    /// Number of recorded, not yet finished decisions of `game`.
    pub fn pending(&self, game: i64) -> usize {
        self.episodes.get(&game).map_or(0, |e| e.actions.len())
    }

    pub fn discard(&mut self, game: i64) {
        self.episodes.remove(&game);
    }

    pub fn clear_episodes(&mut self) {
        self.episodes.clear();
    }

    fn push(&mut self, t: Transition) {
        if self.items.len() < self.capacity {
            self.items.push(t);
        } else if self.capacity > 0 {
            self.items[self.pos] = t;
            self.pos = (self.pos + 1) % self.capacity;
        }
    }

    /// Turn the finished episode into (s_t, a_t, r_t, s_{t+1}) transitions. Returns its length.
    pub fn finish(&mut self, game: i64, rewards: &[f32]) -> Result<usize, String> {
        let Some(ep) = self.episodes.remove(&game) else { return Ok(0) };
        let n = ep.actions.len();
        if n == 0 {
            return Ok(0);
        }
        if rewards.len() != n {
            return Err(format!("expected {n} rewards for game {game}, got {}", rewards.len()));
        }
        for t in 0..n {
            let last = t == n - 1;
            self.push(Transition {
                s: ep.states[t].clone(),
                a: ep.actions[t],
                r: rewards[t],
                s2: if last { None } else { Some(ep.states[t + 1].clone()) },
                legal2: if last { None } else { Some(ep.legal[t + 1].clone()) },
                legal1: ep.legal[t].clone(),
            });
        }
        Ok(n)
    }

    /// Uniform sample without replacement of `min(batch, len)` transitions.
    pub fn sample<R: Rng>(&self, batch: usize, rng: &mut R) -> Vec<&Transition> {
        let k = batch.min(self.items.len());
        sample(rng, self.items.len(), k).into_iter().map(|i| &self.items[i]).collect()
    }

    pub fn raw_items(&self) -> &[Transition] {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn fill(buf: &mut ReplayBuffer, game: i64, n: usize, base: f32) {
        for t in 0..n {
            buf.record(game, &[base + t as f32, 0.5], t as i64, &[t as i32, 9]);
        }
    }

    #[test]
    fn transitions_link_next_state_and_mark_terminal() {
        let mut b = ReplayBuffer::new(10);
        fill(&mut b, 3, 3, 10.0);
        assert_eq!(b.pending(3), 3);
        assert_eq!(b.finish(3, &[0.1, 0.2, 1.0]).unwrap(), 3);
        let it = b.raw_items();
        assert_eq!(it[0].s2.as_ref().unwrap()[0].to_f32(), 11.0);
        assert_eq!(it[0].legal2.as_ref().unwrap()[0], 1);
        assert!(!it[1].done() && it[2].done());
        assert_eq!(it[2].r, 1.0);
        assert_eq!(b.pending(3), 0);
    }

    #[test]
    fn ring_overwrites_oldest() {
        let mut b = ReplayBuffer::new(4);
        fill(&mut b, 0, 3, 0.0);
        b.finish(0, &[0.0; 3]).unwrap();
        fill(&mut b, 1, 3, 100.0);
        b.finish(1, &[0.0; 3]).unwrap();
        assert_eq!(b.len(), 4);
        // capacity 4: ep0 fills slots 0..3, ep1's first item takes slot 3, then 101/102 wrap to 0/1
        let firsts: Vec<f32> = b.raw_items().iter().map(|t| t.s[0].to_f32()).collect();
        assert_eq!(firsts, vec![101.0, 102.0, 2.0, 100.0]);
    }

    #[test]
    fn reward_count_mismatch_is_an_error_and_sampling_is_bounded() {
        let mut b = ReplayBuffer::new(8);
        fill(&mut b, 5, 2, 0.0);
        assert!(b.finish(5, &[1.0]).is_err());
        fill(&mut b, 6, 2, 0.0);
        b.finish(6, &[0.0, 1.0]).unwrap();
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(b.sample(128, &mut rng).len(), 2);
    }
}
