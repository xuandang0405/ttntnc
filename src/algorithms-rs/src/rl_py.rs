use crate::rl::ops::{argmax_legal, log_softmax, sample_logp, trim_legal};
use crate::rl::replay::ReplayBuffer;
use crate::rl::rollout::{Batch, Kind, Rollout};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn err<E: ToString>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn rows(a: &PyReadonlyArray2<f32>, name: &str, n: usize, width: usize) -> PyResult<Vec<Vec<f32>>> {
    let v = a.as_array();
    if v.nrows() != n || v.ncols() != width {
        return Err(err(format!("{name}: expected shape ({n}, {width}), got {:?}", v.shape())));
    }
    Ok(v.rows().into_iter().map(|r| r.to_vec()).collect())
}

/// Per-row values aligned with the legal list: row i keeps its first `legal[i].len()` entries.
fn aligned(a: &PyReadonlyArray2<f32>, name: &str, legal: &[Vec<i32>], width: usize) -> PyResult<Vec<Vec<f32>>> {
    let r = rows(a, name, legal.len(), width)?;
    Ok(r.into_iter().zip(legal).map(|(mut v, l)| { v.truncate(l.len()); v }).collect())
}

fn legal_rows(a: &PyReadonlyArray2<i32>, n: usize, action_size: usize) -> PyResult<Vec<Vec<i32>>> {
    let v = a.as_array();
    if v.nrows() != n {
        return Err(err(format!("legal: expected {n} rows, got {}", v.nrows())));
    }
    v.rows()
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let row = trim_legal(&r.to_vec());
            if row.is_empty() {
                return Err(err(format!("row {i} has no legal actions")));
            }
            if row.iter().any(|&a| a as usize >= action_size) {
                return Err(err(format!("row {i} has an action >= action_size ({action_size})")));
            }
            Ok(row)
        })
        .collect()
}

fn flat2<'py, T: numpy::Element>(py: Python<'py>, v: Vec<T>, n: usize, w: usize) -> PyResult<Bound<'py, PyArray2<T>>> {
    PyArray1::from_vec(py, v).reshape([n, w])
}

#[pyclass]
pub struct DqnCore {
    state_dim: usize,
    action_size: usize,
    batch: usize,
    warmup: usize,
    train_every: usize,
    eps_start: f64,
    eps_end: f64,
    eps_decay_steps: f64,
    act_steps: u64,
    buf: ReplayBuffer,
    rng: StdRng,
}

#[pymethods]
impl DqnCore {
    #[new]
    #[pyo3(signature = (state_dim, action_size, capacity, batch, warmup, train_every, eps_start, eps_end, eps_decay_steps, seed))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        state_dim: usize, action_size: usize, capacity: usize, batch: usize, warmup: usize, train_every: usize,
        eps_start: f64, eps_end: f64, eps_decay_steps: f64, seed: u64,
    ) -> Self {
        Self {
            state_dim, action_size, batch, warmup, train_every: train_every.max(1), eps_start, eps_end,
            eps_decay_steps, act_steps: 0, buf: ReplayBuffer::new(capacity), rng: StdRng::seed_from_u64(seed),
        }
    }

    #[getter]
    fn epsilon(&self) -> f64 {
        let frac = (self.act_steps as f64 / self.eps_decay_steps).min(1.0);
        self.eps_start + (self.eps_end - self.eps_start) * frac
    }
    #[getter]
    fn act_steps(&self) -> u64 {
        self.act_steps
    }
    #[setter]
    fn set_act_steps(&mut self, v: u64) {
        self.act_steps = v;
    }
    fn __len__(&self) -> usize {
        self.buf.len()
    }
    fn pending(&self, game_id: i64) -> usize {
        self.buf.pending(game_id)
    }
    fn clear_episodes(&mut self) {
        self.buf.clear_episodes();
    }

    /// Epsilon-greedy over the legal actions of every row (`q` is (N,L), aligned with `legal`); records (state, action, legal) per game.
    fn act<'py>(
        &mut self, py: Python<'py>, game_ids: Vec<i64>, states: PyReadonlyArray2<f32>, q: PyReadonlyArray2<f32>,
        legal: PyReadonlyArray2<i32>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let n = game_ids.len();
        let width = legal.as_array().ncols();
        let st = rows(&states, "states", n, self.state_dim)?;
        let legal = legal_rows(&legal, n, self.action_size)?;
        let q = aligned(&q, "q", &legal, width)?;
        let eps = self.epsilon();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = if self.rng.gen::<f64>() < eps {
                legal[i][self.rng.gen_range(0..legal[i].len())]
            } else {
                argmax_legal(&q[i], &legal[i])
            };
            self.buf.record(game_ids[i], &st[i], a as i64, &legal[i]);
            out.push(a as i64);
        }
        self.act_steps += n as u64;
        Ok(PyArray1::from_vec(py, out))
    }

    /// Store the finished game's transitions. Returns how many gradient steps are now due.
    fn finish_episode(&mut self, game_id: i64, rewards: Vec<f32>) -> PyResult<usize> {
        let n = self.buf.finish(game_id, &rewards).map_err(err)?;
        if n == 0 || self.buf.len() < self.warmup {
            return Ok(0);
        }
        Ok((n / self.train_every).max(1))
    }

    /// Uniform minibatch: (s, a, r, s2, done, mask2, mask1). `s2` is zeros and `mask2` empty on terminals.
    fn sample<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (s, a) = (self.state_dim, self.action_size);
        let items = self.buf.sample(self.batch, &mut self.rng);
        let n = items.len();
        let (mut s1, mut s2) = (vec![0f32; n * s], vec![0f32; n * s]);
        let (mut m1, mut m2) = (vec![false; n * a], vec![false; n * a]);
        let (mut acts, mut rew, mut done) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for (r, t) in items.iter().enumerate() {
            for (d, v) in s1[r * s..(r + 1) * s].iter_mut().zip(t.s.iter()) {
                *d = v.to_f32();
            }
            if let Some(next) = &t.s2 {
                for (d, v) in s2[r * s..(r + 1) * s].iter_mut().zip(next.iter()) {
                    *d = v.to_f32();
                }
            }
            for &x in t.legal1.iter() {
                m1[r * a + x as usize] = true;
            }
            if let Some(l2) = &t.legal2 {
                for &x in l2.iter() {
                    m2[r * a + x as usize] = true;
                }
            }
            acts.push(t.a);
            rew.push(t.r);
            done.push(if t.done() { 1.0f32 } else { 0.0 });
        }
        PyTuple::new(py, [
            flat2(py, s1, n, s)?.into_any(),
            PyArray1::from_vec(py, acts).into_any(),
            PyArray1::from_vec(py, rew).into_any(),
            flat2(py, s2, n, s)?.into_any(),
            PyArray1::from_vec(py, done).into_any(),
            flat2(py, m2, n, a)?.into_any(),
            flat2(py, m1, n, a)?.into_any(),
        ])
    }
}

// PPO / A2C 
#[pyclass]
pub struct RolloutBatch {
    inner: Batch,
    rng: StdRng,
}

#[pymethods]
impl RolloutBatch {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// One shuffled pass split into minibatches: a list of int64 index arrays.
    fn epoch<'py>(&mut self, py: Python<'py>, minibatch: usize) -> Vec<Bound<'py, PyArray1<i64>>> {
        self.inner
            .epoch(minibatch, &mut self.rng)
            .into_iter()
            .map(|c| PyArray1::from_vec(py, c.into_iter().map(|i| i as i64).collect()))
            .collect()
    }

    /// (states, actions, old_logp, adv, raw_adv, returns, legal_mask) for the given rows.
    fn gather<'py>(&self, py: Python<'py>, idx: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyTuple>> {
        let idx: Vec<usize> = idx.as_array().iter().map(|&i| i as usize).collect();
        if idx.iter().any(|&i| i >= self.inner.len()) {
            return Err(err("index out of range"));
        }
        let (n, s, a) = (idx.len(), self.inner.state_dim, self.inner.action_size);
        let m = self.inner.gather(&idx);
        PyTuple::new(py, [
            flat2(py, m.states, n, s)?.into_any(),
            PyArray1::from_vec(py, m.actions).into_any(),
            PyArray1::from_vec(py, m.old_logp).into_any(),
            PyArray1::from_vec(py, m.adv).into_any(),
            PyArray1::from_vec(py, m.raw_adv).into_any(),
            PyArray1::from_vec(py, m.returns).into_any(),
            flat2(py, m.mask, n, a)?.into_any(),
        ])
    }
}

struct PolicyCore {
    state_dim: usize,
    action_size: usize,
    roll: Rollout,
    rng: StdRng,
}

impl PolicyCore {
    fn new(kind: Kind, state_dim: usize, action_size: usize, gamma: f64, lambda: f64, seed: u64) -> Self {
        Self { state_dim, action_size, roll: Rollout::new(kind, gamma, lambda), rng: StdRng::seed_from_u64(seed) }
    }

    fn act(
        &mut self, game_ids: &[i64], states: &PyReadonlyArray2<f32>, logits: &PyReadonlyArray2<f32>,
        values: Option<&PyReadonlyArray1<f32>>, legal: &PyReadonlyArray2<i32>,
    ) -> PyResult<(Vec<i64>, Vec<f32>)> {
        let n = game_ids.len();
        let st = rows(states, "states", n, self.state_dim)?;
        let width = legal.as_array().ncols();
        let legal = legal_rows(legal, n, self.action_size)?;
        let lg = aligned(logits, "logits", &legal, width)?;
        let vals: Vec<f32> = match values {
            Some(v) => {
                let v = v.as_array();
                if v.len() != n {
                    return Err(err(format!("values: expected {n} entries, got {}", v.len())));
                }
                v.iter().copied().collect()
            }
            None => vec![0.0; n],
        };
        let (mut acts, mut logps) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let lp = log_softmax(&lg[i]);
            let k = sample_logp(&lp, &mut self.rng);
            let a = legal[i][k];
            self.roll.record(game_ids[i], &st[i], a as i64, lp[k], vals[i], &legal[i]);
            acts.push(a as i64);
            logps.push(lp[k]);
        }
        Ok((acts, logps))
    }

    fn take(&mut self) -> Option<RolloutBatch> {
        let inner = self.roll.take(self.state_dim, self.action_size)?;
        Some(RolloutBatch { inner, rng: StdRng::seed_from_u64(self.rng.gen()) })
    }
}

#[pyclass]
pub struct PpoCore(PolicyCore);

#[pymethods]
impl PpoCore {
    #[new]
    fn new(state_dim: usize, action_size: usize, gamma: f64, gae_lambda: f64, seed: u64) -> Self {
        Self(PolicyCore::new(Kind::Ppo, state_dim, action_size, gamma, gae_lambda, seed))
    }

    /// Sample from softmax(logits) over the legal actions (logits aligned with `legal`). Returns (actions int64, logps float32).
    fn act<'py>(
        &mut self, py: Python<'py>, game_ids: Vec<i64>, states: PyReadonlyArray2<f32>, logits: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>, legal: PyReadonlyArray2<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
        let (a, l) = self.0.act(&game_ids, &states, &logits, Some(&values), &legal)?;
        Ok((PyArray1::from_vec(py, a), PyArray1::from_vec(py, l)))
    }
    fn finish_episode(&mut self, game_id: i64, rewards: Vec<f64>) -> PyResult<usize> {
        self.0.roll.finish(game_id, &rewards).map_err(err)
    }
    fn queued_episodes(&self) -> usize {
        self.0.roll.queued_episodes()
    }
    fn pending(&self, game_id: i64) -> usize {
        self.0.roll.pending(game_id)
    }
    fn clear_episodes(&mut self) {
        self.0.roll.clear_episodes();
    }
    fn take_batch(&mut self) -> Option<RolloutBatch> {
        self.0.take()
    }
}

#[pyclass]
pub struct A2cCore(PolicyCore);

#[pymethods]
impl A2cCore {
    #[new]
    fn new(state_dim: usize, action_size: usize, gamma: f64, seed: u64) -> Self {
        Self(PolicyCore::new(Kind::A2c, state_dim, action_size, gamma, 1.0, seed))
    }

    fn act<'py>(
        &mut self, py: Python<'py>, game_ids: Vec<i64>, states: PyReadonlyArray2<f32>, logits: PyReadonlyArray2<f32>,
        legal: PyReadonlyArray2<i32>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let (a, _) = self.0.act(&game_ids, &states, &logits, None, &legal)?;
        Ok(PyArray1::from_vec(py, a))
    }
    fn finish_episode(&mut self, game_id: i64, rewards: Vec<f64>) -> PyResult<usize> {
        self.0.roll.finish(game_id, &rewards).map_err(err)
    }
    fn queued_episodes(&self) -> usize {
        self.0.roll.queued_episodes()
    }
    fn pending(&self, game_id: i64) -> usize {
        self.0.roll.pending(game_id)
    }
    fn clear_episodes(&mut self) {
        self.0.roll.clear_episodes();
    }
    fn take_batch(&mut self) -> Option<RolloutBatch> {
        self.0.take()
    }
}

/// Call from `#[pymodule] fn algorithms_rs(...)`: `rl_py::register(m)?;`
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DqnCore>()?;
    m.add_class::<PpoCore>()?;
    m.add_class::<A2cCore>()?;
    m.add_class::<RolloutBatch>()?;
    // constants so Python can assert it is talking to a compatible build
    m.add("RL_CORE_VERSION", 1)?;
    Ok(())
}
