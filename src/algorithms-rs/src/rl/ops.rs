//! Pure numeric helpers shared by DQN / PPO / A2C. No Python types in here so they are unit-testable.
use rand::Rng;

/// Strip the `-1` padding from one row of a padded legal-action matrix.
pub fn trim_legal(row: &[i32]) -> Vec<i32> {
    row.iter().copied().take_while(|&a| a >= 0).collect()
}

/// log-softmax of values that are already aligned with the legal-action list.
pub fn log_softmax(vals: &[f32]) -> Vec<f32> {
    let mx = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = vals.iter().map(|&v| (v - mx).exp()).sum();
    let lse = mx + sum.ln();
    vals.iter().map(|&v| v - lse).collect()
}

/// Inverse-CDF sample from log-probabilities; returns the position inside the slice.
pub fn sample_logp<R: Rng>(logp: &[f32], rng: &mut R) -> usize {
    let u: f32 = rng.gen();
    let mut acc = 0.0f32;
    for (i, &lp) in logp.iter().enumerate() {
        acc += lp.exp();
        if u < acc {
            return i;
        }
    }
    logp.len() - 1 // rounding slack
}

/// Argmax over `vals` (aligned with `legal`); returns the *action index*. Ties go to the lowest
/// action index, matching a masked dense argmax over the full action space.
pub fn argmax_legal(vals: &[f32], legal: &[i32]) -> i32 {
    let mut best = 0;
    for j in 1..legal.len() {
        if vals[j] > vals[best] || (vals[j] == vals[best] && legal[j] < legal[best]) {
            best = j;
        }
    }
    legal[best]
}

/// Discounted returns G_t = r_t + gamma * G_{t+1}.
pub fn discounted_returns(rewards: &[f64], gamma: f64) -> Vec<f32> {
    let mut out = vec![0f32; rewards.len()];
    let mut g = 0.0;
    for t in (0..rewards.len()).rev() {
        g = rewards[t] + gamma * g;
        out[t] = g as f32;
    }
    out
}

/// Generalised advantage estimation for one finished episode (value after the last step is 0).
/// Returns `(advantages, returns = advantages + values)`.
pub fn gae(rewards: &[f64], values: &[f64], gamma: f64, lambda: f64) -> (Vec<f32>, Vec<f32>) {
    let n = values.len();
    let (mut adv, mut ret) = (vec![0f32; n], vec![0f32; n]);
    let mut last = 0.0;
    for t in (0..n).rev() {
        let next_v = if t + 1 < n { values[t + 1] } else { 0.0 };
        let delta = rewards[t] + gamma * next_v - values[t];
        last = delta + gamma * lambda * last;
        adv[t] = last as f32;
        ret[t] = (last + values[t]) as f32;
    }
    (adv, ret)
}

/// (x - mean) / (std + eps) with the *unbiased* std, like `torch.Tensor.std()`. No-op for n <= 1.
pub fn standardize(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    if n <= 1 {
        return x.to_vec();
    }
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let sd = var.sqrt() as f32;
    x.iter().map(|&v| (v - mean as f32) / (sd + eps)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn log_softmax_sums_to_one_over_legal_only() {
        let lp = log_softmax(&[5.0, 2.0, 0.5]);
        let s: f32 = lp.iter().map(|v| v.exp()).sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(lp[0] > lp[1] && lp[1] > lp[2]);
    }

    #[test]
    fn sampling_follows_distribution() {
        let lp = log_softmax(&[0.0, (3.0f32).ln()]); // p = [.25, .75]
        let mut rng = StdRng::seed_from_u64(7);
        let n = 20000;
        let hits = (0..n).filter(|_| sample_logp(&lp, &mut rng) == 1).count() as f64 / n as f64;
        assert!((hits - 0.75).abs() < 0.02, "{hits}");
    }

    #[test]
    fn argmax_ties_prefer_lowest_index() {
        assert_eq!(argmax_legal(&[3.0, 3.0, 2.0], &[7, 4, 9]), 4); // tie -> lowest action index
        assert_eq!(argmax_legal(&[1.0, 2.0], &[5, 3]), 3);
        assert_eq!(argmax_legal(&[-1.0], &[12]), 12);
    }

    #[test]
    fn returns_and_gae_hand_checked() {
        let r = discounted_returns(&[1.0, 0.0, 2.0], 0.5);
        assert_eq!(r, vec![1.0 + 0.5 * (0.0 + 0.5 * 2.0), 1.0, 2.0]);
        // lambda = 0 -> one-step TD error; lambda = 1 -> return minus value baseline
        let (adv0, _) = gae(&[1.0, 2.0], &[0.5, 0.25], 0.9, 0.0);
        assert!((adv0[0] as f64 - (1.0 + 0.9 * 0.25 - 0.5)).abs() < 1e-6);
        assert!((adv0[1] as f64 - (2.0 - 0.25)).abs() < 1e-6);
        let (adv1, ret1) = gae(&[1.0, 2.0], &[0.5, 0.25], 0.9, 1.0);
        let g = discounted_returns(&[1.0, 2.0], 0.9);
        assert!((ret1[0] - g[0]).abs() < 1e-6 && (adv1[0] - (g[0] - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn standardize_uses_unbiased_std() {
        let z = standardize(&[1.0, 2.0, 3.0], 0.0);
        assert!((z[0] + 1.0).abs() < 1e-6 && z[1].abs() < 1e-6 && (z[2] - 1.0).abs() < 1e-6);
        assert_eq!(standardize(&[4.0], 1e-8), vec![4.0]);
    }
}
