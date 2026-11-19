use super::{Arena, Edge, NodeId};
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum ExpansionError {
    LengthMismatch,
    DuplicateActionIndex(usize),
    InvalidPrior(f64),
    AlreadyExpanded,
}

/// Expand a leaf from legal action indices and their priors.
/// Pass priors after any Dirichlet-noise mixing; noise generation remains in
/// Python for now so its RNG stream and experiment behavior are unchanged.
pub fn expand(
    arena: &mut Arena,
    node_id: NodeId,
    action_indices: &[usize],
    priors: &[f64],
) -> Result<(), ExpansionError> {
    if action_indices.len() != priors.len() {
        return Err(ExpansionError::LengthMismatch);
    }
    if arena.get(node_id).children.is_some() {
        return Err(ExpansionError::AlreadyExpanded);
    }

    let mut seen = HashSet::with_capacity(action_indices.len());
    for (&idx, &prior) in action_indices.iter().zip(priors) {
        if !seen.insert(idx) {
            return Err(ExpansionError::DuplicateActionIndex(idx));
        }
        if !prior.is_finite() || prior < 0.0 {
            return Err(ExpansionError::InvalidPrior(prior));
        }
    }

    let mut edges = Vec::with_capacity(action_indices.len());
    for (&action_idx, &prior) in action_indices.iter().zip(priors) {
        let child = arena.add_node(prior);
        edges.push(Edge { action_idx, child });
    }
    arena.get_mut(node_id).children = Some(edges);
    Ok(())
}
