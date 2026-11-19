use super::{Arena, NodeId};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection {
    pub action_idx: usize,
    pub child: NodeId,
    pub score: f64,
}

/// Select the child maximizing the same PUCT expression as Python `_select`:
/// -Q(child) + c_puct * prior * sqrt(parent_visits + 1)/(1 + child_visits).
/// Ties retain the first edge, matching Python dict insertion-order behavior.
pub fn select_puct(arena: &Arena, parent: NodeId, c_puct: f64) -> Option<Selection> {
    let node = arena.get(parent);
    let children = node.children.as_ref()?;
    let sqrt_n = ((node.visits + 1) as f64).sqrt();
    let mut best: Option<Selection> = None;

    for edge in children {
        let child = arena.get(edge.child);
        let q = if child.visits == 0 {
            0.0
        } else {
            -child.value_sum / child.visits as f64
        };
        let score = q + c_puct * child.prior * sqrt_n / (1.0 + child.visits as f64);
        if best.map_or(true, |current| score > current.score) {
            best = Some(Selection {
                action_idx: edge.action_idx,
                child: edge.child,
                score,
            });
        }
    }
    best
}
