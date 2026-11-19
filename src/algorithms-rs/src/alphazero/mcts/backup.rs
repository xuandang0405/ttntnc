use super::{Arena, NodeId};

/// Back up a leaf value along `path` (root to leaf), flipping perspective
/// after every ply exactly as Python `_backup` does.
pub fn backup(arena: &mut Arena, path: &[NodeId], mut value: f64) {
    for &node_id in path.iter().rev() {
        let node = arena.get_mut(node_id);
        node.visits += 1;
        node.value_sum += value;
        value = -value;
    }
}
