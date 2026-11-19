/// Stable index into an [`Arena`].
pub type NodeId = usize;

/// An action-index edge and the child node reached by that action.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub action_idx: usize,
    pub child: NodeId,
}

/// MCTS node. `value_sum` stores values from this node's side-to-move view,
/// matching the Python implementation's sign-flipping backup convention.
#[derive(Clone, Debug)]
pub struct Node {
    pub prior: f64,
    pub visits: u64,
    pub value_sum: f64,
    pub children: Option<Vec<Edge>>,
}

impl Node {
    pub fn new(prior: f64) -> Self {
        Self {
            prior,
            visits: 0,
            value_sum: 0.0,
            children: None,
        }
    }

    #[inline]
    pub fn mean_value(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f64
        }
    }
}

/// Contiguous node storage avoids one heap allocation per Python-style object
/// lookup and keeps child references compact.
#[derive(Clone, Debug, Default)]
pub struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn with_root(prior: f64) -> (Self, NodeId) {
        let mut arena = Self::new();
        let root = arena.add_node(prior);
        (arena, root)
    }

    pub fn add_node(&mut self, prior: f64) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node::new(prior));
        id
    }

    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }
    pub fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id]
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
