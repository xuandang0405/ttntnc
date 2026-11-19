//! Rust components ported from the project's Python AlphaZero implementation.
//! This crate deliberately contains no chess rules or neural-network code.

pub mod alphazero;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use alphazero::mcts::{backup, expand, select_puct, Arena};

#[pyclass(name = "MctsTree")]
struct PyMctsTree {
    arena: Arena,
}

#[pymethods]
impl PyMctsTree {
    #[new]
    fn new() -> Self {
        let (arena, _) = Arena::with_root(1.0);
        Self { arena }
    }

    fn expand(
        &mut self,
        node_id: usize,
        action_indices: Vec<usize>,
        priors: Vec<f64>,
    ) -> PyResult<Vec<usize>> {
        self.check_node(node_id)?;
        expand(&mut self.arena, node_id, &action_indices, &priors).map_err(|error| {
            PyValueError::new_err(format!("cannot expand MCTS node: {error:?}"))
        })?;
        Ok(self
            .arena
            .get(node_id)
            .children
            .as_ref()
            .unwrap()
            .iter()
            .map(|edge| edge.child)
            .collect())
    }

    fn is_expanded(&self, node_id: usize) -> PyResult<bool> {
        self.check_node(node_id)?;
        Ok(self.arena.get(node_id).children.is_some())
    }

    fn select(&self, node_id: usize, c_puct: f64) -> PyResult<Option<(usize, usize)>> {
        self.check_node(node_id)?;
        Ok(select_puct(&self.arena, node_id, c_puct)
            .map(|selection| (selection.action_idx, selection.child)))
    }

    fn backup(&mut self, path: Vec<usize>, value: f64) -> PyResult<()> {
        if !value.is_finite() {
            return Err(PyValueError::new_err("backup value must be finite"));
        }
        for &node_id in &path {
            self.check_node(node_id)?;
        }
        backup(&mut self.arena, &path, value);
        Ok(())
    }

    fn root_visits(&self) -> Vec<(usize, u64)> {
        self.arena
            .get(0)
            .children
            .as_ref()
            .map_or_else(Vec::new, |edges| {
                edges
                    .iter()
                    .map(|edge| (edge.action_idx, self.arena.get(edge.child).visits))
                    .collect()
            })
    }

    fn node_count(&self) -> usize {
        self.arena.len()
    }
}

impl PyMctsTree {
    fn check_node(&self, node_id: usize) -> PyResult<()> {
        if node_id >= self.arena.len() {
            return Err(PyIndexError::new_err(format!(
                "unknown MCTS node {node_id}"
            )));
        }
        Ok(())
    }
}

#[pymodule]
fn algorithms_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMctsTree>()?;
    Ok(())
}
