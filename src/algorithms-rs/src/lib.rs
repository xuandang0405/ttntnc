//! Native AlphaZero search for the chess application.
//!
//! * `alphazero` – the original arena-based tree primitives (`MctsTree`),
//!   kept for the existing Python-driven backend.
//! * `search` – chess rules, input encoding and a lock-step batched MCTS
//!   (`BatchSearch`). Python only runs the neural network.

pub mod alphazero;
pub mod search;

use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Position};

use search::encode::{CHANNELS, STATE_LEN};
use search::rules::{position_key, terminal_value};
use search::tree::{Game, Tree};

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

fn parse_position(fen: &str) -> Result<Chess, String> {
    let setup: Fen = fen.parse().map_err(|e| format!("bad FEN {fen:?}: {e}"))?;
    // python-chess accepts some setups shakmaty rejects (e.g. unusual castling
    // rights, too much material in curriculum positions); relax those checks.
    setup
        .into_position::<Chess>(CastlingMode::Standard)
        .or_else(|e| e.ignore_invalid_castling_rights())
        .or_else(|e| e.ignore_invalid_ep_square())
        .or_else(|e| e.ignore_too_much_material())
        .or_else(|e| e.ignore_impossible_check())
        .map_err(|e| format!("unsupported position {fen:?}: {e}"))
}

#[derive(PartialEq, Clone, Copy)]
enum Phase {
    Root,
    Simulating,
    Done,
}

/// Lock-step MCTS over many games.
///
/// Protocol (see `algorithms/algo_alphazero.py`):
/// ```text
/// search = BatchSearch(root_fens, move_lists, sims, c_puct, noise_frac, alpha, seed)
/// while (batch := search.next_batch()) is not None:
///     states, legal = batch            # (N,56,8,8) f32, (N,L) i32 padded with -1
///     priors, values = network(states, legal)
///     search.submit(priors, values)    # (N,L) f32 softmax over legal, (N,) f32
/// results = search.results()
/// ```
#[pyclass(name = "BatchSearch")]
struct PyBatchSearch {
    trees: Vec<Tree>,
    sims: usize,
    round: usize,
    c_puct: f64,
    noise: Option<(f64, f64)>,
    phase: Phase,
    /// Tree indices in the order of the rows of the batch last handed out.
    rows: Vec<usize>,
}

#[pymethods]
impl PyBatchSearch {
    #[new]
    #[pyo3(signature = (root_fens, move_lists, sims, c_puct, noise_frac=0.0, dirichlet_alpha=0.3, seed=0))]
    fn new(
        py: Python<'_>,
        root_fens: Vec<String>,
        move_lists: Vec<Vec<String>>,
        sims: usize,
        c_puct: f64,
        noise_frac: f64,
        dirichlet_alpha: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if root_fens.len() != move_lists.len() {
            return Err(PyValueError::new_err("root_fens and move_lists differ in length"));
        }
        let trees: Result<Vec<Tree>, String> = py.allow_threads(|| {
            root_fens
                .par_iter()
                .zip(move_lists.par_iter())
                .enumerate()
                .map(|(i, (fen, moves))| {
                    let root = parse_position(fen)?;
                    let game = Game::new(root, moves)?;
                    let tree_seed = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    Ok(Tree::new(game, tree_seed))
                })
                .collect()
        });
        Ok(Self {
            trees: trees.map_err(PyValueError::new_err)?,
            sims,
            round: 0,
            c_puct,
            noise: (noise_frac > 0.0).then_some((noise_frac, dirichlet_alpha)),
            phase: Phase::Root,
            rows: Vec::new(),
        })
    }

    /// Next batch of leaves to evaluate, or `None` when the search is finished.
    #[allow(clippy::type_complexity)]
    fn next_batch<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<i32>>)>> {
        if !self.rows.is_empty() {
            return Err(PyRuntimeError::new_err("submit() must be called before next_batch()"));
        }
        loop {
            match self.phase {
                Phase::Done => return Ok(None),
                Phase::Simulating if self.round >= self.sims => {
                    self.phase = Phase::Done;
                    return Ok(None);
                }
                _ => {}
            }
            let (phase, c_puct) = (self.phase, self.c_puct);
            let trees = &mut self.trees;
            let wants: Vec<bool> = py.allow_threads(|| {
                trees
                    .par_iter_mut()
                    .map(|t| match phase {
                        Phase::Root => t.start_root(),
                        _ => t.start_simulation(c_puct),
                    })
                    .collect()
            });
            self.rows = wants
                .iter()
                .enumerate()
                .filter_map(|(i, &w)| w.then_some(i))
                .collect();
            if self.rows.is_empty() {
                // Every tree hit a terminal node (or is finished): nothing to evaluate.
                match self.phase {
                    Phase::Root => {
                        self.phase = Phase::Simulating;
                        self.round = 0;
                    }
                    _ => self.round += 1,
                }
                continue;
            }

            let n = self.rows.len();
            let width = self
                .rows
                .iter()
                .map(|&i| self.trees[i].pending.as_ref().unwrap().legal.len())
                .max()
                .unwrap();
            let mut states = vec![0f32; n * STATE_LEN];
            let mut legal = vec![-1i32; n * width];
            for (row, &i) in self.rows.iter().enumerate() {
                let tree = &mut self.trees[i];
                tree.row = row;
                states[row * STATE_LEN..(row + 1) * STATE_LEN].copy_from_slice(&tree.enc);
                for (k, (action, _)) in tree.pending.as_ref().unwrap().legal.iter().enumerate() {
                    legal[row * width + k] = *action as i32;
                }
            }
            let states = PyArray1::from_vec(py, states).reshape([n, CHANNELS, 8, 8])?;
            let legal = PyArray1::from_vec(py, legal).reshape([n, width])?;
            return Ok(Some((states, legal)));
        }
    }

    /// Hand back the network output for the last batch.
    fn submit(
        &mut self,
        py: Python<'_>,
        priors: PyReadonlyArray2<f32>,
        values: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let n = self.rows.len();
        if n == 0 {
            return Err(PyRuntimeError::new_err("submit() called without a pending batch"));
        }
        let shape = priors.shape();
        if shape[0] != n || values.len() != n {
            return Err(PyValueError::new_err(format!(
                "expected {n} rows, got priors {shape:?} and {} values",
                values.len()
            )));
        }
        let width = shape[1];
        let priors = priors.as_slice().map_err(|_| PyValueError::new_err("priors must be C-contiguous"))?;
        let values = values.as_slice().map_err(|_| PyValueError::new_err("values must be C-contiguous"))?;
        if priors.iter().any(|p| !p.is_finite()) || values.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("network output contains non-finite values"));
        }
        let noise = if self.phase == Phase::Root { self.noise } else { None };
        let rows = std::mem::take(&mut self.rows);
        let trees = &mut self.trees;
        py.allow_threads(|| {
            trees.par_iter_mut().enumerate().for_each(|(i, t)| {
                if rows.binary_search(&i).is_ok() {
                    let row = t.row;
                    t.finish_leaf(&priors[row * width..(row + 1) * width], values[row], noise);
                }
            });
        });
        match self.phase {
            Phase::Root => {
                self.phase = Phase::Simulating;
                self.round = 0;
            }
            _ => self.round += 1,
        }
        Ok(())
    }

    /// Per game: `(action indices, visit counts, UCI moves)`; empty for terminal roots.
    fn results(&self) -> Vec<(Vec<u16>, Vec<u32>, Vec<String>)> {
        self.trees
            .iter()
            .map(|t| {
                if !t.active {
                    return (vec![], vec![], vec![]);
                }
                let stats = t.root_stats();
                (
                    stats.iter().map(|s| s.0).collect(),
                    stats.iter().map(|s| s.1).collect(),
                    stats.into_iter().map(|s| s.2).collect(),
                )
            })
            .collect()
    }

    fn node_count(&self) -> usize {
        self.trees.iter().map(|t| t.nodes.len()).sum()
    }
}

/// Debug/test helper: encode one position exactly as the search does.
/// Returns `(planes[56,8,8], legal action indices in generation order, uci moves)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn encode_position<'py>(
    py: Python<'py>,
    root_fen: &str,
    moves: Vec<String>,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Vec<u16>, Vec<String>)> {
    use shakmaty::uci::UciMove;
    let root = parse_position(root_fen).map_err(PyValueError::new_err)?;
    let game = Game::new(root, &moves).map_err(PyValueError::new_err)?;
    let mut tree = Tree::new(game, 0);
    let pos = tree.game.root.clone();
    let mut boards = tree.game.prior_boards.clone();
    boards.push(pos.board().clone());
    let refs: Vec<&shakmaty::Board> = boards.iter().collect();
    let repetitions = tree.game.keys.iter().filter(|&&k| k == position_key(&pos)).count();
    search::encode::encode_into(&mut tree.enc, &refs, &pos, repetitions);
    let color = pos.turn();
    let legal = pos.legal_moves();
    let actions = legal
        .iter()
        .map(|m| search::encode::move_to_index(m, color) as u16)
        .collect();
    let ucis = legal
        .iter()
        .map(|m| UciMove::from_move(m, CastlingMode::Standard).to_string())
        .collect();
    let planes = PyArray1::from_vec(py, tree.enc.clone()).reshape([CHANNELS, 8, 8])?;
    Ok((planes, actions, ucis))
}

/// Debug/test helper: python-chess `outcome(claim_draw=True)` equivalent.
/// `None` if the game continues, else the value for the side to move.
#[pyfunction]
fn terminal_status(root_fen: &str, moves: Vec<String>) -> PyResult<Option<f32>> {
    let root = parse_position(root_fen).map_err(PyValueError::new_err)?;
    let game = Game::new(root, &moves).map_err(PyValueError::new_err)?;
    let pos = &game.root;
    let need = pos.halfmoves() as usize + 1;
    let take = need.min(game.keys.len());
    let window = &game.keys[game.keys.len() - take..];
    let legal = pos.legal_moves();
    Ok(terminal_value(pos, &legal, window))
}

#[pymodule]
fn algorithms_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMctsTree>()?;
    module.add_class::<PyBatchSearch>()?;
    module.add_function(wrap_pyfunction!(encode_position, module)?)?;
    module.add_function(wrap_pyfunction!(terminal_status, module)?)?;
    Ok(())
}
