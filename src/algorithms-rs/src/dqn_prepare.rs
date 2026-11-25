use crate::parse_position;
use crate::search::encode::{encode_into_for_color, move_to_index, STATE_LEN};
use crate::search::rules::position_key;
use crate::search::tree::Game;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use shakmaty::uci::UciMove;
use shakmaty::{Board, CastlingMode, Color, Position};

struct PreparedBoard {
    state: Vec<f32>,
    legal: Vec<i32>,
    ucis: Vec<String>,
}

fn prepare_one(root_fen: &str, moves: &[String], color: Color) -> Result<PreparedBoard, String> {
    let root = parse_position(root_fen)?;
    let game = Game::new(root, moves)?;
    let pos = &game.root;
    let legal = pos.legal_moves();
    if legal.is_empty() {
        return Err("position has no legal actions".to_owned());
    }

    let mut history: Vec<&Board> = game.prior_boards.iter().collect();
    history.push(pos.board());
    let repetitions = game
        .keys
        .iter()
        .filter(|&&key| key == position_key(pos))
        .count();
    let mut state = vec![0.0; STATE_LEN];
    encode_into_for_color(&mut state, &history, pos, repetitions, color);

    let mut actions = Vec::with_capacity(legal.len());
    let mut ucis = Vec::with_capacity(legal.len());
    for mv in &legal {
        actions.push(move_to_index(mv, color) as i32);
        ucis.push(UciMove::from_move(mv, CastlingMode::Standard).to_string());
    }
    Ok(PreparedBoard {
        state,
        legal: actions,
        ucis,
    })
}

/// Prepare DQN state tensors and legal actions for a batch of python-chess histories.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub(crate) fn prepare_dqn_batch<'py>(
    py: Python<'py>,
    root_fens: Vec<String>,
    move_lists: Vec<Vec<String>>,
    white: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<i32>>,
    Vec<Vec<String>>,
)> {
    if root_fens.len() != move_lists.len() {
        return Err(PyValueError::new_err(
            "root_fens and move_lists differ in length",
        ));
    }
    let color = if white { Color::White } else { Color::Black };
    let rows: Result<Vec<PreparedBoard>, String> = py.allow_threads(|| {
        root_fens
            .par_iter()
            .zip(move_lists.par_iter())
            .enumerate()
            .map(|(i, (fen, moves))| {
                prepare_one(fen, moves, color).map_err(|e| format!("board {i}: {e}"))
            })
            .collect()
    });
    let rows = rows.map_err(PyValueError::new_err)?;
    let count = rows.len();
    let width = rows.iter().map(|row| row.legal.len()).max().unwrap_or(0);
    let mut states = Vec::with_capacity(count * STATE_LEN);
    let mut legal = vec![-1; count * width];
    let mut ucis = Vec::with_capacity(count);
    for (i, row) in rows.into_iter().enumerate() {
        states.extend(row.state);
        legal[i * width..i * width + row.legal.len()].copy_from_slice(&row.legal);
        ucis.push(row.ucis);
    }
    let states = PyArray1::from_vec(py, states).reshape([count, STATE_LEN])?;
    let legal = PyArray1::from_vec(py, legal).reshape([count, width])?;
    Ok((states, legal, ucis))
}
