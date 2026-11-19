//! Network input / action-index encoding. Mirrors `src/encoding.py` exactly.

use shakmaty::uci::UciMove;
use shakmaty::{Board, CastlingSide, Chess, Color, EnPassantMode, Move, Piece, Position, Role};

pub const HISTORY_LENGTH: usize = 3;
pub const PIECE_PLANES: usize = 12;
pub const META_PLANES: usize = 8;
pub const CHANNELS: usize = HISTORY_LENGTH * PIECE_PLANES + META_PLANES;
pub const STATE_LEN: usize = CHANNELS * 64;
pub const BOARD_ACTIONS: usize = 64 * 64;
pub const PROMOTION_OFFSET: usize = BOARD_ACTIONS;
pub const ACTION_SIZE: usize = BOARD_ACTIONS * 5;

const ROLES: [Role; 6] = [
    Role::Pawn,
    Role::Knight,
    Role::Bishop,
    Role::Rook,
    Role::Queen,
    Role::King,
];

/// Square as seen by `color` (Black sees the board mirrored vertically).
#[inline]
fn view_sq(sq: usize, color: Color) -> usize {
    if color == Color::White {
        sq
    } else {
        sq ^ 56
    }
}

/// Python `move_to_index`. Castling uses the standard king-two-squares form,
/// which is what python-chess stores in `Move.from_square/to_square`.
pub fn move_to_index(m: &Move, color: Color) -> usize {
    match UciMove::from_standard(m) {
        UciMove::Normal {
            from,
            to,
            promotion,
        } => {
            let base = view_sq(from as usize, color) * 64 + view_sq(to as usize, color);
            match promotion {
                None => base,
                Some(role) => {
                    let slot = match role {
                        Role::Queen => 0,
                        Role::Rook => 1,
                        Role::Bishop => 2,
                        Role::Knight => 3,
                        _ => unreachable!("illegal promotion piece"),
                    };
                    PROMOTION_OFFSET + slot * BOARD_ACTIONS + base
                }
            }
        }
        _ => unreachable!("chess moves are always UciMove::Normal"),
    }
}

/// Fill `out` (length `STATE_LEN`) with the Python `encode(board, board.turn)` planes.
///
/// `boards` holds the most recent positions, oldest first, ending with the
/// current one (at most `HISTORY_LENGTH` are used). `repetitions` is how many
/// times the current position has occurred in the game, counting itself.
pub fn encode_into(out: &mut [f32], boards: &[&Board], pos: &Chess, repetitions: usize) {
    debug_assert_eq!(out.len(), STATE_LEN);
    out.fill(0.0);
    let color = pos.turn();
    let boards = if boards.len() > HISTORY_LENGTH {
        &boards[boards.len() - HISTORY_LENGTH..]
    } else {
        boards
    };
    let offset = HISTORY_LENGTH - boards.len();

    for (i, board) in boards.iter().enumerate() {
        let base = (offset + i) * PIECE_PLANES;
        for piece_color in [Color::White, Color::Black] {
            let side = if piece_color == color { 0 } else { 6 };
            for (r, &role) in ROLES.iter().enumerate() {
                let plane = (base + side + r) * 64;
                for sq in board.by_piece(Piece {
                    color: piece_color,
                    role,
                }) {
                    out[plane + view_sq(sq as usize, color)] = 1.0;
                }
            }
        }
    }

    let meta = HISTORY_LENGTH * PIECE_PLANES;
    let castles = pos.castles();
    let opponent = !color;
    let flags = [
        castles.has(color, CastlingSide::KingSide),
        castles.has(color, CastlingSide::QueenSide),
        castles.has(opponent, CastlingSide::KingSide),
        castles.has(opponent, CastlingSide::QueenSide),
    ];
    for (k, &flag) in flags.iter().enumerate() {
        if flag {
            out[(meta + k) * 64..(meta + k + 1) * 64].fill(1.0);
        }
    }
    // python-chess keeps `ep_square` after every double pawn push, legal or not.
    if let Some(ep) = pos.ep_square(EnPassantMode::Always) {
        out[(meta + 4) * 64 + view_sq(ep as usize, color)] = 1.0;
    }
    let fill = |out: &mut [f32], k: usize, v: f32| out[(meta + k) * 64..(meta + k + 1) * 64].fill(v);
    fill(out, 5, pos.halfmoves().min(100) as f32 / 100.0);
    fill(out, 6, repetitions.clamp(1, 3) as f32 / 3.0);
    fill(out, 7, pos.fullmoves().get().min(100) as f32 / 100.0);
}
