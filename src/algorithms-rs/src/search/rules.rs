//! Terminal-position detection equal to python-chess `Board.outcome(claim_draw=True)`.

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Bitboard, Board, Chess, Color, EnPassantMode, Move, Position};

/// Position hash used for repetition counting (en-passant only when legal,
/// like python-chess `_transposition_key`).
pub fn position_key(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// python-chess `has_insufficient_material(color)`, reproduced exactly.
fn has_insufficient_material(board: &Board, color: Color) -> bool {
    let own = board.by_color(color);
    let opp = board.by_color(!color);
    if (own & (board.pawns() | board.rooks() | board.queens())).any() {
        return false;
    }
    if (own & board.knights()).any() {
        return own.count() <= 2 && (opp & !board.kings() & !board.queens()).is_empty();
    }
    if (own & board.bishops()).any() {
        let bishops = board.bishops();
        let same_color = (bishops & Bitboard::DARK_SQUARES).is_empty()
            || (bishops & Bitboard::LIGHT_SQUARES).is_empty();
        return same_color && board.pawns().is_empty() && board.knights().is_empty();
    }
    true
}

fn is_insufficient_material(board: &Board) -> bool {
    has_insufficient_material(board, Color::White) && has_insufficient_material(board, Color::Black)
}

/// `window` = keys of the positions since (and including) the last zeroing
/// move, ending with the current position. Repetitions cannot cross a
/// zeroing move, so counting inside the window equals counting in the game.
///
/// Returns the value for the side to move: `-1` checkmated, `0` any draw,
/// `None` if the game continues.
pub fn terminal_value(pos: &Chess, legal: &[Move], window: &[u64]) -> Option<f32> {
    if legal.is_empty() {
        return Some(if pos.is_check() { -1.0 } else { 0.0 });
    }
    if is_insufficient_material(pos.board()) {
        return Some(0.0);
    }
    let halfmoves = pos.halfmoves();
    if halfmoves >= 150 {
        return Some(0.0);
    }
    let current = *window.last().expect("window includes the current position");
    let occurrences = window.iter().filter(|&&k| k == current).count();
    if occurrences >= 5 {
        return Some(0.0);
    }
    // Claimable draws (claim_draw=True).
    if halfmoves >= 100 {
        return Some(0.0);
    }
    if halfmoves >= 99 {
        for m in legal.iter().filter(|m| !m.is_zeroing()) {
            let mut next = pos.clone();
            next.play_unchecked(m);
            if next.legal_moves().len() > 0 {
                return Some(0.0);
            }
        }
    }
    if occurrences >= 3 {
        return Some(0.0);
    }
    // A legal move that makes some earlier position occur for the third time.
    let twice: Vec<u64> = window
        .iter()
        .copied()
        .filter(|k| window.iter().filter(|&&o| o == *k).count() >= 2)
        .collect();
    if !twice.is_empty() {
        for m in legal.iter().filter(|m| !m.is_zeroing()) {
            let mut next = pos.clone();
            next.play_unchecked(m);
            if twice.contains(&position_key(&next)) {
                return Some(0.0);
            }
        }
    }
    None
}
