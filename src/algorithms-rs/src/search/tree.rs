//! One search tree per game, plus the leaf-evaluation protocol used by the
//! batched driver in `lib.rs`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};
use shakmaty::uci::UciMove;
use shakmaty::{Board, CastlingMode, Chess, Move, Position};

use super::encode::{encode_into, move_to_index, HISTORY_LENGTH, STATE_LEN};
use super::rules::{position_key, terminal_value};

const TERM_UNKNOWN: i8 = 2;

#[derive(Clone, Debug)]
pub struct Node {
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f64,
    pub first_child: u32,
    pub n_children: u32,
    pub action: u16,
    /// Cached terminal value from this node's side to move (`TERM_UNKNOWN` if not known terminal).
    pub term: i8,
}

impl Node {
    fn new(prior: f32, action: u16) -> Self {
        Node {
            prior,
            visits: 0,
            value_sum: 0.0,
            first_child: 0,
            n_children: 0,
            action,
            term: TERM_UNKNOWN,
        }
    }
}

/// Game history before the search root.
pub struct Game {
    pub root: Chess,
    /// Up to `HISTORY_LENGTH - 1` positions before the root, oldest first.
    pub prior_boards: Vec<Board>,
    /// Zobrist keys of every position from the game start up to and including the root.
    pub keys: Vec<u64>,
}

impl Game {
    pub fn new(root: Chess, ucis: &[String]) -> Result<Game, String> {
        let mut pos = root;
        let mut boards = vec![pos.board().clone()];
        let mut keys = vec![position_key(&pos)];
        for text in ucis {
            let uci = UciMove::from_ascii(text.as_bytes())
                .map_err(|e| format!("bad move {text:?}: {e}"))?;
            let m = uci
                .to_move(&pos)
                .map_err(|e| format!("illegal move {text:?}: {e}"))?;
            pos.play_unchecked(&m);
            boards.push(pos.board().clone());
            keys.push(position_key(&pos));
        }
        // `boards` holds every position; the root is the last one.
        let n = boards.len();
        let prior_boards = boards[n.saturating_sub(HISTORY_LENGTH)..n - 1].to_vec();
        // Rebuild the root as the position after all moves (`pos`).
        Ok(Game {
            root: pos,
            prior_boards,
            keys,
        })
    }
}

/// A leaf waiting for the network.
pub struct Pending {
    pub path: Vec<u32>,
    pub legal: Vec<(u16, Move)>,
}

pub struct Tree {
    pub game: Game,
    pub nodes: Vec<Node>,
    /// Move leading to each node (`None` for the root).
    pub moves: Vec<Option<Move>>,
    pub active: bool,
    pub pending: Option<Pending>,
    /// Row of this tree's pending leaf in the current batch.
    pub row: usize,
    /// Encoded state of the pending leaf.
    pub enc: Vec<f32>,
    rng: StdRng,
    path: Vec<u32>,
}

impl Tree {
    pub fn new(game: Game, seed: u64) -> Tree {
        Tree {
            game,
            nodes: vec![Node::new(1.0, 0)],
            moves: vec![None],
            active: true,
            pending: None,
            row: 0,
            enc: vec![0.0; STATE_LEN],
            rng: StdRng::seed_from_u64(seed),
            path: Vec::with_capacity(64),
        }
    }

    #[inline]
    fn select_child(&self, node: usize, c_puct: f64) -> u32 {
        let n = &self.nodes[node];
        let sqrt_n = ((n.visits + 1) as f64).sqrt();
        let first = n.first_child as usize;
        let mut best = first;
        let mut best_score = -1e9;
        for idx in first..first + n.n_children as usize {
            let c = &self.nodes[idx];
            let q = if c.visits == 0 {
                0.0
            } else {
                -c.value_sum / c.visits as f64
            };
            let score = q + c_puct * c.prior as f64 * sqrt_n / (1.0 + c.visits as f64);
            if score > best_score {
                best = idx;
                best_score = score;
            }
        }
        best as u32
    }

    fn backup(&mut self, path: &[u32], mut value: f64) {
        for &id in path.iter().rev() {
            let node = &mut self.nodes[id as usize];
            node.visits += 1;
            node.value_sum += value;
            value = -value;
        }
    }

    /// Replay `path`, test for a terminal position and, if the game goes on,
    /// encode it into `self.enc` and park it as `pending`.
    /// Returns `Ok(true)` if a network evaluation is needed.
    fn prepare_leaf(&mut self, path: Vec<u32>) -> bool {
        let mut pos = self.game.root.clone();
        let mut recent: Vec<Board> = self.game.prior_boards.clone();
        recent.push(pos.board().clone());
        let mut ext_keys: Vec<u64> = Vec::with_capacity(path.len());
        for &id in &path[1..] {
            let m = self.moves[id as usize].as_ref().expect("non-root node has a move");
            pos.play_unchecked(m);
            recent.push(pos.board().clone());
            ext_keys.push(position_key(&pos));
        }
        if recent.len() > HISTORY_LENGTH {
            recent.drain(..recent.len() - HISTORY_LENGTH);
        }

        // Keys since the last zeroing move (window of `halfmoves + 1` positions).
        let need = pos.halfmoves() as usize + 1;
        let base = &self.game.keys;
        let total = base.len() + ext_keys.len();
        let take = need.min(total);
        let mut window: Vec<u64> = Vec::with_capacity(take);
        let from_ext = take.min(ext_keys.len());
        let from_base = take - from_ext;
        window.extend_from_slice(&base[base.len() - from_base..]);
        window.extend_from_slice(&ext_keys[ext_keys.len() - from_ext..]);

        let legal = pos.legal_moves();
        let leaf = *path.last().unwrap() as usize;
        if let Some(value) = terminal_value(&pos, &legal, &window) {
            self.nodes[leaf].term = value as i8;
            if leaf == 0 {
                self.active = false;
            } else {
                self.backup(&path, value as f64);
            }
            return false;
        }

        let current = *window.last().unwrap();
        let repetitions = window.iter().filter(|&&k| k == current).count();
        let refs: Vec<&Board> = recent.iter().collect();
        encode_into(&mut self.enc, &refs, &pos, repetitions);
        let color = pos.turn();
        let legal: Vec<(u16, Move)> = legal
            .iter()
            .map(|m| (move_to_index(m, color) as u16, m.clone()))
            .collect();
        self.pending = Some(Pending { path, legal });
        true
    }

    /// Prepare the root for evaluation. Returns true if a network call is needed.
    pub fn start_root(&mut self) -> bool {
        self.prepare_leaf(vec![0])
    }

    /// Run one simulation down to a leaf. Returns true if the leaf needs the network.
    pub fn start_simulation(&mut self, c_puct: f64) -> bool {
        if !self.active || self.nodes[0].n_children == 0 {
            return false;
        }
        let mut path = std::mem::take(&mut self.path);
        path.clear();
        let mut node = 0usize;
        path.push(0);
        while self.nodes[node].n_children > 0 {
            node = self.select_child(node, c_puct) as usize;
            path.push(node as u32);
        }
        let term = self.nodes[node].term;
        if term != TERM_UNKNOWN {
            self.backup(&path, term as f64);
            self.path = path;
            return false;
        }
        let needs_eval = self.prepare_leaf(path.clone());
        self.path = path;
        needs_eval
    }

    /// Visit counts of the root's children; also yields the move text.
    pub fn root_stats(&self) -> Vec<(u16, u32, String)> {
        let root = &self.nodes[0];
        let first = root.first_child as usize;
        (first..first + root.n_children as usize)
            .map(|idx| {
                let m = self.moves[idx].as_ref().unwrap();
                let uci = UciMove::from_move(m, CastlingMode::Standard).to_string();
                (self.nodes[idx].action, self.nodes[idx].visits, uci)
            })
            .collect()
    }

    /// Feed back the network output for the pending leaf.
    /// `priors` are probabilities over `pending.legal` (same order).
    /// `noise = Some((fraction, alpha))` mixes Dirichlet noise in (root only).
    pub fn finish_leaf(&mut self, priors: &[f32], value: f32, noise: Option<(f64, f64)>) {
        let Pending { path, legal } = self.pending.take().expect("no pending leaf");
        let leaf = *path.last().unwrap() as usize;
        let n = legal.len();

        let mut p: Vec<f64> = priors[..n].iter().map(|&x| x as f64).collect();
        if let Some((frac, alpha)) = noise {
            if n > 1 {
                let gamma = Gamma::new(alpha, 1.0).expect("valid dirichlet alpha");
                let mut dir: Vec<f64> = (0..n).map(|_| gamma.sample(&mut self.rng)).collect();
                let sum: f64 = dir.iter().sum();
                if sum > 0.0 {
                    dir.iter_mut().for_each(|d| *d /= sum);
                } else {
                    dir.iter_mut().for_each(|d| *d = 1.0 / n as f64);
                }
                for (pi, di) in p.iter_mut().zip(&dir) {
                    *pi = (1.0 - frac) * *pi + frac * di;
                }
            } else {
                p[0] = (1.0 - frac) * p[0] + frac;
            }
        }

        let first = self.nodes.len();
        self.nodes[leaf].first_child = first as u32;
        self.nodes[leaf].n_children = n as u32;
        for ((action, m), prior) in legal.into_iter().zip(p) {
            self.nodes.push(Node::new(prior as f32, action));
            self.moves.push(Some(m));
        }
        if leaf != 0 {
            self.backup(&path, value as f64);
        }
    }

    pub fn reseed(&mut self) -> u64 {
        self.rng.gen()
    }
}
