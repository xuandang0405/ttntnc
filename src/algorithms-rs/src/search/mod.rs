//! Self-contained AlphaZero search: chess rules, network-input encoding and a
//! lock-step batched MCTS, all in native code.
//!
//! The Python side only evaluates the neural network. One call per simulation
//! round hands Rust a batch of encoded leaves and gets priors/values back.

pub mod encode;
pub mod rules;
pub mod tree;
