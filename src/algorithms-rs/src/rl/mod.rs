//! Native building blocks for the policy-gradient / value-based agents (DQN, PPO, A2C).
//!
//! Like `BatchSearch` for AlphaZero, everything that is not a network forward/backward pass
//! lives here: legal-move masking, action sampling, episode bookkeeping, replay storage,
//! return / GAE computation and minibatch assembly. The torch network stays in Python.
pub mod ops;
pub mod replay;
pub mod rollout;
