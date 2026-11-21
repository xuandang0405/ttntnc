
import os

# Shared
MAX_GRAD_NORM = 1.0
ENTROPY_COEF = 0.015

# PPO
PPO_CLIP = 0.2
PPO_EPOCHS = 6
PPO_MINIBATCH = 256
GAE_LAMBDA = 0.95

# DQN
DQN_BUFFER = 50000
DQN_BATCH = 128
DQN_WARMUP = 3000
DQN_TARGET_SYNC = 1000
DQN_TRAIN_EVERY = 4

DQN_EPS_START = 1.0
DQN_EPS_END = 0.05
DQN_EPS_DECAY_STEPS = 100000

DQN_PLAY_TEMPERATURE = 0.1

# AlphaZero
AZ_SIMULATIONS = 128
AZ_PLAY_SIMULATIONS = 256
AZ_C_PUCT = 1.5

AZ_DIRICHLET_ALPHA = 0.3
AZ_NOISE_FRAC = 0.25
AZ_TEMP_PLIES = 30

AZ_BUFFER = 50000
AZ_BATCH = 64
AZ_TRAIN_STEPS = 20
AZ_MIN_SAMPLES = 512

# Rust MCTS
_RUST_MCTS_SETTING = os.environ.get(
    "CHESS_AI_RUST_MCTS", "1"
).strip().lower()

if _RUST_MCTS_SETTING not in {"0", "1"}:
    raise ValueError(
        "CHESS_AI_RUST_MCTS must be set to '0' or '1'"
    )

AZ_USE_RUST_MCTS = _RUST_MCTS_SETTING == "1"

# Full native search
_RUST_SEARCH_SETTING = os.environ.get(
    "CHESS_AI_RUST_SEARCH", "1"
).strip().lower()

if _RUST_SEARCH_SETTING not in {"0", "1"}:
    raise ValueError(
        "CHESS_AI_RUST_SEARCH must be set to '0' or '1'"
    )

AZ_USE_RUST_SEARCH = _RUST_SEARCH_SETTING == "1"

# Native DQN / PPO / A2C (replay, rollouts, GAE, sampling, masks in Rust)
_RUST_RL_SETTING = os.environ.get(
    "CHESS_AI_RUST_RL", "1"
).strip().lower()

if _RUST_RL_SETTING not in {"0", "1"}:
    raise ValueError(
        "CHESS_AI_RUST_RL must be set to '0' or '1'"
    )

USE_RUST_RL = _RUST_RL_SETTING == "1"
