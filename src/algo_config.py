# shared
MAX_GRAD_NORM = 1.0
ENTROPY_COEF = 0.01        # A2C / PPO entropy bonus (raise if policies collapse to one line)

# PPO
PPO_CLIP = 0.2
PPO_EPOCHS = 4             # passes over the collected games per update
PPO_MINIBATCH = 256
GAE_LAMBDA = 0.95

# DQN (double DQN with a replay buffer and a target network)
DQN_BUFFER = 20000         # replay transitions kept
DQN_BATCH = 64
DQN_WARMUP = 1000          # transitions collected before learning starts
DQN_TARGET_SYNC = 500      # gradient steps between target-network copies
DQN_TRAIN_EVERY = 8        # one gradient step per this many new transitions
DQN_EPS_START = 1.0        # epsilon-greedy exploration, decayed linearly
DQN_EPS_END = 0.05
DQN_EPS_DECAY_STEPS = 20000
DQN_PLAY_TEMPERATURE = 0.1  # "Sampled" style at play time samples softmax(Q / T)

# AlphaZero (MCTS + policy/value network)
AZ_SIMULATIONS = 25        # MCTS simulations per move during self-play
AZ_PLAY_SIMULATIONS = 50   # ... when a human / other agent plays against the model
AZ_C_PUCT = 1.5
AZ_DIRICHLET_ALPHA = 0.3   # root exploration noise (self-play only)
AZ_NOISE_FRAC = 0.25
AZ_TEMP_PLIES = 30         # sample moves by visit count for this many plies, then pick the best
AZ_BUFFER = 20000          # training positions kept
AZ_BATCH = 128
AZ_TRAIN_STEPS = 20        # gradient steps per update
AZ_MIN_SAMPLES = 256       # positions needed before the first update
