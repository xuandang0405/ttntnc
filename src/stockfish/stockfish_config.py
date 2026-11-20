import os

# Path to the Stockfish executable. Otherwise STOCKFISH_PATH, then `stockfish` on PATH.
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "")
STOCKFISH_HASH_MB = 16

# coach: Stockfish evaluation as the reward-shaping potential 
_coach = os.environ.get("CHESS_AI_STOCKFISH_COACH", "0").strip().lower()
if _coach not in {"0", "1"}:
    raise ValueError("CHESS_AI_STOCKFISH_COACH must be set to '0' or '1'")
COACH_ENABLED = _coach == "1"
COACH_NODES = int(os.environ.get("CHESS_AI_STOCKFISH_COACH_NODES", "2000"))   # search nodes per evaluated position
COACH_ENGINES = max(1, min(4, (os.cpu_count() or 2) // 2))                    # Stockfish processes
COACH_EVAL_CLIP = 10.0        # evaluation clipped to +-10 pawns (mate = +-10), same unit as material
COACH_CACHE = 200_000         # positions remembered

# teacher: labelling positions
TEACHER_NODES = 50_000        # search nodes per labelled position
TEACHER_MULTIPV = 6           # candidate moves per position (the soft policy target spans these)
TEACHER_POLICY_TEMP_CP = 60.0       # softmax temperature over centipawn scores: lower = sharper target
TEACHER_MOVE_TEMP_CP = 150.0        # temperature used to pick the next move of a generated game
TEACHER_RANDOM_MOVE_PROB = 0.1      # chance of a random move, for position diversity
TEACHER_MAX_PLIES = 200             # labelled plies per generated game

# teacher: supervised pretraining
TEACHER_EPOCHS = 30
TEACHER_BATCH = 256
TEACHER_HOLDOUT = 0.1               # share of positions kept aside to measure agreement with Stockfish
TEACHER_POLICY_W = 1.0              # policy cross-entropy (DQN: Q-value regression instead)
TEACHER_VALUE_W = 1.0               # value head (not used for DQN)
TEACHER_OPP_W = 0.3                 # opponent-model head learns what strong play looks like
TEACHER_PHASE_W = 0.1
