STRATEGY_ENABLED = True
STRATEGY_BATCH = 96
STRATEGY_STEPS = 5               # gradient steps per training call
STRATEGY_TRAIN_EVERY_GAMES = 2   # train after this many finished games
STRATEGY_BUFFER = 20000          # opponent moves remembered
STRATEGY_PHASE_WEIGHT = 0.3      # weight of the phase loss relative to the opponent-move loss
STRATEGY_EVAL_SAMPLES = 512      # newest unseen moves used to measure prediction accuracy
STRATEGY_MIN_SAMPLES = 2000      # opponent moves needed before lookahead trusts the opponent head
STRATEGY_MIN_STEPS = 50 

# game phase labels (0 opening, 1 middlegame, 2 endgame)
PHASE_ENDGAME_MATERIAL = 26      # endgame when non-pawn material of both sides adds up to <= this
PHASE_OPENING_MOVES = 5          # opening: move number <= this and most pieces still on the board

# opponent-aware lookahead (play time)
LOOKAHEAD_CANDIDATES = 5         # best moves by the policy that are re-examined
LOOKAHEAD_REPLIES = 3            # most likely opponent replies considered for each
LOOKAHEAD_TEMPERATURE = 0.3      # "Sampled" style: softmax(score / T) over candidates

# curriculum: share of games that start as each kind (relative weights)
CURRICULUM_WEIGHTS = {"opening": 0.5, "middlegame": 0.2, "endgame": 0.3}
MIDDLEGAME_PLIES = (10, 24)      # random plies played before a "middlegame" start
ENDGAME_MAX_PLIES = 100          # endgame games are cut off earlier (they should end quickly)

# skill tests (key E)
SKILL_POSITIONS = 20             # test positions per skill (AlphaZero uses fewer: it is slow)
SKILL_ENDGAME_GAMES = 4          # games per endgame type (K+Q vs K, K+R vs K)
SKILL_ENDGAME_MAX_PLIES = 60