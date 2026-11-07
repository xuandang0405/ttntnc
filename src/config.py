import chess

MAX_PLIES = 160         
GAMMA = 0.99             
LR = 3e-4                
SHAPING = 0.05           
DRAW_REWARD = -0.1    
CKPT_DIR = "checkpoints"

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

SQ = 72
BOARD_PX = SQ * 8
PANEL_W = 380
W, H = BOARD_PX + PANEL_W, BOARD_PX

LIGHT, DARK = (238, 238, 210), (118, 150, 86)
HILITE = (246, 246, 105)
BG, PANEL_BG, TEXT = (30, 30, 34), (40, 40, 46), (230, 230, 230)

AI_MOVE_DELAY = 450     
MAX_AI_PLIES = 300       