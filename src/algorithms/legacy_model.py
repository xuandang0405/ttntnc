"""Network and move encoding used by the format-2 campaign checkpoints."""

import chess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from encoding import view_sq

LEGACY_ACTION_SIZE = 64 * 64


class LegacyPolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 64, 512)
        self.pi = nn.Linear(512, LEGACY_ACTION_SIZE)
        self.v = nn.Linear(512, 1)
        self.opp = nn.Linear(512, LEGACY_ACTION_SIZE)
        self.phase = nn.Linear(512, 3)

    def trunk(self, states):
        states = F.relu(self.c1(states))
        states = F.relu(self.c2(states))
        states = F.relu(self.c3(states))
        return F.relu(self.fc(states.flatten(1)))

    def forward(self, states):
        features = self.trunk(states)
        return self.pi(features), self.v(features).tanh().squeeze(-1)

    def forward_aux(self, states):
        features = self.trunk(states)
        return self.opp(features), self.phase(features)


def legacy_encode(board, color):
    planes = np.zeros((16, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = piece.piece_type - 1 + (0 if piece.color == color else 6)
        view = view_sq(square, color)
        planes[plane, view // 8, view % 8] = 1
    planes[12].fill(board.has_kingside_castling_rights(color))
    planes[13].fill(board.has_queenside_castling_rights(color))
    planes[14].fill(board.has_kingside_castling_rights(not color))
    planes[15].fill(board.has_queenside_castling_rights(not color))
    return torch.from_numpy(planes).unsqueeze(0)


def legacy_legal_moves(board, color):
    # All promotions shared a single output in format 2. Training used the queen move.
    return {
        view_sq(move.from_square, color) * 64 + view_sq(move.to_square, color): move
        for move in board.legal_moves if move.promotion in (None, chess.QUEEN)
    }
