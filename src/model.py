import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")  # small net: CPU avoids transfer overhead


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(16, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 64, 512)
        self.pi = nn.Linear(512, 4096)
        self.v = nn.Linear(512, 1)
        self.opp = nn.Linear(512, 4096)    # strategy head: opponent's next move
        self.phase = nn.Linear(512, 3)     # strategy head: opening / middlegame / endgame

    def trunk(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return F.relu(self.fc(x.flatten(1)))

    def forward(self, x):
        h = self.trunk(x)
        return self.pi(h), torch.tanh(self.v(h)).squeeze(-1)

    def forward_aux(self, x):
        h = self.trunk(x)
        return self.opp(h), self.phase(h)