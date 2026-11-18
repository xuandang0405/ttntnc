import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import ACTION_SIZE, INPUT_CHANNELS

# Respect an explicit CUDA selection from the application; otherwise auto-select.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=32, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.pi = nn.Linear(2 * 64, ACTION_SIZE)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(64, 128)
        self.v = nn.Linear(128, 1)

        self.opp_conv = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.opp = nn.Linear(2 * 64, ACTION_SIZE)
        self.phase = nn.Linear(128, 3)

    def trunk(self, x):
        return self.residual(self.stem(x))

    def forward(self, x):
        h = self.trunk(x)
        policy = self.pi(self.policy_conv(h).flatten(1))
        value_features = F.relu(self.value_fc1(self.value_conv(h).flatten(1)))
        value = torch.tanh(self.v(value_features)).squeeze(-1)
        return policy, value

    def forward_aux(self, x):
        h = self.trunk(x)
        value_features = F.relu(self.value_fc1(self.value_conv(h).flatten(1)))
        opponent_policy = self.opp(self.opp_conv(h).flatten(1))
        return opponent_policy, self.phase(value_features)
