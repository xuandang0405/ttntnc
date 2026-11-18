"""Mô hình nơ-ron Policy/Value phục vụ các cấp độ chơi cờ vua (MLP, CNN, ResNet)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn
from model.learning.encoding import ACTION_SIZE, CHANNELS


@dataclass
class ModelConfig:
    kind: str = 'cnn'
    width: int = 32
    hidden: int = 128
    blocks: int = 4


class ResBlock(nn.Module):
    """Khối tích chập phần dư (Residual Block) với đường truyền tắt (Skip Connection).

    Tương tự kiến trúc AlphaZero / Leela Chess Zero: Giúp tín hiệu gradient truyền sâu
    qua nhiều tầng mà không bị tiêu biến, cho phép mạng học các thế trận chiến thuật phức tạp.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width < 1 or config.hidden < 1:
            raise ValueError('Kích thước mô hình phải dương.')
        self.config = config
        if config.kind == 'mlp':
            self.trunk = nn.Sequential(
                nn.Flatten(), nn.Linear(CHANNELS * 64, config.hidden), nn.ReLU(),
                nn.Linear(config.hidden, config.hidden), nn.ReLU())
        elif config.kind == 'cnn':
            w = config.width
            self.trunk = nn.Sequential(
                nn.Conv2d(CHANNELS, w, 3, padding=1), nn.ReLU(),
                nn.Conv2d(w, w * 2, 3, padding=1), nn.ReLU(), nn.Flatten(),
                nn.Linear(w * 2 * 64, config.hidden), nn.ReLU())
        elif config.kind == 'resnet':
            w = config.width
            blocks = getattr(config, 'blocks', 4)
            res_layers = [ResBlock(w) for _ in range(blocks)]
            self.trunk = nn.Sequential(
                nn.Conv2d(CHANNELS, w, 3, padding=1), nn.ReLU(),
                *res_layers,
                nn.Conv2d(w, w * 2, 3, padding=1), nn.ReLU(), nn.Flatten(),
                nn.Linear(w * 2 * 64, config.hidden), nn.ReLU())
        else:
            raise ValueError('kind phải là mlp, cnn hoặc resnet.')
        self.policy = nn.Linear(config.hidden, ACTION_SIZE)
        self.value_head = nn.Sequential(nn.Linear(config.hidden, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return self.policy(features), self.value_head(features).squeeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        # Tìm kiếm Alpha-Beta và TD(0) chỉ cần đầu Value, không cần chiếu 20.480 actions
        return self.value_head(self.trunk(x)).squeeze(-1)


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if not bool(legal_mask.any(dim=-1).all()):
        raise ValueError('Mẫu huấn luyện phải có ít nhất một nước hợp lệ.')
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def save_model(path, model, **metadata):
    from model.learning.encoding import ENCODER_VERSION
    torch.save({
        'schema': 1, 'encoder': ENCODER_VERSION,
        'config': asdict(model.config),
        'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'metadata': metadata,
    }, path)


def load_model(path, device='cpu'):
    from model.learning.encoding import ENCODER_VERSION
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != 1 or checkpoint.get('encoder') != ENCODER_VERSION:
        raise ValueError('Checkpoint không tương thích với encoder hiện tại.')
    model = PolicyValueNet(ModelConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, checkpoint['metadata']
