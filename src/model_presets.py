from dataclasses import dataclass
from pathlib import Path

from algorithms.algorithms import ALGORITHMS
from config import CKPT_DIR

PRESET_DIR = Path(__file__).resolve().parent.parent / CKPT_DIR / "play_models"


@dataclass(frozen=True)
class ModelPreset:
    directory: Path
    label: str
    algorithm: str

    @property
    def campaign(self):
        return self.directory.name.startswith("campaign_")

    @property
    def title(self):
        return self.label if self.campaign else f"{self.label} · {self.directory.name}"

    def checkpoint(self, white):
        color = "white" if white else "black"
        return self.directory / f"{color}_{self.algorithm}.pt"


def find_model_presets(root=PRESET_DIR):
    if not root.is_dir():
        return []
    directories = sorted((folder for folder in root.iterdir() if folder.is_dir()), reverse=True)
    presets = []
    for folder in directories:
        for label, agent_class in ALGORITHMS.items():
            preset = ModelPreset(folder, label, agent_class.algorithm)
            if preset.checkpoint(True).is_file() or preset.checkpoint(False).is_file():
                presets.append(preset)
    return presets


def model_run_name(path):
    folder = Path(path).resolve().parent
    return folder.name if folder.parent == PRESET_DIR else Path(path).name
