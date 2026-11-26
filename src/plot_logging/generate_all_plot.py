import re
import sys
from pathlib import Path

if __package__:
    from .plot_training import save_training_plot
else:
    from plot_training import save_training_plot


EXPERIMENT_PATTERN = re.compile(r"exp_\d{8}_\d{6}")


def generate_all_plots(checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)

    experiment_dirs = sorted(
        path for path in checkpoint_dir.glob("exp_*")
        if path.is_dir() and EXPERIMENT_PATTERN.fullmatch(path.name)
    )
    csv_paths = sorted(
        csv_path
        for experiment_dir in experiment_dirs
        for csv_path in experiment_dir.rglob("*.csv")
    )
    if not csv_paths:
        sys.exit(f"No CSV files found under {checkpoint_dir}")

    output_paths = []
    for csv_path in csv_paths:
        output_path = csv_path.with_suffix(".jpg")
        save_training_plot(str(csv_path), output_path=str(output_path))
        output_paths.append(str(output_path))
    return output_paths


if __name__ == "__main__":
    generate_all_plots()
