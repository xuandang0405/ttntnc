import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_record_plot(record_path, output_path=None):
    with open(record_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    totals = data.get("totals", {})
    sides = ("white", "black")
    labels = ("White", "Black")
    outcomes = ("wins", "losses", "draws")
    colors = ("#4c9f70", "#c65353", "#d8a43b")
    x = range(len(sides))
    width = 0.23

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (outcome, color) in enumerate(zip(outcomes, colors)):
        values = [totals.get(side, {}).get(outcome, 0) for side in sides]
        ax.bar([position + (i - 1) * width for position in x], values,
               width, label=outcome.title(), color=color)

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Games")
    ax.set_title(f"Game results — {data.get('mode', 'Unknown mode')}")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_path or os.path.splitext(record_path)[0] + "_plot.png"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print("saved", output_path)
    return output_path


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        records = sorted(glob.glob(os.path.join("checkpoints", "game_records", "game_results_*.json")))
        if not records:
            sys.exit("No game result JSON files found.")
        path = records[-1]
    save_record_plot(path)


if __name__ == "__main__":
    main()
