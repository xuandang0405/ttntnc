import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_training_plot(log_path, output_path=None, title=None):
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.get("records", []))
    output_path = output_path or os.path.splitext(log_path)[0] + ".jpg"
    title = title or data.get("title") or data.get("experiment", "Training")

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(5, 3, figsize=(19, 21), sharex=True)
    axes = axes.flatten()

    panels = [
        (["white_win_rate", "black_win_rate", "draw_rate"], ["White win", "Black win", "Draw"], "Win / draw rate (recent window)"),
        (["white_actor_loss", "black_actor_loss"], ["White", "Black"], "Actor loss"),
        (["white_critic_loss", "black_critic_loss"], ["White", "Black"], "Critic loss"),
        (["white_total_loss", "black_total_loss"], ["White", "Black"], "Total loss"),
        (["white_policy_entropy", "black_policy_entropy"], ["White", "Black"], "Policy entropy"),
        (["white_value_explained_variance", "black_value_explained_variance"], ["White", "Black"], "Value explained variance"),
        (["mean_episode_reward", "terminal_reward_mean", "material_reward_mean", "draw_reward_mean", "repetition_penalty_mean"],
         ["Episode", "Terminal", "Material", "Draw", "Repetition"], "Reward components (recent episodes)"),
        (["repetition_rate", "threefold_repetition_rate", "reversal_move_rate", "stagnation_rate"],
         ["Repeated positions", "Threefold", "Reversal moves", "Stagnation"], "Game quality / repetition"),
        (["avg_game_length_plies"], ["Average plies"], "Average game length"),
        (["games_per_second"], ["Games/s"], "Training throughput: games/s"),
        (["plies_per_second"], ["Plies/s"], "Training throughput: plies/s"),
        (["cpu_percent", "gpu_utilization_percent"], ["CPU %", "GPU %"], "CPU / GPU utilization"),
        (["draw_by_repetition_rate", "draw_by_fifty_moves_rate", "draw_by_stalemate_rate",
          "draw_by_insufficient_material_rate", "draw_by_max_plies_rate", "draw_by_other_rate"],
         ["Repetition", "50-move", "Stalemate", "Insufficient material", "Max plies", "Other"], "Draw reasons (cumulative rate)"),
        (["white_gradient_norm", "black_gradient_norm"], ["White", "Black"], "Gradient norm"),
        (["unique_position_ratio"], ["Unique position ratio"], "Position diversity"),
    ]

    if not df.empty:
        if "time_seconds" not in df:
            df["time_seconds"] = range(len(df))
        df["minutes"] = pd.to_numeric(df["time_seconds"], errors="coerce") / 60.0
    else:
        df["minutes"] = pd.Series(dtype=float)

    for ax, (cols, labels, panel_title) in zip(axes, panels):
        available = [c for c in cols if c in df.columns]
        if available and not df.empty:
            plot_df = df[["minutes"] + available].melt(
                id_vars="minutes", value_vars=available,
                var_name="series", value_name="value"
            ).dropna(subset=["value"])
            if not plot_df.empty:
                mapping = dict(zip(cols, labels))
                plot_df["series"] = plot_df["series"].map(mapping)
                sns.lineplot(data=plot_df, x="minutes", y="value", hue="series", ax=ax)
        ax.set_title(panel_title)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("")

    fig.suptitle(title, fontsize=13, y=0.995, wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", output_path)
    return output_path


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        logs = sorted(glob.glob(os.path.join("checkpoints", "exp_*", "training_log.json")))
        if not logs:
            sys.exit("No training_log.json found - train and save first.")
        path = logs[-1]
    save_training_plot(path)


if __name__ == "__main__":
    main()
