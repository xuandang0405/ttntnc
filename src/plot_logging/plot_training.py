import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        logs = sorted(glob.glob(os.path.join("checkpoints", "exp_*", "training_log.json")))
        if not logs:
            sys.exit("No training_log.json found - train and save first.")
        path = logs[-1]
    data = json.load(open(path))
    df = pd.DataFrame(data["records"])
    df["minutes"] = df["time_seconds"] / 60
    sns.set_theme(style="darkgrid")

    def long(cols, names):
        d = df.melt(id_vars="minutes", value_vars=cols, var_name="series", value_name="value")
        d["series"] = d["series"].map(dict(zip(cols, names)))
        return d

    fig, ax = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    panels = [
        (ax[0, 0], long(["white_win_rate", "black_win_rate", "draw_rate"], ["White win", "Black win", "Draw"]),
         "Win / draw rate (last 100 games)"),
        (ax[0, 1], long(["white_actor_loss", "black_actor_loss"], ["White", "Black"]), "Actor loss"),
        (ax[1, 0], long(["white_critic_loss", "black_critic_loss"], ["White", "Black"]), "Critic loss"),
        (ax[1, 1], long(["white_policy_entropy", "black_policy_entropy"], ["White", "Black"]), "Policy entropy"),
        (ax[2, 0], long(["repetition_rate"], ["Repetition rate"]), "Repetition rate"),
        (ax[2, 1], long(["cpu_percent", "gpu_utilization_percent"], ["CPU %", "GPU %"]), "CPU / GPU utilization"),
    ]
    for a, d, title in panels:
        d = d.dropna(subset=["value"])
        if len(d):
            sns.lineplot(data=d, x="minutes", y="value", hue="series", ax=a)
        a.set_title(title)
        a.set_xlabel("minutes")
        a.set_ylabel("")
    fig.suptitle(data["experiment"])
    fig.tight_layout()
    out = os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=130)
    print("saved", out)
    plt.show()


if __name__ == "__main__":
    main()