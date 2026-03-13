import os
import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"lines.markersize": 3, "lines.linewidth": 2.0, "font.size": 15})

COLOURS = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#F0E442",
    "#0072B2",
    "#CC79A7",
    "#000000",
]
FACECOLOUR = "#E5E5E5"
LINESTYLES = ["solid", "dotted", "dashdot"]
MARKERS = ["v", "s", "*", "p"]
FIG_DIR = "./figs"


plot_settings = {
    "Adam": {"color": COLOURS[1]},
    "cMetaGrad": {"color": COLOURS[2]},
    "sMetaGrad": {"color": COLOURS[5]},
    "KT": {"color": COLOURS[6]},
    "COCOB": {"color": COLOURS[6], "linestyle": LINESTYLES[1]},
}


def plot_and_save(plot_settings, data, fname="linear"):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 14))
    plot_types = [["train_losses", "train_accs"], ["test_losses", "test_accs"]]
    
    for i in range(2):
        for j in range(2):
            for model in plot_settings:
                axs[i][j].plot(data[model][plot_types[i][j]], label=model, **plot_settings[model])

            axs[i][j].set_xlabel("Epochs")
            axs[i][j].set_ylabel(plot_types[i][j])
            axs[i][j].set_title(f"{plot_types[i][j]}")
            axs[i][j].set_facecolor(FACECOLOUR)
            axs[i][j].grid(color="white")

            if i == 0 and j == 1:
                axs[i][j].legend()

            if "losses" in plot_types[i][j]:
                axs[i][j].set_ylim((0.1, 0.6))

            if "accs" in plot_types[i][j]:
                axs[i][j].set_ylim((0.7, 0.95))

    fig.savefig(f"./figs/{fname}.pdf", bbox_inches="tight")
    fig.savefig(f"./figs/{fname}.png", bbox_inches="tight")



if __name__ == "__main__":
    CKPT_DIR = "./checkpoints/models"
    FIG_DIR = "./figs"

    data = {}
    for opt in plot_settings:
        data_dir = os.path.join(CKPT_DIR, f"{opt.lower()}_metrics.json")
        with open(data_dir) as f:
            data[opt] = json.load(f)
    
    plot_and_save(plot_settings, data, fname="fashion_mnist")
