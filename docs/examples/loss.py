import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adagrad, Adam
from tqdm import trange

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

from parameterfree import COCOB, KT

from metagrad import (
    CoordinateMetaGrad,
    FullBlockMetagrad,
    FullMetaGrad,
    SketchedBlockMetaGrad,
    SketchedMetaGrad,
)

optimizers = {
    "AdaGrad": (Adagrad, {"lr": 0.1}),
    "Adam": (Adam, {"lr": 0.1}),
    "cMetaGrad": (CoordinateMetaGrad, {"sigma": 3.0, "D_inf": 5}),
    "MetaGrad (Full)": (FullMetaGrad, {"sigma": 3.0, "D_inf": 5}),
    "MetaGrad (Block)": (FullBlockMetagrad, {"sigma": 3.0, "D_inf": 5}),
    "sMetaGrad (Full)": (
        SketchedMetaGrad,
        {"sigma": 3.0, "D_inf": 5, "sketch_size": 5},
    ),
    "sMetaGrad (Block)": (
        SketchedBlockMetaGrad,
        {"sigma": 3.0, "D_inf": 5, "sketch_size": 5},
    ),
    "KT": (KT, {}),
    "COCOB": (COCOB, {}),
}

plot_settings = {
    "AdaGrad": {"color": COLOURS[0]},
    "Adam": {"color": COLOURS[1]},
    "cMetaGrad": {"color": COLOURS[2]},
    "MetaGrad (Full)": {"color": COLOURS[3]},
    "MetaGrad (Block)": {"color": COLOURS[3], "linestyle": LINESTYLES[1]},
    "sMetaGrad (Full)": {"color": COLOURS[5]},
    "sMetaGrad (Block)": {"color": COLOURS[4], "linestyle": LINESTYLES[1]},
    "KT": {"color": COLOURS[6]},
    "COCOB": {"color": COLOURS[6], "linestyle": LINESTYLES[1]},
}

# Printing settings
OPT_LENGTH = 20
key_spacing = [" "*(OPT_LENGTH - len(key)) for key in optimizers]


def generate_linear(n_samples=1000, dim=10):
    true_param = np.random.uniform(-5, 5, size=dim)
    x_train = np.random.randn(n_samples, dim) * 3

    noise = np.random.randn(n_samples)
    targets = x_train @ true_param + noise
    x_train = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return x_train, targets


def generate_sin(n_samples=1000):
    x_train = np.random.randn(n_samples, 1) * 2

    noise = np.random.randn(n_samples, 1) * 0.2
    targets = np.sin(np.pi * x_train) + noise
    x_train = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return x_train, targets


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class LinearModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x @ self.weight


class SimpleNN(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def train_online(model, optimizer, data_stream, epochs=1):
    losses = []

    criterion = torch.nn.MSELoss()

    for epoch in trange(epochs):
        x, y = data_stream
        optimizer.zero_grad()
        # Simulate prediction and loss
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def plot_and_save(models, losses, fname_prefix="linear"):
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    for model in models:
        axs[0].plot(losses[model], label=model, **models[model])

        regret = np.cumsum(losses[model])
        axs[1].plot(regret, label=model, **models[model])
        if model == "Adam":
            max_loss = max(losses[model])

    axs[0].set_xlabel("Step (T)")
    axs[0].set_ylabel("MSE")
    if fname_prefix == "sin":
        axs[0].set_ylim((0, 2))
    else:
        axs[0].set_ylim((0, 1.5 * max_loss))
    axs[0].set_title(f"{fname_prefix} regression, multiple optimizers")
    axs[0].set_facecolor(FACECOLOUR)
    axs[0].grid(color="white")

    axs[1].set_xlabel("Step (T)")
    axs[1].set_ylabel("Cumulative Regret")
    axs[1].set_facecolor(FACECOLOUR)
    axs[1].grid(color="white")
    axs[1].legend()

    fig.savefig(f"{fname_prefix}_mse_regret_all_optimizers.pdf", bbox_inches="tight")
    
    axs[0].set_yscale("log")
    if fname_prefix == "sin":
        axs[0].set_ylim((1e-1, 2))
    else:
        axs[0].set_ylim((1e-1, 1.5 * max_loss))
    fig.savefig(f"{fname_prefix}_mse_regret_all_optimizers_log.pdf", bbox_inches="tight")


if __name__ == "__main__":
    set_seed(seed=42)

    DIM = 50
    EPOCHS = 1000
    N_SAMPLES = 10000

    linear_data = generate_linear(n_samples=N_SAMPLES, dim=DIM)
    sin_data = generate_sin(n_samples=N_SAMPLES)

    linear_losses = {}
    sin_losses = {}

    print("Running model experiments...")
    for i, opt in enumerate(optimizers):
        lin_model = LinearModel(DIM)
        optimizer = optimizers[opt][0](lin_model.parameters(), **optimizers[opt][1])

        linear_losses[opt] = train_online(
            lin_model, optimizer, linear_data, epochs=EPOCHS
        )

        nn_model = SimpleNN(dim=1)
        optimizer = optimizers[opt][0](nn_model.parameters(), **optimizers[opt][1])

        sin_losses[opt] = train_online(nn_model, optimizer, sin_data, epochs=EPOCHS)

        print(f"  {opt}:{key_spacing[i]}{linear_losses[opt][-1]:.4f}")
        print(f"  {opt}:{key_spacing[i]}{sin_losses[opt][-1]:.4f}")

    plot_and_save(models=plot_settings, losses=linear_losses)
    plot_and_save(models=plot_settings, losses=sin_losses, fname_prefix="sin")
