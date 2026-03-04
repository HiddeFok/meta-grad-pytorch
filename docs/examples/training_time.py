import random
import time

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

# Printing settings
OPT_LENGTH = 20
key_spacing = [" "*(OPT_LENGTH - len(key)) for key in optimizers]

plot_settings = {
    "AdaGrad": {"color": COLOURS[0]},
    "Adam": {"color": COLOURS[1]},
    "cMetaGrad": {"color": COLOURS[2]},
    "MetaGrad (Full)": {"color": COLOURS[3]},
    "MetaGrad (Block)": {"color": COLOURS[3], "linestyle": LINESTYLES[1]},
    "sMetaGrad (Full)": {"color": COLOURS[5]},
    "sMetaGrad (Block)": {"color": COLOURS[5], "linestyle": LINESTYLES[1]},
    "KT": {"color": COLOURS[6]},
    "COCOB": {"color": COLOURS[6], "linestyle": LINESTYLES[1]},
}


def generate_linear(n_samples=1000, dim=10):
    true_param = np.random.uniform(-5, 5, size=dim)
    x_train = np.random.randn(n_samples, dim) * 3

    noise = np.random.randn(n_samples)
    targets = x_train @ true_param + noise
    x_train = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return x_train, targets


def generate_sin(n_samples=1000, dim=1):
    x_train = np.random.randn(n_samples, dim) * 2

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


class VariableDepthNN(torch.nn.Module):
    def __init__(self, input_dim, n_layers):
        super().__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, 5))
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(5, 5))
        layers.append(torch.nn.Linear(5, 1))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.tanh(layer(x))
        x = self.layers[-1](x)
        return x


def train_and_time(model, optimizer, data_stream, epochs=1):
    criterion = torch.nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        x, y = data_stream
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    return end_time - start_time


def plot_training_time(dims, times, title, fname):
    fig, ax = plt.subplots(figsize=(8, 6))
    for opt_name in times:
        ax.plot(dims, times[opt_name], label=opt_name, **plot_settings[opt_name])
    
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title(title)
    ax.set_facecolor(FACECOLOUR)
    ax.grid(color="white")
    ax.legend()
    
    fig.savefig(fname, bbox_inches="tight")


if __name__ == "__main__":
    set_seed(seed=42)

    # Parameters for linear model experiment
    linear_dims = [10, 20, 50, 100, 200, 400, 800]
    EPOCHS = 1000
    N_SAMPLES = 1000

    # Parameters for NN experiment
    nn_layers = [2, 4, 8, 16, 32]  # Increasing number of layers
    
    # Store training times
    linear_times = {opt: [] for opt in optimizers}
    nn_times = {opt: [] for opt in optimizers}

    print("Running linear model experiments...")
    for dim in linear_dims:
        print(f"Dimension: {dim}")
        linear_data = generate_linear(n_samples=N_SAMPLES, dim=dim)
        
        for i, opt in enumerate(optimizers):
            lin_model = LinearModel(dim)
            optimizer = optimizers[opt][0](lin_model.parameters(), **optimizers[opt][1])
            
            training_time = train_and_time(lin_model, optimizer, linear_data, epochs=EPOCHS)
            linear_times[opt].append(training_time)
            print(f"  {opt}:{key_spacing[i]}{training_time:.4f}s")

    print("\nRunning neural network experiments...")
    for n_layers in nn_layers:
        print(f"Layers: {n_layers}")
        nn_data = generate_sin(n_samples=N_SAMPLES, dim=1)
        
        for i, opt in enumerate(optimizers):
            nn_model = VariableDepthNN(input_dim=1, n_layers=n_layers)
            optimizer = optimizers[opt][0](nn_model.parameters(), **optimizers[opt][1])
            
            training_time = train_and_time(nn_model, optimizer, nn_data, epochs=EPOCHS)
            nn_times[opt].append(training_time)
            print(f"  {opt}:{key_spacing[i]}{training_time:.4f}s")

    # Plot results
    plot_training_time(linear_dims, linear_times, 
                      "Training Time vs Dimension (Linear Model)",
                      "linear_training_time.pdf")
    
    plot_training_time(nn_layers, nn_times,
                      "Training Time vs Network Depth (NN Model)",
                      "nn_training_time.pdf")
    
    print("Plots saved as linear_training_time.pdf and nn_training_time.pdf")