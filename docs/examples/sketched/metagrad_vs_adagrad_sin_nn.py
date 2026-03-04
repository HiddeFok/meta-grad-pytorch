import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adagrad, Adam
from tqdm import trange

from metagrad import SketchedBlockMetaGrad, SketchedMetaGrad


def generate_data_stream(n_samples=1000):
    np.random.seed(42)
    x_train = np.random.randn(n_samples, 1) * 2

    noise = np.random.randn(n_samples, 1) * 0.2
    targets = np.sin(np.pi * x_train) + noise
    x_train = torch.tensor(x_train, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    return x_train, targets


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


def plot_and_save(losses_1, losses_2, losses_3, losses_4, fname_prefix="plot"):
    epoch_losses_1 = losses_1
    epoch_losses_2 = losses_2
    epoch_losses_3 = losses_3
    epoch_losses_4 = losses_4
    x_range_1 = np.arange(len(epoch_losses_1))
    # x_range_2 = np.arange(len(all_losses_1)) / (len(all_losses_1) / len(epoch_losses_1))

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    axs[0].plot(x_range_1, epoch_losses_1, label="AdaGrad", color="blue")
    # axs[0].plot(x_range_2, all_losses_1, label="AdaGrad", color="blue")
    axs[0].plot(x_range_1, epoch_losses_2, label="MetaGrad (Full)", color="red")
    axs[0].plot(
        x_range_1,
        epoch_losses_3,
        label="MetaGrad (Block)",
        color="red",
        linestyle="dotted",
    )
    # axs[0].plot(x_range_2, all_losses_2, label="MetaGrad", color="red")
    axs[0].plot(x_range_1, epoch_losses_4, label="Adam", color="purple")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Average Loss")
    axs[0].set_title("Online Convex Optimization: AdaGrad vs. MetaGrad")
    axs[0].legend()
    axs[0].grid()

    # --- Regret Analysis ---
    regret_1 = np.cumsum(epoch_losses_1)
    regret_2 = np.cumsum(epoch_losses_2)
    regret_3 = np.cumsum(epoch_losses_3)
    regret_4 = np.cumsum(epoch_losses_4)

    axs[1].plot(regret_1, label="AdaGrad Regret", color="blue")
    axs[1].plot(regret_2, label="MetaGrad (Full)", color="red")
    axs[1].plot(regret_3, label="MetaGrad (Block)", color="red", linestyle="dotted")
    axs[1].plot(regret_4, label="Adam", color="purple")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Cumulative Regret")
    axs[1].set_title("Cumulative Regret Comparison")
    axs[1].legend()
    axs[1].grid()

    fig.savefig(f"{fname_prefix}_loss_regret_sin_nn.pdf", bbox_inches="tight")


def plot_predictions(
    model_1, model_2, model_3, model_4, x_train, y_train, fname_prefix="plot_meta"
):
    x_range = torch.linspace(-5, 5, steps=100).reshape(-1, 1)
    y_1 = model_1(x_range).detach().numpy()
    y_2 = model_2(x_range).detach().numpy()
    y_3 = model_3(x_range).detach().numpy()
    y_4 = model_4(x_range).detach().numpy()

    y_true = torch.sin(torch.pi * x_range)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 5))
    x_range = x_range.detach().numpy()
    ax.plot(x_range, y_1, label="AdaGrad", color="tab:blue", linewidth=2.5)
    ax.plot(x_range, y_2, label="MetaGrad (Full)", color="tab:red", linewidth=2.5)
    ax.plot(
        x_range,
        y_3,
        label="MetaGrad (Block)",
        color="tab:red",
        linewidth=2.5,
        linestyle="dotted",
    )
    ax.plot(x_range, y_4, label="Adam", color="tab:purple", linewidth=2.5)

    ax.scatter(x_train.detach(), y_train.detach(), color="k", alpha=0.3)

    ax.plot(x_range, y_true, label="True", color="k", linewidth=2.5)

    ax.set_xlabel("Domain")
    ax.set_ylabel("Outcome")
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Predictions of the different trained models")
    ax.legend()
    ax.grid()

    fig.savefig(f"{fname_prefix}_predictions.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # --- Experiment ---
    dim = 1
    data_stream = generate_data_stream()
    # Initialize models and optimizers
    model_adagrad = SimpleNN(dim)
    model_metagrad_full = SimpleNN(dim)
    model_metagrad_block = SimpleNN(dim)
    model_adam = SimpleNN(dim)

    optimizer_adagrad = Adagrad(model_adagrad.parameters(), lr=0.01)
    optimizer_metagrad_full = SketchedMetaGrad(
        model_metagrad_full.parameters(), sigma=3, D_inf=5, sketch_size=10
    )
    optimizer_metagrad_block = SketchedBlockMetaGrad(
        model_metagrad_block.parameters(), sigma=3, D_inf=5, sketch_size=10
    )
    optimizer_adam = Adam(model_adam.parameters(), lr=0.01)
    # Train
    print("Train AdaGrad")
    losses_adagrad = train_online(
        model_adagrad, optimizer_adagrad, data_stream, epochs=500
    )
    print("Train MetaGrad")
    losses_metagrad_full = train_online(
        model_metagrad_full, optimizer_metagrad_full, data_stream, epochs=500
    )
    losses_metagrad_block = train_online(
        model_metagrad_block, optimizer_metagrad_block, data_stream, epochs=500
    )
    print("Train Adam")
    losses_adam = train_online(model_adam, optimizer_adam, data_stream, epochs=500)

    plot_and_save(
        losses_adagrad,
        losses_metagrad_full,
        losses_metagrad_full,
        losses_adam,
        fname_prefix="plot_meta",
    )
    plot_predictions(
        model_adagrad,
        model_metagrad_full,
        model_metagrad_full,
        model_adam,
        data_stream[0],
        data_stream[1],
    )
