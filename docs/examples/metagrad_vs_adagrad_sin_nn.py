import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adagrad, Adam
import torch.nn.functional as F
from metagrad import CoordinateMetaGrad
from tqdm import trange

def quadratic_loss(x, target):
    return 0.5 * (x - target) ** 2


def generate_data_stream(n_samples=1000, dim=10):
    np.random.seed(42)
    true_param = np.random.uniform(-5, 5, size=dim)
    x_train = np.random.randn(n_samples, dim) * 3
    
    noise = np.random.randn(n_samples)
    targets = np.sin(x_train @ true_param) + noise
    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


class SimpleNN(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_online(model, optimizer, data_stream, epochs=1):
    losses = []
    all_losses = []
    for epoch in trange(epochs):
        epoch_loss = 0
        for x, target in zip(*data_stream):
            optimizer.zero_grad()
            # Simulate prediction and loss
            pred = model(x)
            loss = quadratic_loss(pred, target).sum()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_losses.append(loss.item())
        losses.append(epoch_loss / len(data_stream))

    return losses, all_losses


def plot_and_save(losses_1, losses_2, losses_3, fname_prefix="plot"):
    epoch_losses_1, all_losses_1 = losses_1
    epoch_losses_2, all_losses_2 = losses_2
    epoch_losses_3, all_losses_2 = losses_3
    x_range_1 = np.arange(len(epoch_losses_1))
    # x_range_2 = np.arange(len(all_losses_1)) / (len(all_losses_1) / len(epoch_losses_1))

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    axs[0].plot(x_range_1, epoch_losses_1, label="AdaGrad", color="blue")
    # axs[0].plot(x_range_2, all_losses_1, label="AdaGrad", color="blue")
    axs[0].plot(x_range_1, epoch_losses_2, label="MetaGrad", color="red")
    # axs[0].plot(x_range_2, all_losses_2, label="MetaGrad", color="red")
    axs[0].plot(x_range_1, epoch_losses_3, label="Adam", color="purple")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Average Loss")
    axs[0].set_title("Online Convex Optimization: AdaGrad vs. MetaGrad")
    axs[0].legend()
    axs[0].grid()

    # --- Regret Analysis ---
    regret_1 = np.cumsum(epoch_losses_1)
    regret_2 = np.cumsum(epoch_losses_2)
    regret_3 = np.cumsum(epoch_losses_3)

    axs[1].plot(regret_1, label="AdaGrad Regret", color="blue")
    axs[1].plot(regret_2, label="MetaGrad", color="red")
    axs[1].plot(regret_3, label="Adam", color="purple")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Cumulative Regret")
    axs[1].set_title("Cumulative Regret Comparison")
    axs[1].legend()
    axs[1].grid()

    fig.savefig(f"{fname_prefix}_loss_regret.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # --- Experiment ---
    dim = 10
    data_stream = generate_data_stream(dim=dim)
    # Initialize models and optimizers
    model_adagrad = SimpleNN(dim)
    model_metagrad = SimpleNN(dim)
    model_adam = SimpleNN(dim)

    optimizer_adagrad = Adagrad(model_adagrad.parameters(), lr=0.1)
    optimizer_metagrad = CoordinateMetaGrad(
        model_metagrad.parameters(), sigma=1.0, D_inf=10
    )
    optimizer_adam = Adam(
        model_adam.parameters(), lr=0.1
    )

    # Train
    print("Train AdaGrad")
    losses_adagrad = train_online(
        model_adagrad, optimizer_adagrad, data_stream, epochs=10
    )
    print("Train MetaGrad")
    losses_metagrad = train_online(model_metagrad, optimizer_metagrad, data_stream, epochs=10)
    print("Train Adam")
    losses_adam = train_online(model_adam, optimizer_adam, data_stream, epochs=10)



    plot_and_save(losses_adagrad, losses_metagrad, losses_adam, fname_prefix="plot_meta")
