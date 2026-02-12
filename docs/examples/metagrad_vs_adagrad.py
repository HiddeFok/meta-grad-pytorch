import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Optimizer, Adagrad
from metagrad import CoordinateMetaGrad

class CustomSGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.5):
        defaults = dict(lr=lr, momentum=momentum)
        super(CustomSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                momentum_avg = state['momentum_avg']
                momentum_avg.mul_(momentum).add_(p.grad)

                p.add_(momentum_avg, alpha=-lr)

def quadratic_loss(x, target):
    return 0.5 * (x - target) ** 2

def generate_data_stream(n_samples=1000, dim=10):
    np.random.seed(42)
    targets = np.random.randn(n_samples, dim)
    return torch.tensor(targets, dtype=torch.float32)

class LinearModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x @ self.weight

def train_online(model, optimizer, data_stream, epochs=1):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for target in data_stream:
            optimizer.zero_grad()
            # Simulate prediction and loss
            x = torch.randn_like(target)  # Random input
            pred = model(x)
            loss = quadratic_loss(pred, target).sum()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(data_stream))
    return losses


def plot_and_save(losses_1, losses_2, fname_prefix="plot"):
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    axs[0].plot(losses_1, label="AdaGrad", color="blue")
    axs[0].plot(losses_2, label="Custom SGD (Momentum)", color="red")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Average Loss")
    axs[0].set_title("Online Convex Optimization: AdaGrad vs. Custom SGD")
    axs[0].legend()
    axs[0].grid()

    # --- Regret Analysis ---
    regret_1 = np.cumsum(losses_1)
    regret_2 = np.cumsum(losses_2)

    axs[1].plot(regret_1, label="AdaGrad Regret", color="blue")
    axs[1].plot(regret_2, label="Custom SGD Regret", color="red")
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
    model_adagrad = LinearModel(dim)
    model_custom = LinearModel(dim)

    optimizer_adagrad = Adagrad(model_adagrad.parameters(), lr=3e-4)
    optimizer_custom = CoordinateMetaGrad(model_custom.parameters(), sigma=1.0, D_inf=10)

    # Train
    losses_adagrad = train_online(model_adagrad, optimizer_adagrad, data_stream, epochs=100)
    losses_custom = train_online(model_custom, optimizer_custom, data_stream, epochs=100)

    plot_and_save(losses_adagrad, losses_custom, fname_prefix="plot_meta")