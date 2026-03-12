"""Adaptivity experiments

Here we show the adaptivity of MetaGrad, compared to OGD

The settings we consider are:

1. exp-concave vs strongly-convex vs general convex
2. Bernstein condition
3. Fixed convex function

"""

import random
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from tqdm import trange


from metagrad import (
    CoordinateMetaGrad,
    FullBlockMetagrad,
    FullMetaGrad,
    SketchedBlockMetaGrad,
    SketchedMetaGrad,
)

plt.rcParams.update({"lines.markersize": 10, "lines.linewidth": 2.2, "font.size": 15})


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
LINESTYLES = ["solid", "dashed", "dashdot"]
MARKERS = ["v", "s", "*", "p"]
FIG_DIR = "./figs"

class OGD(Optimizer):
    """
    Online Gradient Descent with L2-ball projection and configurable
    learning rate decay.
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        lr_schedule: str = "sqrt",
        domain_radius: float = float("inf"),
    ):
        defaults = dict(lr=lr, lr_schedule=lr_schedule, domain_radius=domain_radius)
        super(OGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_base = group["lr"]
            schedule = group["lr_schedule"]
            radius = group["domain_radius"]

            if "step_count" not in group:
                group["step_count"] = 0
            group["step_count"] += 1
            t = group["step_count"]  # 1-indexed, matching the original code's (t+1)

            if schedule == "sqrt":
                eta = lr_base / np.sqrt(t)
            elif schedule == "inv_t":
                eta = lr_base / t
            else:  # constant
                eta = lr_base

            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(-eta * p.grad)

                # project
                # NOTE: this is only correct for the linear predictions
                p_norm_sq = p.square().sum()
                if p_norm_sq > radius ** 2:
                    p.div_(p_norm_sq.sqrt()).mul_(radius)

        return loss

RADIUS = 3
optimizers = {
    "cMetaGrad": (CoordinateMetaGrad, {"sigma": 1.0, "D_inf": RADIUS}),
    "MetaGrad (Full)": (FullMetaGrad, {"sigma": 1.0, "D_inf": RADIUS}),
    "MetaGrad (Block)": (FullBlockMetagrad, {"sigma": 1.0, "D_inf": RADIUS}),
    "sMetaGrad (Full)": (
        SketchedMetaGrad,
        {"sigma": 1.0, "D_inf": RADIUS, "sketch_size": 5},
    ),
    "sMetaGrad (Block)": (
        SketchedBlockMetaGrad,
        {"sigma": 1.0, "D_inf": RADIUS, "sketch_size": 5},
    ),
    "OGD(1 / √t)": (
        OGD, 
        {"lr": 1.0, "lr_schedule": "sqrt", "domain_radius": RADIUS}
    ),
    "OGD(1 / t)": (
        OGD, 
        {"lr": 1.0, "lr_schedule": "inv_t", "domain_radius": RADIUS}
    ),
    "OGD(const)": (
        OGD, 
        {"lr": 0.1, "lr_schedule": "const", "domain_radius": RADIUS}
    )
}

# Printing settings
OPT_LENGTH = 20
key_spacing = [" " * (OPT_LENGTH - len(key)) for key in optimizers]

plot_settings = {
    "cMetaGrad": {"color": COLOURS[2]},
    "MetaGrad (Full)": {"color": COLOURS[3]},
    "MetaGrad (Block)": {"color": COLOURS[3], "linestyle": LINESTYLES[1]},
    "sMetaGrad (Full)": {"color": COLOURS[5]},
    "sMetaGrad (Block)": {"color": COLOURS[5], "linestyle": LINESTYLES[1]},
    "OGD(1 / √t)": {"color": COLOURS[7], "marker": MARKERS[0], "markevery": 1000},
    "OGD(1 / t)": {"color": COLOURS[7], "linestyle": LINESTYLES[2], "marker": MARKERS[1], "markevery": 1000},
    "OGD(const)": {"color": COLOURS[7], "linestyle": LINESTYLES[1], "marker": MARKERS[2], "markevery": 1000},
}

class LinearModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x @ self.weight

# The below loss functions have the desired properties, if we assume that pred = w @ x, aka linear
def strongly_convex_loss(pred, y):
    return ((pred - y) ** 2)

def exp_concave_loss(pred, y):
    return torch.log(1 + torch.exp(-y * (pred)))

def general_convex_loss(pred, y):
    return torch.clamp(1 - y * (pred), min=0)

def experiment_1_data(dim, epochs):
    w_star = torch.randn(epochs, dim)
    w_star = w_star / w_star.norm() 

    X = torch.randn(size=(epochs, dim)) * 1.5
    y = torch.einsum("ij,ij->i", X, w_star)
    y = torch.sign(y + 0.3  * torch.randn(epochs))
    return X, y, w_star


def train_online(model, optimizer, X, y, loss_func, w_star):
    losses = []
    regret = []

    epochs = X.shape[0]
    total_regret = 0
    for t in trange(epochs):
        optimizer.zero_grad()
        # Simulate prediction and loss
        pred = model(X[t])
        loss_model = loss_func(pred, y[t])
        loss_comp = loss_func(w_star[t] @ X[t], y[t])
        total_regret += loss_model.item() - loss_comp.item()

        losses.append(loss_model.item())
        regret.append(total_regret)

        loss_model.backward()
        optimizer.step()


    return losses, regret

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def plot_and_save(models, losses, fname_prefix="linear"):
    fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
    for model in models:
        axs.plot(losses[model][1], label=model, **models[model])

    axs.set_xlabel("Step (T)")
    axs.set_ylabel("Cumulative Regret")
    axs.set_title(f"{fname_prefix} adaptivity")
    if "strongly" in fname_prefix:
        axs.set_ylim((0, 25000))
    elif "exp" in fname_prefix: 
        axs.set_ylim((-2, 25))
    elif "general" in fname_prefix: 
        axs.set_ylim((-30, 100))
    axs.set_facecolor(FACECOLOUR)
    axs.grid(color="white")
    axs.legend(bbox_to_anchor=(1, 1))

    fig.savefig(f"./figs/{fname_prefix}_adaptivity.pdf", bbox_inches="tight")
    fig.savefig(f"./figs/{fname_prefix}_adaptivity.png", bbox_inches="tight")



if __name__ == "__main__":
    set_seed(123)

    DIM = 10
    EPOCHS = 5000

    CKPT_DIR = "./checkpoints/"
    USE_CKPT = True

    X, y, w_star = experiment_1_data(dim=DIM, epochs=EPOCHS)

    loss_funcs = [
        (strongly_convex_loss, {}),
        (exp_concave_loss, {}),
        (general_convex_loss, {}),
    ]

    if not USE_CKPT:
        print("Running experiment")
        for i, opt in enumerate(optimizers):
            print(f"\n\tWorking on optimizer: {opt}\n")

            lin_model = LinearModel(dim=DIM)
            optimizer = optimizers[opt][0](lin_model.parameters(), **optimizers[opt][1])

            for loss_func in loss_funcs:
                loss_func[1][opt] = train_online(
                    lin_model, optimizer, X, y, loss_func[0], w_star
                )

        strongly_convex_data = loss_funcs[0][1]
        exp_concave_data = loss_funcs[1][1]
        general_convex_data = loss_funcs[2][1]

        with open(f"{CKPT_DIR}/strongly_convex_losses.pkl", "wb") as f:
            pickle.dump(strongly_convex_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{CKPT_DIR}/exp_concave_losses.pkl", "wb") as f:
            pickle.dump(exp_concave_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{CKPT_DIR}/general_convex_losses.pkl", "wb") as f:
            pickle.dump(general_convex_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(f"{CKPT_DIR}/strongly_convex_losses.pkl", "rb") as f:
            strongly_convex_data = pickle.load(f)
        with open(f"{CKPT_DIR}/exp_concave_losses.pkl", "rb") as f:
            exp_concave_data = pickle.load(f)
        with open(f"{CKPT_DIR}/general_convex_losses.pkl", "rb") as f:
            general_convex_data = pickle.load(f)

    plot_and_save(models=plot_settings, losses=strongly_convex_data, fname_prefix="strongly_convex")
    plot_and_save(models=plot_settings, losses=exp_concave_data, fname_prefix="exp_concave")
    plot_and_save(models=plot_settings, losses=general_convex_data, fname_prefix="general_convex")
