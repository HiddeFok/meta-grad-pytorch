from typing import Dict

import torch
from torch.optim import Optimizer


class CoordinateMetaGrad(Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf: float = 1.0):
        defaults = dict(sigma=sigma, D_inf=D_inf)
        super(CoordinateMetaGrad, self).__init__(params, defaults)

        self.grid_size = 15
        self.eta_grid = torch.tensor(self._init_eta_grid()).unsqueeze(0)
        print(self.eta_grid)
        print(self.eta_grid.shape)

    def _init_eta_grid(self):
        return [2**i for i in range(-self.grid_size, 0)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            sigma = group["sigma"]
            D_inf = group["D_inf"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["b_t"] = torch.zeros_like(p)
                    state["B_t"] = torch.zeros_like(p)
                    state["B_t-1"] = torch.zeros_like(p)

                    state["b_sum"] = torch.zeros_like(p)
                    state["B_sum"] = torch.zeros_like(p)

                    # This is a grid of 0s and 1s to keep track of which eta's are active for each
                    # parameter
                    state["active_etas"] = torch.ones(
                        size=(*p.shape, self.grid_size), device=p.device
                    )
                    state["eta_exp_weights"] = torch.ones(
                        size=(*p.shape, self.grid_size), device=p.device
                    )

                    state["eta_experts"] = {}
                    state["eta_experts"]["w_hat"] = torch.zeros(
                        (*p.shape, self.grid_size), device=p.device
                    )
                    state["eta_experts"]["lambda"] = torch.ones(
                        (*p.shape, self.grid_size), device=p.device
                    ) / (sigma**2)

                state["step"] += 1

                p_proj = torch.clamp(p, min=-D_inf, max=D_inf)
                state["b_t"] = (D_inf + p_proj.abs()) * grad.abs()
                state["B_t-1"] = state["B_t"].clone()
                state["B_t"] = torch.maximum(state["B_t"], state["b_t"])

                if "epoch_start_B" not in state:
                    state["epoch_start_B"] = state["B_t"]

                state["b_sum"].add_(state["b_t"] / (state["B_t"] + 1e-10))
                state["B_sum"].add_(
                    state["b_t"] * state["B_t-1"] / (state["B_t"] + 1e-10)
                )
                reset_mask = state["B_t"] > state["epoch_start_B"] * state["b_sum"]
                # if reset_mask.any():
                #     state["epoch_start_B"] = torch.where(
                #         reset_mask,
                #         state["B_t"],
                #         state["epoch_start_B"]
                #     )
                #     state["b_sum"] = torch.where(
                #         reset_mask,
                #         torch.zeros_like(p),
                #         state["b_sum"]
                #     )
                #     for eta in state["eta_weights"]:
                #         state["eta_weights"][eta] = torch.where(
                #             reset_mask,
                #             torch.ones_like(p),
                #             state["eta_weights"][eta]
                #         )

                eta_max_denom = torch.clamp(state["B_t"], min=1e-10)
                eta_max = 1.0 / (2.0 * eta_max_denom)
                eta_max = eta_max.unsqueeze(-1)
                eta_min_denom = torch.clamp(state["B_sum"] + state["B_t"], min=1e-10)
                eta_min = 1.0 / (2 * eta_min_denom)
                eta_min = eta_min.unsqueeze(-1)
                print(self.eta_grid.shape)
                print(eta_min.shape)
                state["active_etas"] = torch.logical_and(
                    self.eta_grid > eta_min, self.eta_grid < eta_max
                )

                any_active = (state["active_etas"].int().sum(axis=-1) > 0)

                w_eta = torch.clamp(state["eta_experts"]["w_hat"], -D_inf, D_inf)
                weight = torch.where(
                    state["active_etas"], state["eta_exp_weights"] * self.eta_grid, 0
                )

                weighted_pred = (weight * w_eta).sum(axis=-1)
                weight_sum = weight.sum(axis=-1)

                w_controller = torch.where(
                    any_active, 
                    weighted_pred / (weight_sum + 1e-10),
                    1
                )

                clipped_grad = (state["B_t-1"] / state["B_t"] + 1e-10) * grad

                diff = w_eta - p.unsqueeze(1)
                linear_term = self.eta_grid * diff * clipped_grad.unsqueeze(1)
                expert_losses = linear_term + linear_term**2

                state["eta_experts"]["lambda"].add_(
                    2 * (grad * grad).unsqueeze(1) * (self.eta_grid * self.eta_grid)
                )
                state["eta_experts"]["w_hat"].add_(
                    -(
                        1
                        + 2
                        * self.eta_grid
                        * (state["eta_experts"]["w_hat"] - w_controller.unsqueeze(1))
                        * grad.unsqueeze(1)
                    )
                    * self.eta_grid
                    * grad.unsqueeze(1)
                    / state["eta_experts"]["lambda"]
                )

                state["eta_exp_weights"].mul_(
                    torch.where(
                        state["active_etas"], 
                        torch.exp(-expert_losses), 
                        1
                    )
                )
                state["eta_exp_weights"].div_(
                    torch.where(
                        any_active.unsqueeze(1),
                        state["eta_exp_weights"].sum(axis=-1).unsqueeze(1) + 1e-10,
                        1
                    )
                )
                p.copy_(w_controller)

        return loss
