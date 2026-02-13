from typing import Dict

import torch
from torch.optim import Optimizer

class CoordinateMetaGrad(Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf: float = 1.0):
        defaults = dict(sigma=sigma, D_inf = D_inf)
        super(CoordinateMetaGrad, self).__init__(params, defaults)

        self.eta_grid = self._init_eta_grid()

    def _init_eta_grid(self):
        return [2 ** i for i in range(-15, 1)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            sigma = group['sigma']
            D_inf = group['D_inf']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                print("grad", grad)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["b_t"] = torch.zeros_like(p)
                    state["B_t"] = torch.zeros_like(p)
                    state["B_t-1"] = torch.zeros_like(p)

                    state["b_sum"] = torch.zeros_like(p)
                    state["B_sum"] = torch.zeros_like(p)

                    state["active_etas"] = []
                    state["eta_weights"] = {}
                    state["eta_experts"] = {}

                    state["epoch_start_B"] = torch.zeros_like(p)

                state["step"] += 1

                state["b_t"] = (D_inf + p.abs()) * grad.abs()
                state["B_t-1"] = state["B_t"].clone()
                state["B_t"] = torch.maximum(state["B_t"], state["b_t"])

                state["b_sum"] += state["b_t"] / (state["B_t"] + 1e-10)
                state["B_sum"] += state["b_t"] * state["B_t-1"] / (state["B_t"] + 1e-10)
                reset_mask = (state["B_t"] > state["epoch_start_B"] * state["b_sum"])

                if reset_mask.any():
                    state["epoch_start_B"] = torch.where(
                        reset_mask, 
                        state["B_t"],
                        state["epoch_start_B"]
                    )
                    state["b_sum"] = torch.where(
                        reset_mask,
                        torch.zeros_like(p),
                        state["b_sum"]
                    )
                    for eta in state["eta_weights"]:
                        state["eta_weights"][eta] = torch.where(
                            reset_mask,
                            torch.ones_like(p),
                            state["eta_weights"][eta]
                        )

                        

                eta_max_denom = torch.clamp(state["B_t"], min=1e-10)               
                eta_max = 1.0 / (2.0 * eta_max_denom)
                eta_min_denom = torch.clamp(state["B_sum"] + state["B_t"], min=1e-10)
                eta_min = 1.0 / (2 * eta_min_denom)

                global_max = eta_max.max().item()
                global_min = eta_min.min().item()
                state["active_etas"] = [eta for eta in self.eta_grid if (eta > global_min and eta < global_max)]

                for eta in state["active_etas"]:
                    if eta not in state["eta_weights"]:
                        state["eta_weights"][eta] = torch.ones_like(p)
                        state["eta_experts"][eta] = {
                            "w_hat": torch.zeros_like(p),
                            "lambda": (1 / (sigma ** 2)) * torch.ones_like(p),
                        }

                clipped_grad = (state["B_t-1"] / state["B_t"] + 1e-10)  * grad

                weighted_pred = torch.zeros_like(p)
                weighted_sum = torch.zeros_like(p)

                expert_losses = {}

                for eta in state["active_etas"]:
                    if eta not in state["eta_experts"]:
                        continue

                    expert = state["eta_experts"][eta]
                    w_eta = torch.clamp(expert["w_hat"], -D_inf, D_inf)

                    weight = state["eta_weights"][eta] * eta
                    weighted_pred.add_(weight * w_eta)
                    weighted_sum.add_(weight)

                    diff = w_eta - p
                    linear_term = eta * diff * clipped_grad
                    expert_losses[eta] = linear_term + linear_term ** 2

                
                w_controller = weighted_pred / (weighted_sum + 1e-10)

                for eta in state["active_etas"]:
                    if eta not in state["eta_experts"]:
                        continue

                    expert = state["eta_experts"][eta]
                    w_eta = torch.clamp(expert["w_hat"], -D_inf, D_inf)

                    expert["lambda"].add_(2 * (eta ** 2) * (grad ** 2))
                    w_eta.add_(-(1 + 2 * eta*(w_eta - w_controller) * grad) * eta * grad / expert["lambda"])

                
                total_weight = torch.zeros_like(p)
                for eta in state["active_etas"]:
                    if eta not in state["eta_weights"]:
                        continue
                        
                    loss_val = expert_losses[eta]
                    state["eta_weights"][eta].mul_(torch.exp(-loss_val))
                    total_weight.add_(state["eta_weights"][eta])

                for eta in state["active_etas"]:
                    if eta not in state["eta_weights"]:
                        continue

                    state["eta_weights"][eta].div_(total_weight + 1e-10)

                print(w_controller)
                p.copy_(w_controller)

        return loss
