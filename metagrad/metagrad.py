from typing import Dict

import torch
from torch.optim import Optimizer


class MetaGradMixin:
    """Class collecting any function that is shared amongst te different
    MetaGrad implementations
    """

    def _init_eta_grid(self):
        return [2**i for i in range(-self.grid_size, 1)]


class CoordinateMetaGrad(MetaGradMixin, Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf: float = 1.0):
        defaults = dict(sigma=sigma, D_inf=D_inf)
        super(CoordinateMetaGrad, self).__init__(params, defaults)

        self.grid_size = 20
        self.eta_grid = torch.tensor(self._init_eta_grid()).unsqueeze(0)
        self.grid_size = self.eta_grid.shape[1]

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
                    state["B_t_prev"] = torch.zeros_like(p)

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
                    state["eta_experts"]["w_hat"] = (
                        p.detach()
                        .unsqueeze(-1)
                        .expand(*p.shape, self.grid_size)
                        .clone()
                    )
                    state["eta_experts"]["lambda"] = torch.ones(
                        (*p.shape, self.grid_size), device=p.device
                    ) / (sigma**2)

                state["step"] += 1

                p_proj = torch.clamp(p, min=-D_inf, max=D_inf)
                state["b_t"] = (D_inf + p_proj.abs()) * grad.abs()
                state["B_t_prev"] = state["B_t"].clone()
                state["B_t"] = torch.maximum(state["B_t"], state["b_t"])

                if "epoch_start_B" not in state:
                    state["epoch_start_B"] = state["B_t"]

                state["b_sum"].add_(state["b_t"] / (state["B_t"] + 1e-10))
                state["B_sum"].add_(
                    state["b_t"] * state["B_t_prev"] / (state["B_t"] + 1e-10)
                )

                eta_max_denom = torch.clamp(state["B_t"], min=1e-10)
                eta_max = 1.0 / (2.0 * eta_max_denom)
                eta_max = eta_max.unsqueeze(-1)
                eta_min_denom = torch.clamp(state["B_sum"] + state["B_t"], min=1e-10)
                eta_min = 1.0 / (2 * eta_min_denom)
                eta_min = eta_min.unsqueeze(-1)

                state["active_etas"] = torch.logical_and(
                    self.eta_grid > eta_min, self.eta_grid < eta_max
                )

                reset_mask = state["B_t"] > state["epoch_start_B"] * state["b_sum"]
                if reset_mask.any():
                    state["epoch_start_B"] = torch.where(
                        reset_mask, state["B_t"], state["epoch_start_B"]
                    )
                    state["eta_exp_weights"] = torch.where(
                        torch.logical_and(
                            state["active_etas"], reset_mask.unsqueeze(-1)
                        ),
                        torch.ones(size=(*p.shape, self.grid_size), device=p.device),
                        state["eta_exp_weights"],
                    )

                any_active = state["active_etas"].int().sum(axis=-1) > 0

                w_eta = torch.clamp(state["eta_experts"]["w_hat"], -D_inf, D_inf)
                weight = torch.where(
                    state["active_etas"], state["eta_exp_weights"] * self.eta_grid, 0
                )

                weighted_pred = (weight * w_eta).sum(axis=-1)
                weight_sum = weight.sum(axis=-1)

                w_controller = torch.where(
                    any_active,
                    weighted_pred / (weight_sum + 1e-10),
                    p,
                    # 0
                    # torch.randn_like(p) / (p.numel() ** 2)
                )

                diff = w_eta - w_controller.unsqueeze(-1)
                state["eta_experts"]["lambda"].add_(
                    2 * (grad * grad).unsqueeze(-1) * (self.eta_grid * self.eta_grid)
                )
                state["eta_experts"]["w_hat"].add_(
                    -(1 + 2 * self.eta_grid * diff * grad.unsqueeze(-1))
                    * self.eta_grid
                    * grad.unsqueeze(-1)
                    / state["eta_experts"]["lambda"]
                )

                clipped_grad = (state["B_t_prev"] / state["B_t"] + 1e-10) * grad
                linear_term = self.eta_grid * diff * clipped_grad.unsqueeze(-1)
                expert_losses = linear_term + linear_term**2

                state["eta_exp_weights"].mul_(
                    torch.where(state["active_etas"], torch.exp(-expert_losses), 1)
                )
                state["eta_exp_weights"].div_(
                    torch.where(
                        any_active.unsqueeze(-1),
                        state["eta_exp_weights"].sum(axis=-1).unsqueeze(-1) + 1e-10,
                        1,
                    )
                )
                p.copy_(w_controller)

        return loss


class FullMetaGrad(MetaGradMixin, Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf: float = 1.0):
        defaults = dict(sigma=sigma, D_inf=D_inf)
        super(FullMetaGrad, self).__init__(params, defaults)

        self.grid_size = 20
        self.eta_grid = torch.tensor([2.0**i for i in range(-self.grid_size, 1)])
        self.grid_size = self.eta_grid.shape[0]

    @torch.no_grad()
    def step(self, closure=None):
        all_params = []
        all_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                all_params.append(p)
                all_grads.append(p.grad.flatten())

        g = torch.cat(all_grads)  # shape (N,)
        N = g.shape[0]
        sigma = self.param_groups[0]["sigma"]
        D_inf = self.param_groups[0]["D_inf"]

        w_flat = torch.cat([p.data.flatten() for p in all_params])

        state = self.state
        if "step" not in state:
            state["step"] = 0
            state["b_t"] = torch.tensor(0.0)
            state["B_t"] = torch.tensor(0.0)
            state["B_t_prev"] = torch.tensor(0.0)

            state["b_sum"] = torch.tensor(0.0)
            state["B_sum"] = torch.tensor(0.0)

            K = self.eta_grid.shape[0]
            state["Lambda"] = (torch.eye(N) / (sigma**2)).unsqueeze(-1).repeat(1, 1, K)
            state["Sigma"] = (torch.eye(N) * (sigma**2)).unsqueeze(-1).repeat(1, 1, K)

            state["w_hat"] = w_flat.unsqueeze(-1).repeat(1, K)  # (N, K)
            state["exp_weights"] = torch.ones(K)

        state["step"] += 1

        ## Collecting gradient information ##
        state["b_t"] = (D_inf + w_flat.norm()) * g.norm()
        state["B_t_prev"] = state["B_t"].clone()
        state["B_t"] = torch.max(state["B_t"], state["b_t"])

        if "epoch_start_B" not in state:
            state["epoch_start_B"] = state["B_t"]

        state["b_sum"].add_(state["b_t"] / (state["B_t"] + 1e-10))
        state["B_sum"].add_(state["b_t"] * state["B_t"] / (state["B_t"] + 1e-10))

        ## Updating active etas and checking reset ##
        eta_max = 1.0 / (2.0 * state["B_t"] + 1e-10)
        eta_min = 1.0 / (2.0 * (state["B_sum"] + state["B_t"]) + 1e-10)
        active = torch.logical_and(self.eta_grid > eta_min, self.eta_grid < eta_max)
        reset_mask = state["B_t"] > state["epoch_start_B"] * state["b_sum"]
        if reset_mask:
            state["epoch_start_B"] = state["B_t"]
            state["exp_weights"] = torch.where(
                active, torch.ones(K), state["exp_weights"]
            )

        ## Calculating the mean prediction of the experts ##

        # TODO More general projections?
        w_eta = torch.clamp(state["w_hat"], -D_inf, D_inf)  # (N,K)
        weights = state["exp_weights"] * self.eta_grid * active  # (K,)
        weight_sum = weights.sum()

        print(weight_sum)
        if weight_sum > 1e-10:
            w_controller = (weights.unsqueeze(0) * w_eta).sum(axis=-1) / weight_sum
        else:
            print("hey")
            w_controller = w_flat

        print(w_controller)
        ## Update experts with unclipped losses

        # TODO: Write tests to ensure that these shapes are correct and remain correct
        Sigma_g = torch.einsum("ijk,j -> ik", state["Sigma"], g).clone()  # (N, K)

        # Sigma is a symmetric matrix
        Sigma_g_g_Sigma = (
            Sigma_g[:, None, :] * Sigma_g[None, :, :]
        )  # (N, 1, K) * (1, N, K) -> (N, N, K)
        g_Sigma_g = torch.einsum("i, ik -> k", g, Sigma_g)

        state["Sigma"].add_(
            -2
            * (self.eta_grid**2)
            * Sigma_g_g_Sigma
            / (1 + 2 * (self.eta_grid**2) * g_Sigma_g)
            * active
        )

        state["Lambda"].add_(
            (2 * (self.eta_grid**2) * torch.outer(g, g).unsqueeze(-1)) * active
        )
        diff = w_eta - w_controller.unsqueeze(-1)  # (N, K) - (N, 1) -> (N, K)
        Sigma_g = torch.einsum("ijk,j -> k", state["Sigma"], g)  # (N, K)
        state["w_hat"].add_(
            -((1 + 2 * self.eta_grid * (diff.T @ g)) * self.eta_grid * Sigma_g) * active
        )

        ## Update exponential weights with clipped gradients
        clipped_grad = (state["B_t_prev"] / (state["B_t"] + 1e-10)) * g
        linear_term = self.eta_grid * (diff.T @ clipped_grad)
        expert_losses = linear_term + linear_term**2

        state["exp_weights"].mul_(torch.where(active, torch.exp(-expert_losses), 1))

        state["exp_weights"].div_(
            torch.where(
                active, state["exp_weights"].sum(axis=-1).unsqueeze(-1) + 1e-10, 1
            )
        )

        print(w_controller)
        # Write w_controller back to parameters
        offset = 0
        for p in all_params:
            numel = p.numel()
            p.data.copy_(w_controller[offset : offset + numel].view(p.shape))
            offset += numel
