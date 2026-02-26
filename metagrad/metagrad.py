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

    def _init_state(self, p, sigma):
        """Initialize optimizer state for parameter p. Returns the state dict."""
        state = self.state[p]
        state["step"] = 0
        state["b_t"] = torch.zeros_like(p)
        state["B_t"] = torch.zeros_like(p)
        state["B_t_prev"] = torch.zeros_like(p)
        state["b_sum"] = torch.zeros_like(p)
        state["B_sum"] = torch.zeros_like(p)
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
        return state

    def _update_gradient_info(self, state, p, grad, D_inf):
        """Update b_t, B_t, b_sum, B_sum from the current gradient.

        Returns (p_shape,), (p_shape,), (p_shape,), (p_shape,)
        """
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

    def _update_active_etas(self, state):
        """Compute eta bounds and update the active eta mask.

        Returns (*p_shape, K) boolean tensor.
        """
        eta_max = 1.0 / (2.0 * torch.clamp(state["B_t"], min=1e-10))
        eta_min = 1.0 / (2 * torch.clamp(state["B_sum"] + state["B_t"], min=1e-10))
        state["active_etas"] = torch.logical_and(
            self.eta_grid > eta_min.unsqueeze(-1), self.eta_grid < eta_max.unsqueeze(-1)
        )
        return state["active_etas"]

    def _check_reset(self, state, p):
        """Reset exponential weights where B_t exceeds epoch threshold.

        Modifies state in-place.
        """
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

    def _compute_controller(self, state, p, D_inf):
        """Compute the weighted controller prediction from experts.

        Returns (p_shape,) tensor.
        """
        any_active = state["active_etas"].int().sum(axis=-1) > 0
        w_eta = torch.clamp(state["eta_experts"]["w_hat"], -D_inf, D_inf)
        weight = torch.where(
            state["active_etas"], state["eta_exp_weights"] * self.eta_grid, 0
        )
        weighted_pred = (weight * w_eta).sum(axis=-1)
        weight_sum = weight.sum(axis=-1)
        return torch.where(
            any_active,
            weighted_pred / (weight_sum + 1e-10),
            p,
        )

    def _update_experts(self, state, grad, w_controller, D_inf):
        """Update expert parameters (lambda, w_hat).

        Modifies state in-place.
        """
        w_eta = torch.clamp(state["eta_experts"]["w_hat"], -D_inf, D_inf)
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

    def _update_exp_weights(self, state, grad, w_controller, D_inf):
        """Update exponential weights using clipped gradient losses.

        Modifies state in-place.
        """
        w_eta = torch.clamp(state["eta_experts"]["w_hat"], -D_inf, D_inf)
        any_active = state["active_etas"].int().sum(axis=-1) > 0
        diff = w_eta - w_controller.unsqueeze(-1)
        clipped_grad = (state["B_t_prev"] / state["B_t"] + 1e-10) * grad
        linear_term = self.eta_grid * diff * clipped_grad.unsqueeze(-1)
        expert_losses = linear_term + linear_term**2

        state["eta_exp_weights"].mul_(
            torch.where(state["active_etas"], torch.exp(-expert_losses), 1)
        )
        state["eta_exp_weights"].div_(
            torch.where(
                any_active.unsqueeze(-1),
                (state["eta_exp_weights"] * state["active_etas"]).sum(axis=-1).unsqueeze(-1) + 1e-10,
                1,
            )
        )

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
                    state = self._init_state(p, sigma)

                state["step"] += 1

                self._update_gradient_info(state, p, grad, D_inf)
                self._update_active_etas(state)
                self._check_reset(state, p)
                w_controller = self._compute_controller(state, p, D_inf)
                self._update_experts(state, grad, w_controller, D_inf)
                self._update_exp_weights(state, grad, w_controller, D_inf)
                p.copy_(w_controller)

        return loss


class FullMetaGrad(MetaGradMixin, Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf: float = 1.0):
        defaults = dict(sigma=sigma, D_inf=D_inf)
        super(FullMetaGrad, self).__init__(params, defaults)

        self.grid_size = 20
        self.eta_grid = torch.tensor([2.0**i for i in range(-self.grid_size, 1)])
        self.grid_size = self.eta_grid.shape[0]

    def _flatten_params_and_grads(self):
        """Flatten all parameters and gradients into single vectors.

        Returns (list[Parameter], Tensor of shape (N,), Tensor of shape (N,)).
        """
        all_params = []
        all_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                all_params.append(p)
                all_grads.append(p.grad.flatten())
        g = torch.cat(all_grads)
        w_flat = torch.cat([p.data.flatten() for p in all_params])
        return all_params, w_flat, g

    def _init_state(self, N, w_flat, sigma):
        """Initialize optimizer state for the full-matrix case.

        Returns the state dict.
        """
        state = self.state
        state["step"] = 0
        state["b_t"] = torch.tensor(0.0)
        state["B_t"] = torch.tensor(0.0)
        state["B_t_prev"] = torch.tensor(0.0)
        state["b_sum"] = torch.tensor(0.0)
        state["B_sum"] = torch.tensor(0.0)

        K = self.grid_size
        state["Lambda"] = (torch.eye(N) / (sigma**2)).unsqueeze(-1).repeat(1, 1, K)
        state["Sigma"] = (torch.eye(N) * (sigma**2)).unsqueeze(-1).repeat(1, 1, K)
        state["w_hat"] = w_flat.unsqueeze(-1).repeat(1, K)  # (N, K)
        state["exp_weights"] = torch.ones(K)
        return state

    def _update_gradient_info(self, state, w_flat, g, D_inf):
        """Update scalar b_t, B_t, b_sum, B_sum from the current gradient.

        All returned/stored values are scalar tensors.
        """
        w_proj = torch.clamp(w_flat, -D_inf, D_inf)
        state["b_t"] = (D_inf + w_proj.norm()) * g.norm()
        state["B_t_prev"] = state["B_t"].clone()
        state["B_t"] = torch.max(state["B_t"], state["b_t"])

        if "epoch_start_B" not in state:
            state["epoch_start_B"] = state["B_t"]

        state["b_sum"].add_(state["b_t"] / (state["B_t"] + 1e-10))
        state["B_sum"].add_(state["b_t"] * state["B_t_prev"] / (state["B_t"] + 1e-10))

    def _update_active_etas_and_reset(self, state):
        """Compute active eta mask and perform reset if needed.

        Returns (K,) boolean tensor of active etas.
        """
        K = self.grid_size
        eta_max = 1.0 / (2.0 * state["B_t"] + 1e-10)
        eta_min = 1.0 / (2.0 * (state["B_sum"] + state["B_t"]) + 1e-10)
        active = torch.logical_and(self.eta_grid > eta_min, self.eta_grid < eta_max)

        reset_mask = state["B_t"] > state["epoch_start_B"] * state["b_sum"]
        if reset_mask:
            state["epoch_start_B"] = state["B_t"]
            state["exp_weights"] = torch.where(
                active, torch.ones(K), state["exp_weights"]
            )
        return active

    def _compute_controller(self, state, active, w_flat, D_inf):
        """Compute the weighted controller prediction from experts.

        Returns (N,) tensor.
        """
        w_eta = torch.clamp(state["w_hat"], -D_inf, D_inf)  # (N, K)
        weights = state["exp_weights"] * self.eta_grid * active  # (K,)
        weight_sum = weights.sum()

        if active.any():
            if weight_sum > 1e-10:
                w_controller = (weights.unsqueeze(0) * w_eta).sum(axis=-1) / weight_sum
            else:
                w_controller = w_flat
        else:
            w_controller = w_flat
        return w_controller

    def _update_experts(self, state, g, w_controller, active, D_inf):
        """Update expert covariance (Sigma, Lambda) and predictions (w_hat).

        Modifies state in-place.
        """
        w_eta = torch.clamp(state["w_hat"], -D_inf, D_inf)  # (N, K)

        # Update Sigma
        Sigma_g = torch.einsum("ijk,j -> ik", state["Sigma"], g)  # (N, K)
        Sigma_g_g_Sigma = Sigma_g[:, None, :] * Sigma_g[None, :, :]  # (N, N, K)
        g_Sigma_g = torch.einsum("i, ik -> k", g, Sigma_g)  # (K,)

        state["Sigma"].add_(
            -2
            * (self.eta_grid**2)
            * Sigma_g_g_Sigma
            / (1 + 2 * (self.eta_grid**2) * g_Sigma_g)
            * active
        )
        state["Sigma"] = (state["Sigma"] + state["Sigma"].transpose(0, 1)) / 2

        # Update Lambda
        state["Lambda"].add_(
            (2 * (self.eta_grid**2) * torch.outer(g, g).unsqueeze(-1)) * active
        )

        # Update w_hat
        diff = w_eta - w_controller.unsqueeze(-1)  # (N, K)
        Sigma_g = torch.einsum("ijk,j -> ik", state["Sigma"], g)  # (N, K)
        state["w_hat"].add_(
            -((1 + 2 * self.eta_grid * (diff.T @ g)) * self.eta_grid * Sigma_g) * active
        )

    def _update_exp_weights(self, state, g, w_controller, active, D_inf):
        """Update exponential weights using clipped gradient losses.

        Modifies state in-place.
        """
        w_eta = torch.clamp(state["w_hat"], -D_inf, D_inf)
        diff = w_eta - w_controller.unsqueeze(-1)
        clipped_grad = (state["B_t_prev"] / (state["B_t"] + 1e-10)) * g
        linear_term = self.eta_grid * (diff.T @ clipped_grad)
        expert_losses = linear_term + linear_term**2

        state["exp_weights"].mul_(torch.where(active, torch.exp(-expert_losses), 1))
        state["exp_weights"].div_(torch.where(active, (state["exp_weights"] * active).sum() + 1e-10, 1))

    def _write_back_params(self, all_params, w_controller):
        """Write the flat controller vector back into parameter tensors."""
        offset = 0
        for p in all_params:
            numel = p.numel()
            p.data.copy_(w_controller[offset : offset + numel].view(p.shape))
            offset += numel

    @torch.no_grad()
    def step(self, closure=None):
        all_params, w_flat, g = self._flatten_params_and_grads()
        N = g.shape[0]
        sigma = self.param_groups[0]["sigma"]
        D_inf = self.param_groups[0]["D_inf"]

        state = self.state
        if "step" not in state:
            state = self._init_state(N, w_flat, sigma)

        state["step"] += 1

        self._update_gradient_info(state, w_flat, g, D_inf)
        active = self._update_active_etas_and_reset(state)
        w_controller = self._compute_controller(state, active, w_flat, D_inf)
        self._update_experts(state, g, w_controller, active, D_inf)
        self._update_exp_weights(state, g, w_controller, active, D_inf)
        self._write_back_params(all_params, w_controller)


class SketchedMetaGrad(MetaGradMixin, Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf=1.0, sketch_size=10):
        defaults = dict(sigma=sigma, D_inf=D_inf)
        super(SketchedMetaGrad, self).__init__(params, defaults)

        self.grid_size = 20
        self.eta_grid = torch.tensor([2.0**i for i in range(-self.grid_size, 1)])
        self.grid_size = self.eta_grid.shape[0]

        self.m = sketch_size
        self.k = 2 * sketch_size

    def _flatten_params_and_grads(self):
        """Flatten all parameters and gradients into single vectors.

        Returns (list[Parameter], Tensor of shape (N,), Tensor of shape (N,)).
        """
        all_params = []
        all_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                all_params.append(p)
                all_grads.append(p.grad.flatten())
        g = torch.cat(all_grads)
        w_flat = torch.cat([p.data.flatten() for p in all_params])
        return all_params, w_flat, g

    def _init_state(self, N, K, w_flat, sigma):
        """Initialize optimizer state for the full-matrix case.

        Returns the state dict.
        """
        state = self.state
        state["step"] = 0
        state["b_t"] = torch.tensor(0.0)
        state["B_t"] = torch.tensor(0.0)
        state["B_t_prev"] = torch.tensor(0.0)
        state["b_sum"] = torch.tensor(0.0)
        state["B_sum"] = torch.tensor(0.0)

        # Sketch states
        state["S"] = torch.zeros(self.k, N, K)
        state["H"] = (torch.eye(self.k) * (sigma**2)).unsqueeze(-1).repeat(1, 1, K)

        state["w_hat"] = w_flat.unsqueeze(-1).repeat(1, K)  # (N, K)
        state["exp_weights"] = torch.ones(K)

        state["epoch_counter"] = torch.zeros(K, dtype=torch.long)

        return state

    def _update_gradient_info(self, state, w_flat, g, D_inf):
        """Update scalar b_t, B_t, b_sum, B_sum from the current gradient.

        All returned/stored values are scalar tensors.
        """
        w_proj = torch.clamp(w_flat, -D_inf, D_inf)
        state["b_t"] = (D_inf + w_proj.norm()) * g.norm()
        state["B_t_prev"] = state["B_t"].clone()
        state["B_t"] = torch.max(state["B_t"], state["b_t"])

        if "epoch_start_B" not in state:
            state["epoch_start_B"] = state["B_t"]

        state["b_sum"].add_(state["b_t"] / (state["B_t"] + 1e-10))
        state["B_sum"].add_(state["b_t"] * state["B_t_prev"] / (state["B_t"] + 1e-10))

    def _update_active_etas_and_reset(self, state):
        """Compute active eta mask and perform reset if needed.

        Returns (K,) boolean tensor of active etas.
        """
        K = self.grid_size
        eta_max = 1.0 / (2.0 * state["B_t"] + 1e-10)
        eta_min = 1.0 / (2.0 * (state["B_sum"] + state["B_t"]) + 1e-10)
        active = torch.logical_and(self.eta_grid > eta_min, self.eta_grid < eta_max)

        reset_mask = state["B_t"] > state["epoch_start_B"] * state["b_sum"]
        if reset_mask:
            state["epoch_start_B"] = state["B_t"]
            state["exp_weights"] = torch.where(
                active, torch.ones(K), state["exp_weights"]
            )
        return active
            
    def _compute_controller(self, state, active, w_flat, D_inf):
        """Compute the weighted controller prediction from experts.

        Returns (N,) tensor.
        """
        w_eta = torch.clamp(state["w_hat"], -D_inf, D_inf)  # (N, K)
        weights = state["exp_weights"] * self.eta_grid * active  # (K,)
        weight_sum = weights.sum()

        if active.any():
            if weight_sum > 1e-10:
                w_controller = (weights.unsqueeze(0) * w_eta).sum(axis=-1) / weight_sum
            else:
                w_controller = w_flat
        else:
            w_controller = w_flat
        return w_controller

    def _update_experts(self, K, state, g, w_controller, active, D_inf):
        """Update sketched_expert (H, S) matrices and predictions (w_hat).

        Modifies state in-place.
        """
        tau = torch.remainder(state['epoch_counter'], self.m + 1)
        row_idx = tau + self.m
        state["S"][tau] = g.unsqueeze(-1).repeat(1, K)

    def _update_exp_weights(self, state, g, w_controller, active, D_inf):
        """Update exponential weights using clipped gradient losses.

        Modifies state in-place.
        """
        w_eta = torch.clamp(state["w_hat"], -D_inf, D_inf)
        diff = w_eta - w_controller.unsqueeze(-1)
        clipped_grad = (state["B_t_prev"] / (state["B_t"] + 1e-10)) * g
        linear_term = self.eta_grid * (diff.T @ clipped_grad)
        expert_losses = linear_term + linear_term**2

        state["exp_weights"].mul_(torch.where(active, torch.exp(-expert_losses), 1))
        state["exp_weights"].div_(torch.where(active, (state["exp_weights"] * active).sum() + 1e-10, 1))

    # TODO: FINISH THIS

    def _write_back_params(self, all_params, w_controller):
        """Write the flat controller vector back into parameter tensors."""
        offset = 0
        for p in all_params:
            numel = p.numel()
            p.data.copy_(w_controller[offset : offset + numel].view(p.shape))
            offset += numel


    @torch.no_grad()
    def step(self, closure=None):
        all_params, w_flat, g = self._flatten_params_and_grads()
        N = g.shape[0]

        sigma = self.param_groups[0]['sigma']
        D_inf = self.param_groups[0]['D_inf']
        K = self.grid_size

        state = self.stata
        if "step" not in state:
            state = self._init_state(N, K, w_flat, sigma)

        state["step"] += 1

        self._update_gradient_info(state, w_flat, g, D_inf)
        active = self._update_active_etas_and_reset(state)
        w_controller = self._compute_controller(state, active, w_flat, D_inf)
        self._update_experts(N, K, state, g, w_controller, active, D_inf)
        self._update_exp_weights(state, g, w_controller, active, D_inf)
        self._write_back_params(all_params, w_controller)