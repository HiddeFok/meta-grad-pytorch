import torch
from torch.optim import Optimizer

from metagrad.metagrad import MetaGradMixin


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
        state["exp_weights"].div_(
            torch.where(active, (state["exp_weights"] * active).sum() + 1e-10, 1)
        )

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


class FullBlockMetagrad(MetaGradMixin, Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf: float = 1.0):
        defaults = dict(sigma=sigma, D_inf=D_inf)
        super(FullBlockMetagrad, self).__init__(params, defaults)

        self.grid_size = 20
        self.eta_grid = torch.tensor([2.0**i for i in range(-self.grid_size, 1)])
        self.grid_size = self.eta_grid.shape[0]

    def _init_state(self, state, N, K, w_flat, sigma):
        """Initialize optimizer state for the full-matrix case.

        Returns the state dict.
        """
        state["step"] = 0
        state["b_t"] = torch.tensor(0.0)
        state["B_t"] = torch.tensor(0.0)
        state["B_t_prev"] = torch.tensor(0.0)
        state["b_sum"] = torch.tensor(0.0)
        state["B_sum"] = torch.tensor(0.0)

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
        state["exp_weights"].div_(
            torch.where(active, (state["exp_weights"] * active).sum() + 1e-10, 1)
        )

    def _write_back_params(self, p, w_controller):
        """Write the flat controller vector back into parameter tensors."""
        p.data.copy_(w_controller.view(p.shape))

    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            sigma = group["sigma"]
            D_inf = group["D_inf"]

            K = len(self.eta_grid)
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.flatten()
                w_flat = torch.clamp(p.data.flatten(), -D_inf, D_inf)
                N = g.shape[0]

                state = self.state[p]
                if "step" not in state:
                    state = self._init_state(state, N, K, w_flat, sigma)
                state["step"] += 1

                self._update_gradient_info(state, w_flat, g, D_inf)
                active = self._update_active_etas_and_reset(state)
                w_controller = self._compute_controller(state, active, w_flat, D_inf)
                self._update_experts(state, g, w_controller, active, D_inf)
                self._update_exp_weights(state, g, w_controller, active, D_inf)
                self._write_back_params(p, w_controller)
