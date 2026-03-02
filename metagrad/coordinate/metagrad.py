import torch
from torch.optim import Optimizer

from metagrad.metagrad import MetaGradMixin


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
            p.detach().unsqueeze(-1).expand(*p.shape, self.grid_size).clone()
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
        state["B_sum"].add_(state["b_t"] * state["B_t_prev"] / (state["B_t"] + 1e-10))

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
                torch.logical_and(state["active_etas"], reset_mask.unsqueeze(-1)),
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
                (state["eta_exp_weights"] * state["active_etas"])
                .sum(axis=-1)
                .unsqueeze(-1)
                + 1e-10,
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
