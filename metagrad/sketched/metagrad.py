import torch
from torch.optim import Optimizer

from metagrad.metagrad import MetaGradMixin

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

    def _tau_less_update(self, row_idx, K, g):
        e = torch.zeros(self.k, K)
        e[row_idx, :] = torch.ones(K)
        S_g = torch.einsum("ijk,j -> ik", self.state["S"], g)
        q = 2 * (self.eta_grid**2) * (S_g - 0.5 * (g @ g) * e)

        H_q = torch.einsum("ijk,j -> ik", self.state["H"], q)
        H_e = torch.einsum("ijk,jk-> ik", self.state["H"], e)
        H_q_e_H = torch.einsum("ik,jk -> ijk", H_q, H_e )
        e_H_q = torch.einsum("ik,i -> k", H_e, q)

        H_tilde = self.state["H" ] - H_q_e_H /  (1 + e_H_q)

        H_q = torch.einsum("ijk,j -> ik", H_tilde, q)
        H_e = torch.einsum("ijk,jk-> ik", H_tilde, e)
        H_q_e_H = torch.einsum("ik,jk -> ijk", H_e, H_q )
        e_H_q = torch.einsum("ik,i -> k", H_q, e)

        return self.state["H"] - H_q_e_H /  (1 + e_H_q)
        

    def _tau_more_update(self):

    def _update_experts(self, K, state, g, w_controller, active, D_inf):
        """Update sketched_expert (H, S) matrices and predictions (w_hat).

        Modifies state in-place.
        """
        tau = torch.remainder(state['epoch_counter'], self.m + 1)
        row_idx = tau + self.m
        state["S"][tau, :, :] = g.unsqueeze(-1).repeat(1, K)

        H_1 = self._tau_less_update(row_idx, K, g)
        H_2 = self._tau_more_update()

        state["H"] = torch.where(tau < self.m, H_1, H_2)

    # TODO: FINISH THIS

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