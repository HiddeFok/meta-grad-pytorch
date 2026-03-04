import pytest
import torch
import torch.nn as nn

from metagrad import CoordinateMetaGrad, FullMetaGrad

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def coord_setup():
    """Create a CoordinateMetaGrad optimizer with a single (5,3) parameter,
    run one gradient step to populate state, and return everything tests need.
    """
    p = nn.Parameter(torch.randn(5, 3))
    opt = CoordinateMetaGrad([p], sigma=1.0, D_inf=2.0)
    K = opt.grid_size

    # Simulate a gradient
    p.grad = torch.randn_like(p)
    state = opt._init_state(p, sigma=1.0)
    state["step"] += 1
    return opt, p, state, K


@pytest.fixture
def full_setup():
    """Create a FullMetaGrad optimizer with two parameters, flatten them,
    initialise state, and return everything tests need.
    """
    p1 = nn.Parameter(torch.randn(4, 3))
    p2 = nn.Parameter(torch.randn(5))
    opt = FullMetaGrad([p1, p2], sigma=1.0, D_inf=2.0)
    K = opt.grid_size
    N = p1.numel() + p2.numel()

    # Simulate gradients
    p1.grad = torch.randn_like(p1)
    p2.grad = torch.randn_like(p2)

    all_params, w_flat, g = opt._flatten_params_and_grads()
    state = {}
    state = opt._init_state(state, N, K, w_flat, sigma=1.0)
    state["step"] += 1
    return opt, all_params, w_flat, g, state, N, K


# ── CoordinateMetaGrad dimension tests ───────────────────────────────────────


class TestCoordinateMetaGradDimensions:
    def test_init_state_shapes(self, coord_setup):
        opt, p, state, K = coord_setup
        P = p.shape  # (5, 3)

        assert state["b_t"].shape == P
        assert state["B_t"].shape == P
        assert state["B_t_prev"].shape == P
        assert state["b_sum"].shape == P
        assert state["B_sum"].shape == P
        assert state["active_etas"].shape == (*P, K)
        assert state["eta_exp_weights"].shape == (*P, K)
        assert state["eta_experts"]["w_hat"].shape == (*P, K)
        assert state["eta_experts"]["lambda"].shape == (*P, K)

    def test_update_gradient_info_shapes(self, coord_setup):
        opt, p, state, K = coord_setup
        P = p.shape

        opt._update_gradient_info(state, p, p.grad, D_inf=2.0)

        assert state["b_t"].shape == P
        assert state["B_t"].shape == P
        assert state["B_t_prev"].shape == P
        assert state["b_sum"].shape == P
        assert state["B_sum"].shape == P
        assert state["epoch_start_B"].shape == P

    def test_update_active_etas_shape(self, coord_setup):
        opt, p, state, K = coord_setup
        P = p.shape

        opt._update_gradient_info(state, p, p.grad, D_inf=2.0)
        active = opt._update_active_etas(state)

        assert active.shape == (*P, K)
        assert active.dtype == torch.bool

    def test_check_reset_preserves_shapes(self, coord_setup):
        opt, p, state, K = coord_setup
        P = p.shape

        opt._update_gradient_info(state, p, p.grad, D_inf=2.0)
        opt._update_active_etas(state)
        opt._check_reset(state, p)

        assert state["eta_exp_weights"].shape == (*P, K)
        assert state["epoch_start_B"].shape == P

    def test_compute_controller_shape(self, coord_setup):
        opt, p, state, K = coord_setup

        opt._update_gradient_info(state, p, p.grad, D_inf=2.0)
        opt._update_active_etas(state)
        w_ctrl = opt._compute_controller(state, p, D_inf=2.0)

        assert w_ctrl.shape == p.shape

    def test_update_experts_preserves_shapes(self, coord_setup):
        opt, p, state, K = coord_setup
        P = p.shape

        opt._update_gradient_info(state, p, p.grad, D_inf=2.0)
        opt._update_active_etas(state)
        w_ctrl = opt._compute_controller(state, p, D_inf=2.0)
        opt._update_experts(state, p.grad, w_ctrl, D_inf=2.0)

        assert state["eta_experts"]["w_hat"].shape == (*P, K)
        assert state["eta_experts"]["lambda"].shape == (*P, K)

    def test_update_exp_weights_preserves_shapes(self, coord_setup):
        opt, p, state, K = coord_setup
        P = p.shape

        opt._update_gradient_info(state, p, p.grad, D_inf=2.0)
        opt._update_active_etas(state)
        w_ctrl = opt._compute_controller(state, p, D_inf=2.0)
        opt._update_exp_weights(state, p.grad, w_ctrl, D_inf=2.0)

        assert state["eta_exp_weights"].shape == (*P, K)

    def test_full_step_preserves_param_shape(self):
        p = nn.Parameter(torch.randn(5, 3))
        opt = CoordinateMetaGrad([p], sigma=1.0, D_inf=2.0)
        original_shape = p.shape

        p.grad = torch.randn_like(p)
        opt.step()

        assert p.shape == original_shape


# ── FullMetaGrad dimension tests ─────────────────────────────────────────────


class TestFullMetaGradDimensions:
    def test_flatten_params_and_grads_shapes(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup

        assert w_flat.shape == (N,)
        assert g.shape == (N,)

    def test_init_state_shapes(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup

        assert state["b_t"].shape == ()
        assert state["B_t"].shape == ()
        assert state["B_t_prev"].shape == ()
        assert state["b_sum"].shape == ()
        assert state["B_sum"].shape == ()
        assert state["Lambda"].shape == (N, N, K)
        assert state["Sigma"].shape == (N, N, K)
        assert state["w_hat"].shape == (N, K)
        assert state["exp_weights"].shape == (K,)

    def test_update_gradient_info_shapes(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup

        opt._update_gradient_info(state, w_flat, g, D_inf=2.0)

        assert state["b_t"].shape == ()
        assert state["B_t"].shape == ()
        assert state["B_t_prev"].shape == ()
        assert state["b_sum"].shape == ()
        assert state["B_sum"].shape == ()
        assert state["epoch_start_B"].shape == ()

    def test_update_active_etas_and_reset_shape(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup

        opt._update_gradient_info(state, w_flat, g, D_inf=2.0)
        active = opt._update_active_etas_and_reset(state)

        assert active.shape == (K,)
        assert active.dtype == torch.bool

    def test_compute_controller_shape(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup

        opt._update_gradient_info(state, w_flat, g, D_inf=2.0)
        active = opt._update_active_etas_and_reset(state)
        w_ctrl = opt._compute_controller(state, active, w_flat, D_inf=2.0)

        assert w_ctrl.shape == (N,)

    def test_update_experts_preserves_shapes(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup

        opt._update_gradient_info(state, w_flat, g, D_inf=2.0)
        active = opt._update_active_etas_and_reset(state)
        w_ctrl = opt._compute_controller(state, active, w_flat, D_inf=2.0)
        opt._update_experts(state, g, w_ctrl, active, D_inf=2.0)

        assert state["Sigma"].shape == (N, N, K)
        assert state["Lambda"].shape == (N, N, K)
        assert state["w_hat"].shape == (N, K)

    def test_update_exp_weights_preserves_shapes(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup

        opt._update_gradient_info(state, w_flat, g, D_inf=2.0)
        active = opt._update_active_etas_and_reset(state)
        w_ctrl = opt._compute_controller(state, active, w_flat, D_inf=2.0)
        opt._update_exp_weights(state, g, w_ctrl, active, D_inf=2.0)

        assert state["exp_weights"].shape == (K,)

    def test_write_back_params_preserves_shapes(self, full_setup):
        opt, all_params, w_flat, g, state, N, K = full_setup
        original_shapes = [p.shape for p in all_params]

        opt._update_gradient_info(state, w_flat, g, D_inf=2.0)
        active = opt._update_active_etas_and_reset(state)
        w_ctrl = opt._compute_controller(state, active, w_flat, D_inf=2.0)
        opt._write_back_params(all_params, w_ctrl)

        for p, expected_shape in zip(all_params, original_shapes):
            assert p.shape == expected_shape

    def test_full_step_preserves_param_shapes(self):
        p1 = nn.Parameter(torch.randn(4, 3))
        p2 = nn.Parameter(torch.randn(5))
        opt = FullMetaGrad([p1, p2], sigma=1.0, D_inf=2.0)
        shapes = [p1.shape, p2.shape]

        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        opt.step()

        assert p1.shape == shapes[0]
        assert p2.shape == shapes[1]
