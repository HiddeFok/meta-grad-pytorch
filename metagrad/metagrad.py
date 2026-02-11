import torch
from torch.optim import Optimizer

class MetaGrad(Optimizer):
    def __init__(self, params, sigma: float = 1.0, D_inf: float = 1.0):
        defaults = dict(sigma=sigma, D_inf = D_inf)
        super(MetaGrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]["w_etas"] = {}
                self.state[p]["B_prev"] = 0.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            sigma = group['sigma']
            D_inf = group['D_inf']
            for p in group['params']:
                loss = closure()

            grad = p.grad
            state = self.state[p]
            b_t = torch.abs(grad).max().item() * D_inf
            B_t = max(state["B_prev"], b_t)