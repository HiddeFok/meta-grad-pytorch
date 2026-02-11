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

    def _clip_gradients(self, grad, B_current, B_prev):
        return grad.mul_(B_prev / B_current)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            sigma = group['sigma']
            D_inf = group['D_inf']
            for p in group['params']:
                loss = closure()

            state = self.state[p]
            if len(state) == 0:
                state['w_etas'][] = torch.zeros_like()