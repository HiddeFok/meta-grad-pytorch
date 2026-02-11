import torch
from torch.optim import Optimizer

class MetaGrad(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()