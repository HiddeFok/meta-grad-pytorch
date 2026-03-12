![code coverage](https://gist.githubusercontent.com/HiddeFok/be967539e481ae6bb8bb785c4cac289e/raw/coverage_badge.svg)
![License](https://img.shields.io/badge/license-GPLv3-green.svg?style=flat)
![python](https://img.shields.io/badge/python-3.12-blue?style=flat&logo=python)


# PyTorch implementation of MetaGrad

This is a *PyTorch* implementation of the optimization procedure  **MetaGrad** proposed in the paper:

[MetaGrad: Adaptation using Multiple Learning Rates in Online Learning
](https://jmlr.org/papers/v22/20-1444.html), Tim van Erven, Wouter M. Koolen, Dirk van der Hoeven, 2021.

This optimizer does not need to be supplied a learning rate. It automatically
adepts the learning rate to the appropriate size. This ensures that the
optimizer will achieve fast rates automatically in settings where there is
*Strong Convexity* and *Exp-Concavity* for example.

There are 3 flavours to this optimizer:

1. `CoordinateMetaGrad`, where each parameter is treated independently
2. `SketchedMetaGrad`, where the covariance matrices are approximated using a Singular Value Decomposition.
3. `FullMetaGrad`, where the complete covariance matrix is tracked. 

## A note on the layerwise update

PyTorch was specifically developed to optimize the parameters of functions with a layer-wise 
architecture. This means that PyTorch uses backpropagation to efficiently go through the layers
in reverse order. The gradients are calculated *on the fly* when reversing through these layers. 
This is done to be memory and computationally efficient. One consequence of this is the boiler plate
loop that one has to implement when writing their own optimizer:

```python
import torch
from torch.optim import Optimizer

class CustomOptimizer(Optimizer)
    ...

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group:
                ...
                # do custom gradient update
```

For any method that uses any information of the form $g_tg_t^{\top}$ or the full
gradient, we first need tor roll out the full gradient, perform the computation
to update the parameters, and then go through the parameters again. 

```python
import torch
from torch.optim import Optimizer

class FullBlockOptimizer(Optimizer):
    ...

    @torch.no_grad()
    def step(self, closure=None):
        # 1. Collect ALL parameters and gradients into flat vectors
        all_params = []
        all_grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                all_params.append(p)
                all_grads.append(p.grad.flatten())

        # 2. Do computation that requires full knowledge
        ...
        # 3. Do uppdate of all parameters
        for p in all_params:
            # update
            ...
```

This  completely erases any benefit PyTorch provides.


### Workaround

The standard workaround is to have a hybrid between a coordinate-wise version and full version, 
by performing the full version for each version. 
