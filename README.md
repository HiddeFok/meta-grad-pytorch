# PyTorch implementation of MetaGrad

1. Coordinate version has a controller per coordinate
2. Sketch version has one controller for all

## Previous problems run in

1. Vector update of Controller part seems to be not efficient
2. Layerwise update has a weird syntax
3. The optimizer has to know about if autograd and if 
        its GPU or CPU which seems weird


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

class FullCustomOptimizer(Optimizer):
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
