# PyTorch implementation of MetaGrad

1. Coordinate version has a controller per coordinate
2. Sketch version has one controller for all

## Previous problems run in

1. Vector update of Controller part seems to be not efficient
2. Layerwise update has a weird syntax
3. The optimizer has to know about if autograd and if 
        its GPU or CPU which seems weird


