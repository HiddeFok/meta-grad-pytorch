from typing import Dict


class MetaGradMixin:
    """Class collecting any function that is shared amongst te different
    MetaGrad implementations
    """

    def _init_eta_grid(self):
        return [2**i for i in range(-self.grid_size, 1)]
