from typing import Callable

import torch
from torch.optim import Optimizer

from cirkit.backend.torch.circuits import TorchCircuit
from cirkit.backend.torch.layers.base import TorchLayer


def _create_hook(layer: TorchLayer):
    def _hook(*args, **kwargs):
        layer.em_accumulate()

    return _hook


class EM(Optimizer):
    """Expectation Maximization optimizer for torch backend."""

    def __init__(self, pc: TorchCircuit, lr: float, pseudocount: float, alpha: float = 1e-8):
        """Initialize the optimizer.

        Args:
            pc: Compiled circuit to optimized.
            lr: Learning rate / Step size.
            pseudocount: Pseudocount for laplace smoothing (when implemented).
            alpha: Minimal value for clamping gradients.

        Raises:
            ValueError: Raised if the step size is not between 0 and 1.
        """
        if not 0.0 <= lr <= 1.0:
            raise ValueError("lr must be in [0,1]")
        defaults = dict(lr=lr, pseudocount=pseudocount, alpha=alpha)
        super().__init__(pc.parameters(), defaults)
        self.pc = pc

        for layer in self.pc.layers:
            layer.enable_em()

            # Accumulate the update right after the gradient are computed
            # we use the first parameter of the layer to trigger the layer update
            # which updates all parameters.
            params = list(layer.parameters())
            if len(params) > 0:
                params[0].register_post_accumulate_grad_hook(_create_hook(layer))

    def step(self, closure: Callable | None = None):
        """Update parameters.

        Args:
            closure: Not used, present for compatibility.
        """
        for layer in self.pc.layers:
            layer.em_step(
                self.param_groups[0]["lr"],
                self.param_groups[0]["pseudocount"],
                self.param_groups[0]["alpha"],
            )
