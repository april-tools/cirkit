import functools

import torch
from torch import nn

from cirkit.new.layers.inner.inner import InnerLayer
from cirkit.new.reparams import Reparameterization


class SumProductLayer(InnerLayer):
    """The abstract base class for "fused" sum-product layers.

    The fusion of sum and product can sometimes save the instantiation of the product units, but \
    the sum units are limited to dense connection along the units dim, and the arity is only for \
    product. The sum over different layers, a MixingLayer is still required.
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        for child in self.children():
            if isinstance(child, Reparameterization):
                child.initialize(functools.partial(nn.init.uniform_, a=0.01, b=0.99))
