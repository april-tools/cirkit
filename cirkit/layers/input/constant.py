from typing import Callable, Union

import torch
from torch import Tensor

from cirkit.layers.input import InputLayer

# TODO: rework interface and docstring, the const value should be properly shaped


class ConstantLayer(InputLayer):
    """The constant layer, i.e., an input layer that returns constant values."""

    def __init__(self, value: Union[float, Tensor, Callable[[], Tensor]]):
        """Initialize the constant layer.

        Args:
            rg_nodes: The list of region nodes.
            value: It can be either a float (which is wrapped in a tensor of shape (1,)),
             a tensor, or a function without arguments that returns a tensor.
        """
        # TODO: what should be here? none of them is used so current all 1s
        super().__init__(num_vars=1, num_output_units=1)
        self._value: Union[Tensor, Callable[[], Tensor]] = (
            torch.tensor([value], requires_grad=False)  # type: ignore[misc]
            if isinstance(value, float)
            else value
        )

    def reset_parameters(self) -> None:
        """Do nothing.

        This layer does not have any parameters.
        """

    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
        raise NotImplementedError("The integration of constants functions is not implemented")

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (*B, D, C).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        # TODO: shape of x should be carried to output
        if callable(self._value):
            return self._value()
        return self._value  # TODO: shape of _value???
