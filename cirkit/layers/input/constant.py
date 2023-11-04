from typing import Callable, List, Union

import torch
from torch import Tensor

from cirkit.layers.input import InputLayer
from cirkit.region_graph import RegionNode


class ConstantLayer(InputLayer):
    """The constant layer, i.e., an input layer that returns constant values."""

    def __init__(
        self,
        rg_nodes: List[RegionNode],
        value: Union[float, Tensor, Callable[[], Tensor]],
    ):
        """Initialize the constant layer.

        Args:
            rg_nodes: The list of region nodes.
            value: It can be either a float (which is wrapped in a tensor of shape (1,)),
             a tensor, or a function without arguments that returns a tensor.
        """
        super().__init__(rg_nodes)
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

    def forward(self, _: Tensor) -> Tensor:
        """Run forward pass.

        Returns:
            Tensor: The output of this layer.
        """
        if callable(self._value):
            return self._value()
        return self._value
