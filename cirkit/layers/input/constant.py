from typing import Callable, List, Union

import torch

from cirkit.layers.input import InputLayer
from cirkit.region_graph import RegionNode


class ConstantLayer(InputLayer):
    """The constant layer, i.e., an input layer that returns constant values."""

    def __init__(
        self,
        rg_nodes: List[RegionNode],
        value: Union[float, torch.Tensor, Callable[[], torch.Tensor]],
    ):
        """Initialize the constant layer.

        Args:
            rg_nodes: The list of region nodes.
            value: It can be either a float (which is wrapped in a tensor of shape (1,)),
             a tensor, or a function without arguments that returns a tensor.
        """
        super().__init__(rg_nodes)
        self._value: Union[torch.Tensor, Callable[[], torch.Tensor]] = (
            torch.tensor([value], requires_grad=False)  # type: ignore[misc]
            if isinstance(value, float)
            else value
        )

    def integrate(self) -> torch.Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            torch.Tensor: The integration of the layer over all variables.
        """
        raise NotImplementedError("The integration of constants functions is not implemented")

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Compute the output of the layer.

        Args:
            *args: Unused.
            **kwargs: Unused.

        Returns:
            A tensor.
        """
        if callable(self._value):
            return self._value()
        return self._value
