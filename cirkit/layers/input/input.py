from abc import abstractmethod
from typing import Any, List

from torch import Tensor

from cirkit.layers import Layer
from cirkit.region_graph import RegionNode


class InputLayer(Layer):
    """Abstract input layer class."""

    def __init__(self, rg_nodes: List[RegionNode], **_: Any):  # type: ignore[misc]
        """Initialize an input layer.

        Args:
            rg_nodes: The region nodes on which it is defined.
            **kwargs: Unused.
        """
        # TODO: what should be here?
        super().__init__(num_input_units=1, num_output_units=1)
        self.rg_nodes = rg_nodes
        self.num_vars = len(set(v for n in rg_nodes for v in n.scope))

    # TODO: this should be in Layer? for some layers it's no-op but interface should exist
    @abstractmethod
    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
