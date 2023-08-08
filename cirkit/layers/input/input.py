from abc import abstractmethod
from typing import Any, List

import torch

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
        super().__init__()
        self.rg_nodes = rg_nodes
        self.num_vars = len(set(v for n in rg_nodes for v in n.scope))

    @abstractmethod
    def integrate(self) -> torch.Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            torch.Tensor: The integration of the layer over all variables.
        """
