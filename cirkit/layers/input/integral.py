from typing import List

import torch

from cirkit.atlas.integrate import IntegrationContext
from cirkit.layers.input import InputLayer
from cirkit.region_graph import RegionNode


class IntegralLayer(InputLayer):
    """The integral layer.

    Computes the integral of another input layers over some variables.
    """

    def __init__(
        self, rg_nodes: List[RegionNode], in_layer: InputLayer, icontext: IntegrationContext
    ):
        """Initialize an integral layer.

        Args:
            rg_nodes: The region nodes on which it is defined.
            in_layer: The input layer on which integration is applied.
            icontext: The integration context.
        """
        super().__init__(rg_nodes)
        self._in_layer = in_layer
        self._icontext = icontext

    def integrate(self) -> torch.Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            torch.Tensor: The integration of the layer over all variables.
        """
        raise NotImplementedError("The integration of integrated functions is not implemented yet")

    # pylint: disable-next=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute the output of the layer.

        Args:
            x: The input tensor.

        Returns:
            Tensor: The integration output.
        """
        imask = self._icontext.as_mask.unsqueeze(dim=2).unsqueeze(dim=3)
        # y: (batch_size, num_variables, num_units, num_replicas)
        y = self._in_layer(x)
        assert (
            imask.shape[0] == 1 or imask.shape[0] == y.shape[0]
        ), "The integration context batch dimension is unbroadcastable"
        z = self._in_layer.integrate()
        return y * (1 - imask) + z * imask  # type: ignore[no-any-return,misc]
