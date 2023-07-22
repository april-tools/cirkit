import torch

from cirkit.layers.input import InputLayer


class IntegralInputLayer(InputLayer):
    """The integral layer.

    Computes the integral of another input layers over some variables.
    """

    def __init__(self, in_layer: InputLayer):
        """Initialize an integral layer.

        Args:
            in_layer: The input layer on which integration is applied.
        """
        super().__init__(in_layer.rg_nodes)
        self._in_layer = in_layer

    def integrate(self) -> torch.Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            torch.Tensor: The integration of the layer over all variables.
        """
        raise NotImplementedError("The integration of integrated functions is not implemented")

    # pylint: disable-next=arguments-differ
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, in_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the output of the layer.

        Args:
            x: The input tensor of shape (batch_size, num_vars, num_channels).
            in_mask: The mask of variables to integrate of shape
             (batch_size, num_vars) or (1, num_vars).

        Returns:
            Tensor: The integration output.
        """
        in_mask = in_mask.unsqueeze(dim=2).unsqueeze(dim=3)
        # y: (batch_size, num_variables, num_units, num_replicas)
        y = self._in_layer(x)
        assert (
            in_mask.shape[0] == 1 or in_mask.shape[0] == y.shape[0]
        ), "The integration context batch dimension is unbroadcastable"
        z = self._in_layer.integrate()
        return y * (1 - in_mask) + z * in_mask  # type: ignore[no-any-return,misc]
