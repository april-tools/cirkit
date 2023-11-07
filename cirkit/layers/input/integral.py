from torch import Tensor

from cirkit.layers.input import InputLayer

# TODO: rework interface and docstring, all tensors should be properly shaped


class IntegralInputLayer(InputLayer):
    """The integral layer.

    Computes the integral of another input layers over some variables.
    """

    def __init__(self, in_layer: InputLayer):
        """Initialize an integral layer.

        Args:
            in_layer: The input layer on which integration is applied.
        """
        # TODO: what should be here? none of them is used so current all 1s
        super().__init__(num_vars=1, num_output_units=1)
        self._in_layer = in_layer

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
        raise NotImplementedError("The integration of integrated functions is not implemented")

    def __call__(self, x: Tensor, in_mask: Tensor) -> Tensor:  # type: ignore[override]
        """Invoke forward function.

        Args:
            x: The input tensor of shape (batch_size, num_vars, num_channels).
            in_mask: The mask of variables to integrate of shape
             (batch_size, num_vars) or (1, num_vars).

        Returns:
            Tensor: The integration output.
        """
        return super().__call__(x, in_mask)

    # TODO: consider change interface -- among all Layer subclasses only this is different
    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor, in_mask: Tensor) -> Tensor:  # type: ignore[override]
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
