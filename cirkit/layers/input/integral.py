from typing import cast

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
        # in_layer should have been inited, reset_parameters only for reset later

    def reset_parameters(self) -> None:
        """Reset parameters of the wrapped layer."""
        self._in_layer.reset_parameters()

    def integrate(self) -> Tensor:
        """Return the definite integral of units activations over the variables domain.

        In case of discrete variables this computes a sum.

        Returns:
            Tensor: The integration of the layer over all variables.
        """
        raise NotImplementedError("The integration of integrated functions is not implemented")

    def __call__(self, x: Tensor, in_mask: Tensor) -> Tensor:  # type: ignore[override]
        """Invoke the forward function.

        Args:
            x (Tensor): The input to this layer, shape (*B, D, C).
            in_mask (Tensor): The mask of variables to integrate, shape (*B, D).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        return super().__call__(x, in_mask)

    # TODO: consider change interface -- among all Layer subclasses only this is different
    # pylint: disable-next=arguments-differ
    def forward(self, x: Tensor, in_mask: Tensor) -> Tensor:  # type: ignore[override]
        """Invoke the forward function.

        Args:
            x (Tensor): The input to this layer, shape (*B, D, C).
            in_mask (Tensor): The mask of variables to integrate, shape (*B, D).

        Returns:
            Tensor: The output of this layer, shape (*B, D, K, P).
        """
        in_mask = in_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)  # shape (*B, D, 1, 1)
        y = self._in_layer(x)  # shape (*B, D, K, P)
        z = self._in_layer.integrate()  # shape (*B, D, K, P)
        # TODO: torch __rsub__ issue
        return y * cast(Tensor, 1 - in_mask) + z * in_mask  # shape (*B, D, K, P)
