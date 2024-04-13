from typing import Callable, Literal, Optional

from torch import Tensor

from cirkit.backend.torch.reparams import Reparameterization
from cirkit.layers.inner import InnerLayer


class ProductLayer(InnerLayer):
    """The abstract base class for product layers."""

    # NOTE: We don't change the __init__ of InnerLayer here. We still accept any Reparameterization
    #       instance in ProductLayer, but it will be ignored.

    # NOTE: We need to annotate as Optional instead of None to make SumProdL work.
    @property
    def _default_initializer_(self) -> Optional[Callable[[Tensor], Tensor]]:
        """The default inplace initializer for the parameters of this layer.

        No initialization, as ProductLayer has no parameters.
        """
        return None


class HadamardLayer(ProductLayer):
    """The Hadamard product layer."""

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be the same as input.
            arity (int, optional): The arity of the layer. Defaults to 2.
            reparam (Optional[Reparameterization], optional): Ignored. This layer has no params. \
                Defaults to None.
        """
        assert (
            num_output_units == num_input_units
        ), "The number of input and output units must be the same for Hadamard product."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.comp_space.prod(x, dim=0, keepdim=False)  # shape (H, *B, K) -> (*B, K).


class KroneckerLayer(ProductLayer):
    """The Kronecker product layer."""

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[2] = 2,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units, must be input**arity.
            arity (Literal[2], optional): The arity of the layer, must be 2. Defaults to 2.
            reparam (Optional[Reparameterization], optional): Ignored. This layer has no params. \
                Defaults to None.
        """
        assert num_output_units == num_input_units**arity, (
            "The number of output units must be the number of input units raised to the power of "
            "arity for Kronecker product."
        )
        if arity != 2:
            raise NotImplementedError("Kronecker only implemented for binary product units.")
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        x0 = x[0].unsqueeze(dim=-1)  # shape (*B, Ki, 1).
        x1 = x[1].unsqueeze(dim=-2)  # shape (*B, 1, Ki).
        # shape (*B, Ki, Ki) -> (*B, Ko=Ki**2).
        return self.comp_space.mul(x0, x1).flatten(start_dim=-2)
