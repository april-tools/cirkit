from torch import Tensor

from cirkit.backend.torch.reparams import Reparameterization
from cirkit.layers.input import InputLayer


class ParameterizedConstantLayer(InputLayer):
    """The constant input layer, with parameters and optionally a transform function, but constant \
    w.r.t. input."""

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.params = reparam
        self.materialize_params((num_output_units,), dim=())

    @property
    def _default_initializer_(self) -> None:
        """The default inplace initializer for the parameters of this layer.

        No initialization, as ParameterizedConstantLayer's parameters should come from other layers.
        """
        return None

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        # TODO: comp_space?
        return self.params().expand(*x.shape[1:-1], -1)
