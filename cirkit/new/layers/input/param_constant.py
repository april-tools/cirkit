from typing_extensions import Never, Self  # FUTURE: in typing from 3.11

from torch import Tensor

from cirkit.new.layers.input.constant import ConstantLayer
from cirkit.new.layers.input.input import InputLayer
from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


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

    @classmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> Never:
        """Get the symbolic config to construct the definite integral of this layer.

        We don't allow integration of this layer because it only makes sense when the output is \
        zero, which is not guaranteed with changing params.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.

        Raises:
            TypeError: When this method is called on ParameterizedConstantLayer.
        """
        raise TypeError(
            "The definite integral of ParameterizedConstantLayer generally goes to infinity."
        )

    @classmethod
    def get_partial(
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> SymbCfgFactory[InputLayer]:
        """Get the symbolic config to construct the partial differential w.r.t. the given channel \
        of the given variable in the scope of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            order (int, optional): The order of differentiation. Defaults to 1.
            var_idx (int, optional): The variable to diffrentiate. The idx is counted within this \
                layer's scope but not global variable id. Defaults to 0.
            ch_idx (int, optional): The channel of variable to diffrentiate. Defaults to 0.

        Returns:
            SymbCfgFactory[InputLayer]: The symbolic config for the partial differential w.r.t. \
                the given channel of the given variable.
        """
        assert order > 0, "The order of differentiation must be positive."

        # IGNORE: Unavoidable for kwargs.
        return SymbCfgFactory(
            layer_cls=ConstantLayer, layer_kwargs={"const_value": 0.0}  # type: ignore[misc]
        )

    @classmethod
    def get_product(
        cls, left_symb_cfg: SymbLayerCfg[Layer], right_symb_cfg: SymbLayerCfg[Layer]
    ) -> SymbCfgFactory[Layer]:
        """Get the symbolic config to construct the product of this layer and the other layer.

        InputLayer generally can be multiplied with any InputLayer, yet specific combinations may \
        be unimplemented. However, the signature typing is not narrowed down, and wrong arg type \
        will not be captured by static checkers but only during runtime.

        Args:
            left_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the left operand.
            right_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the right operand.

        Returns:
            SymbCfgFactory[Layer]: The symbolic config for the product. NOTE: Implicit to typing, \
                NotImplemented may also be returned, which indicates the reflection should be tried.
        """
        # TODO: merge with ConstL?
        return NotImplemented
