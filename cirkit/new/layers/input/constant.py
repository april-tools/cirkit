from typing import Optional
from typing_extensions import Self  # FUTURE: in typing from 3.11

from torch import Tensor

from cirkit.new.layers.input.input import InputLayer
from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


class ConstantLayer(InputLayer):
    """The constant input layer, with no parameters."""

    # We still accept any Reparameterization instance for reparam, but it will be ignored.
    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Optional[Reparameterization] = None,
        const_value: float,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Optional[Reparameterization], optional): Ignored. This layer has no params. \
                Defaults to None.
            const_value (float): The constant value, in linear space.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=None,
        )

        self.const_value = const_value

    @property
    def _default_initializer_(self) -> None:
        """The default inplace initializer for the parameters of this layer.

        No initialization, as ConstantLayer has no parameters.
        """
        return None

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.comp_space.from_linear(x.new_full((), self.const_value)).expand(
            *x.shape[1:-1], self.num_output_units
        )

    @classmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> SymbCfgFactory[InputLayer]:
        """Get the symbolic config to construct the definite integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.

        Raises:
            ValueError: When const_value != 0, in which case the integral is infinity.

        Returns:
            SymbCfgFactory[InputLayer]: The symbolic config for the integral.
        """
        # IGNORE: Unavoidable for kwargs.
        assert isinstance(
            const_value := symb_cfg.layer_kwargs.get("const_value", None),  # type: ignore[misc]
            (float, int),
        ), "Mismatched kwargs for this layer."

        if const_value:
            raise ValueError(
                "The definite integral of ConstantLayer with const_value!=0 is infinity."
            )

        # IGNORE: Unavoidable for kwargs.
        return SymbCfgFactory(
            layer_cls=ConstantLayer, layer_kwargs={"const_value": 0.0}  # type: ignore[misc]
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
        assert issubclass(left_symb_cfg.layer_cls, cls) or issubclass(
            right_symb_cfg.layer_cls, cls
        ), "At least one of the inputs to InputLayer.get_product must be of self class."

        # The product with ConstantLayer is still ConstantLayer.
        if issubclass(left_symb_cfg.layer_cls, ConstantLayer) and issubclass(
            right_symb_cfg.layer_cls, ConstantLayer
        ):
            # IGNORE: Unavoidable for kwargs.
            return SymbCfgFactory(
                layer_cls=ConstantLayer,
                layer_kwargs={
                    "const_value": left_symb_cfg.layer_kwargs["const_value"]  # type: ignore[misc]
                    * right_symb_cfg.layer_kwargs["const_value"]  # type: ignore[misc]
                },
            )

        # TODO: scaling of other layers.
        return NotImplemented
