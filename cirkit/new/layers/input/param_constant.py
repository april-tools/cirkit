from typing import Callable
from typing_extensions import Never, Self  # FUTURE: in typing from 3.11

from torch import Tensor

from cirkit.new.layers.input.constant import ConstantLayer
from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class ParameterizedConstantLayer(InputLayer):
    """The constant input layer, with parameters and optionally a transform function, but constant \
    w.r.t. input."""

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Reparameterization,
        func: Callable[[Tensor], Tensor] = lambda x: x,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
            func (Callable[[Tensor], Tensor], optional): The function to transform the parameters \
                to output. Defaults to lambda x: x.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.params = reparam
        # NOTE: We cannot materialize because we don't know what the shape of params should be but
        #       only the shape after func. Therefore we skip materialize/initialize here.
        assert (
            self.params.is_materialized
        ), "The reparam for ParameterizedConstantLayer must be materialized before passed to layer."
        # TODO: reparam.is_materialized?? put self.params = reparam to super()?

        self.func = func

        # NOTE: The output is constant to x, so should only have shape (K,).
        assert func(reparam()).shape == (
            num_output_units,
        ), "The actual output shape does not match the expected."

    def reset_parameters(self) -> None:
        """Do nothing, as the parameters should be handled by other layers."""

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        return self.func(self.params()).expand(*x.shape[1:-1], self.num_output_units)

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
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the partial differential w.r.t. the given channel \
        of the given variable in the scope of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            order (int, optional): The order of differentiation. Defaults to 1.
            var_idx (int, optional): The variable to diffrentiate. The idx is counted within this \
                layer's scope but not global variable id. Defaults to 0.
            ch_idx (int, optional): The channel of variable to diffrentiate. Defaults to 0.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the partial differential w.r.t. the \
                given channel of the given variable.
        """
        assert order >= 0, "The order of differential must be non-negative."
        if not order:
            return symb_cfg

        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=ConstantLayer, layer_kwargs={"const_value": 0.0}  # type: ignore[misc]
        )

    @classmethod
    def get_product(
        cls,
        self_symb_cfg: SymbLayerCfg[Self],
        other_symb_cfg: SymbLayerCfg[InputLayer],
    ) -> SymbLayerCfg[InputLayer]:
        """Get the symbolic config to construct the product of this input layer with the other \
        input layer.

        Args:
            self_symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            other_symb_cfg (SymbLayerCfg[InputLayer]): The symbolic config for the other layer, \
                must be of InputLayer.

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[InputLayer]: The symbolic config for the product of the two input layers.
        """
        raise NotImplementedError(
            "Product for constant input layer and other input layers not implemented."
        )
