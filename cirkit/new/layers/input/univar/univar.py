from abc import abstractmethod
from typing import Literal, Optional
from typing_extensions import Self  # FUTURE: in typing from 3.11

from cirkit.new.layers.input.input import InputLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


class UnivariateInputLayer(InputLayer):
    """The abstract base class for univariate input layers, with only one variable, one channel."""

    def __init__(
        self,
        *,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (Literal[1], optional): The number of input units, i.e. number of \
                channels for variables, must be 1. Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, i.e., number of variables in the \
                scope, must be 1. Defaults to 1.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if the layer has no params. Defaults to None.
        """
        if num_input_units != 1:
            raise ValueError(
                "UnivariateInputLayer can only have one channel; for more than one, use "
                "Multivariate instead."
            )
        if arity != 1:
            raise ValueError(
                "UnivariateInputLayer can only have one variable; for more than one, use "
                "Multivariate instead."
            )
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

    # NOTE: Due to mypy override rules, we cannot annotate var_idx and ch_idx as Literal[0], but
    #       it's OK because we note Literal[0] in docstring and will check in runtime anyway. Same
    #       for all subclasses.
    @classmethod
    @abstractmethod
    def get_partial(
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> SymbCfgFactory[InputLayer]:
        """Get the symbolic config to construct the partial differential w.r.t. the given channel \
        of the given variable in the scope of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            order (int, optional): The order of differentiation. Defaults to 1.
            var_idx (Literal[0], optional): The variable to diffrentiate, must be 0. Defaults to 0.
            ch_idx (Literal[0], optional): The channel of variable to diffrentiate, must be 0. \
                Defaults to 0.

        Returns:
            SymbCfgFactory[InputLayer]: The symbolic config for the partial differential w.r.t. \
                the given channel of the given variable.
        """
