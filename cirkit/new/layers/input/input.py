from abc import abstractmethod
from typing import Optional
from typing_extensions import Self  # FUTURE: in typing from 3.11

from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


class InputLayer(Layer):
    """The abstract base class for input layers."""

    # NOTE: We use exactly the same interface (H, *B, K) -> (*B, K) for __call__ of input layers:
    #           1. Define arity(H)=num_vars(D), reusing the H dimension.
    #           2. Define num_input_units(K)=num_channels(C), which reuses the K dimension.
    #       For dimension D (variables), we should parse the input in circuit according to the
    #       scope of the corresponding region node/symbolic input layer.

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        reparam: Optional[Reparameterization] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer, i.e., number of variables in the scope. \
                Defaults to 1.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if the layer has no params. Defaults to None.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

    # TODO: enable integ on part of H dim? if yes, also lift assert subset in functional
    @classmethod
    @abstractmethod
    def get_integral(cls, symb_cfg: SymbLayerCfg[Self]) -> SymbCfgFactory["InputLayer"]:
        """Get the symbolic config to construct the definite integral of this layer.

        Args:
            symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.

        Returns:
            SymbCfgFactory[InputLayer]: The symbolic config for the integral.
        """

    @classmethod
    @abstractmethod
    def get_partial(
        cls, symb_cfg: SymbLayerCfg[Self], *, order: int = 1, var_idx: int = 0, ch_idx: int = 0
    ) -> SymbCfgFactory["InputLayer"]:
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
