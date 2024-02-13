from abc import abstractmethod
from typing import Optional
from typing_extensions import Self  # FUTURE: in typing from 3.11

from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


class InnerLayer(Layer):
    """The abstract base class for inner layers."""

    # __init__ is overriden here to change the default value of arity, as arity=2 is the most common
    # case for all inner layers.
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
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if the layer has no params. Defaults to None.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

    @classmethod
    @abstractmethod
    def _infer_num_prod_units(cls, num_input_units: int, arity: int = 2) -> int:
        """Infer the number of product units in the layer based on given information.

        Args:
            num_input_units (int): The number of input units.
            arity (int, optional): The arity of the layer. Defaults to 2.

        Returns:
            int: The inferred number of product units.
        """

    @classmethod
    @abstractmethod
    def get_product(
        cls,
        self_symb_cfg: SymbLayerCfg[Self],
        other_symb_cfg: SymbLayerCfg[Self],
    ) -> SymbLayerCfg[Self]:
        """Get the symbolic config to construct the product of this inner layer with the other \
        inner layer.

        The two inner layers for product must be of the same layer class. This means that the \
        product must be performed between two circuits with the same region graph.

        Args:
            self_symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            other_symb_cfg (SymbLayerCfg[Self]): The symbolic config for the other layer.

        Returns:
            SymbLayerCfg[Self]: The symbolic config for the product of the two inner layers.
        """
