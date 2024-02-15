from typing import Optional
from typing_extensions import Self  # FUTURE: in typing from 3.11

from torch import Tensor

from cirkit.new.layers.inner.product.product import ProductLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


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

    @classmethod
    def _infer_num_prod_units(cls, num_input_units: int, arity: int = 2) -> int:
        """Infer the number of product units in the layer based on given information.

        Args:
            num_input_units (int): The number of input units.
            arity (int, optional): The arity of the layer. Defaults to 2.

        Returns:
            int: The inferred number of product units.
        """
        return num_input_units

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        return self.comp_space.prod(x, dim=0, keepdim=False)  # shape (H, *B, K) -> (*B, K).

    @classmethod
    def get_product(
        cls, self_symb_cfg: SymbLayerCfg[Self], other_symb_cfg: SymbLayerCfg[Self]
    ) -> SymbLayerCfg[Self]:
        """Get the symbolic config to construct the product of this Hadamard layer with the other \
        Hadamard layer.

        The two inner layers for product must be of the same layer class. This means that the \
        product must be performed between two circuits with the same region graph.

        The layer config is unchanged, because:
            (a_1 ⊙ a_2 ⊙ ... ⊙ a_n) ⊗ (b_1 ⊙ b_2 ⊙ ... ⊙ b_n)
            = (a_1 ⊗ b_1) ⊙ (a_2 ⊗ b_2) ⊙ ... ⊙ (a_n ⊗ b_n).

        Args:
            self_symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            other_symb_cfg (SymbLayerCfg[Self]): The symbolic config for the other layer.

        Returns:
            SymbLayerCfg[Self]: The symbolic config for the product of the two Hadamard layers.
        """
        return self_symb_cfg
