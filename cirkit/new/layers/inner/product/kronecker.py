from typing import Literal, Optional, cast
from typing_extensions import Self  # FUTURE: in typing from 3.11

from torch import Tensor

from cirkit.new.layers.inner.product.product import ProductLayer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbLayerCfg


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
        assert (
            num_output_units == num_input_units**arity
        ), "The number of input and output units must be the same for Hadamard product."
        if arity != 2:
            raise NotImplementedError("Kronecker only implemented for binary product units.")
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
        # CAST: int**int is not guaranteed to be int.
        return cast(int, num_input_units**arity)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, K).

        Returns:
            Tensor: The output of this layer, shape (*B, K).
        """
        x0 = x[0].unsqueeze(dim=-1)  # shape (*B, K, 1).
        x1 = x[1].unsqueeze(dim=-2)  # shape (*B, 1, K).
        return self.comp_space.mul(x0, x1).flatten(start_dim=-2)  # shape (*B, K, K) -> (*B, K**2).

    @classmethod
    def get_product(
        cls,
        self_symb_cfg: SymbLayerCfg[Self],
        other_symb_cfg: SymbLayerCfg[Self],
    ) -> SymbLayerCfg[Self]:
        """Get the symbolic config to construct the product of this Kronecker layer with the other \
        Kronecker layer.

        The two inner layers for product must be of the same layer class. This means that the \
        product must be performed between two circuits with the same region graph.

        TODO: we need permutation: \
            (a_1 ⊗ a_2) ⊗ (b_1 ⊗ b_2) = PermuteMatrix((a_1 ⊗ b_1) ⊗ (a_2 ⊗ b_2)).

        Args:
            self_symb_cfg (SymbLayerCfg[Self]): The symbolic config for this layer.
            other_symb_cfg (SymbLayerCfg[Self]): The symbolic config for the other layer.

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked.

        Returns:
            SymbLayerCfg[Self]: The symbolic config for the product of the two Kronecker layers.
        """
        raise NotImplementedError("Product between two kroneker layers is not implemented yet.")
