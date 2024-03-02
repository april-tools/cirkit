from typing import Literal, Optional, cast

from torch import Tensor

from cirkit.new.layers.inner.product.product import ProductLayer
from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


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
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        x0 = x[0].unsqueeze(dim=-1)  # shape (*B, Ki, 1).
        x1 = x[1].unsqueeze(dim=-2)  # shape (*B, 1, Ki).
        # shape (*B, Ki, Ki) -> (*B, Ko=Ki**2).
        return self.comp_space.mul(x0, x1).flatten(start_dim=-2)

    @classmethod
    def get_product(
        cls, left_symb_cfg: SymbLayerCfg[Layer], right_symb_cfg: SymbLayerCfg[Layer]
    ) -> SymbCfgFactory[Layer]:
        """Get the symbolic config to construct the product of this layer and the other layer.

        KroneckerLayer can only be multiplied with the same class. However, the signature typing \
        is not narrowed down, and wrong arg type will not be captured by static checkers but only \
        during runtime.

        The product with the same class is:
            (a_1 ⊗ a_2) ⊗ (b_1 ⊗ b_2) = permute4D((a_1 ⊗ b_1) ⊗ (a_2 ⊗ b_2)).

        Args:
            left_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the left operand.
            right_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the right operand.

        Raises:
            NotImplementedError: When "not-yet-implemented feature" is invoked.

        Returns:
            SymbCfgFactory[Layer]: The symbolic config for the product. NOTE: Implicit to typing, \
                NotImplemented may also be returned, which indicates the reflection should be tried.
        """
        # TODO: we need permutation. Also, don't forget to add the cls check, see HadamardLayer
        raise NotImplementedError("Product between two kroneker layers is not implemented yet.")
