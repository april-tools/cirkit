from typing import Optional

from torch import Tensor

from cirkit.new.layers.inner.product.product import ProductLayer
from cirkit.new.layers.layer import Layer
from cirkit.new.reparams import Reparameterization
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


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
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.comp_space.prod(x, dim=0, keepdim=False)  # shape (H, *B, K) -> (*B, K).

    @classmethod
    def get_product(
        cls, left_symb_cfg: SymbLayerCfg[Layer], right_symb_cfg: SymbLayerCfg[Layer]
    ) -> SymbCfgFactory[Layer]:
        """Get the symbolic config to construct the product of this layer and the other layer.

        HadamardLayer can only be multiplied with the same class. However, the signature typing is \
        not narrowed down, and wrong arg type will not be captured by static checkers but only \
        during runtime.

        The product with the same class is:
            (a_1 ⊙ a_2 ⊙ ... ⊙ a_n) ⊗ (b_1 ⊙ b_2 ⊙ ... ⊙ b_n) \
            = (a_1 ⊗ b_1) ⊙ (a_2 ⊗ b_2) ⊙ ... ⊙ (a_n ⊗ b_n).

        Args:
            left_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the left operand.
            right_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the right operand.

        Returns:
            SymbCfgFactory[Layer]: The symbolic config for the product. NOTE: Implicit to typing, \
                NotImplemented may also be returned, which indicates the reflection should be tried.
        """
        assert (
            issubclass(left_symb_cfg.layer_cls, cls)
            and left_symb_cfg.layer_cls == right_symb_cfg.layer_cls
        ), "Both inputs to HadamardLayer.get_product must be the same and of self class."

        return left_symb_cfg  # Just reuse the same config.
