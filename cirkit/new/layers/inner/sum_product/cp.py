from torch import Tensor

from cirkit.new.layers.inner.product.hadamard import HadamardLayer
from cirkit.new.layers.inner.sum.dense import DenseLayer
from cirkit.new.layers.inner.sum_product.sum_product import SumProductLayer
from cirkit.new.reparams import Reparameterization


class CPLayer(SumProductLayer):
    """The Candecomp Parafac (collapsed) layer, which is a fused dense-hadamard.

    The fusion actually does not gain anything, and is just a plain connection. We don't because \
    it cannot save computation but enforced the product into linear space, which might be worse \
    numerically.
    """

    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        reparam: Reparameterization,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            reparam (Reparameterization): The reparameterization for layer parameters.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

        self.prod_layer = HadamardLayer(  # Any arity but fixed num_units.
            num_input_units=num_input_units,
            num_output_units=num_input_units,
            arity=arity,
            reparam=None,
        )
        self.sum_layer = DenseLayer(  # Fixed arity but any num_units.
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=1,
            reparam=reparam,
        )
        # The params belong to DenseLayer so we don't handle it here.

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
        # shape (H, *B, Ki) -> (*B, Ki) -> (H, *B, Ki) -> (*B, Ko).
        return self.sum_layer(self.prod_layer(x).unsqueeze(dim=0))

    # NOTE: get_product is inherited from SumLayer. The product between CPLayer leads to the
    #       Kronecker of the param, just like DenseLayer. This method will also be called for
    #       SymbProdL, but what's returned is still correct with reparam unused.


# TODO: Uncollapsed?
