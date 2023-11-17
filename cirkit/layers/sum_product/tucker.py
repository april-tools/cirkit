from typing import Literal

import torch
from torch import Tensor

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.reparams.reparam import Reparameterization
from cirkit.utils.log_trick import log_func_exp
from cirkit.utils.type_aliases import ReparamFactory

# TODO: do we support arity>2 (and fold_mask not None)? it's possible but may not be useful


class TuckerLayer(SumProductLayer):
    """Tucker (2) layer."""

    params: Reparameterization
    """The reparameterizaion that gives the parameters for sum units, shape (F, I, J, O)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[2] = 2,
        num_folds: int = 1,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamIdentity,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (Literal[2], optional): The arity of the layer, must be 2. Defaults to 2.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.

        Raises:
            NotImplementedError: When arity is not 2.
        """
        if arity != 2:
            raise NotImplementedError("Tucker layers only implement binary product units.")
        assert fold_mask is None, "Input for Tucker layer should not be masked."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=None,
            reparam=reparam,
        )

        self.params = reparam(
            (num_folds, num_input_units, num_input_units, num_output_units), dim=(1, 2)
        )

        self.reset_parameters()

    def _forward_linear(self, left: Tensor, right: Tensor) -> Tensor:
        # shape (F, I, *B), (F, J ,*B) -> (F, O, *B)
        return torch.einsum("fi...,fj...,fijo->fo...", left, right, self.params())

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, K, *B).

        Returns:
            Tensor: The output of this layer, shape (F, K, *B).
        """
        return log_func_exp(x[:, 0], x[:, 1], func=self._forward_linear, dim=1, keepdim=True)
