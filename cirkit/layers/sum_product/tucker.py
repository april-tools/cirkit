from typing import Literal

import torch
from torch import Tensor

from cirkit.layers.sum_product.sum_product import SumProductLayer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.log_trick import log_func_exp
from cirkit.utils.type_aliases import ReparamFactory


class TuckerLayer(SumProductLayer):
    """Tucker (2) layer."""

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
        """
        if arity != 2:
            raise NotImplementedError("Tucker layers only implemented binary product units.")
        assert fold_mask is None, "Input for Tucker layer should not be masked."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
        )

        self.params = reparam(
            (num_folds, num_input_units, num_input_units, num_output_units), dim=(1, 2)
        )

        self.reset_parameters()

    def _forward_linear(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.einsum("fib,fjb,fijo->fob", left, right, self.params())

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer.

        Returns:
            Tensor: The output of this layer.
        """
        return log_func_exp(x[:, 0], x[:, 1], func=self._forward_linear, dim=1, keepdim=True)
