from typing import Any, Optional

import torch
from torch import Tensor, nn

from cirkit.layers.layer import Layer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory


class SumProductLayer(Layer):
    """The abstract base for all "fused" sum-product layers.

    reset_parameters is defined here, but subclasses can override, and they should call it.
    """

    def __init__(  # type: ignore[misc]  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,
        **_: Any,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 2.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (Optional[Tensor], optional): The mask of valid folds. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99)."""
        for param in self.parameters():
            nn.init.uniform_(param, 0.01, 0.99)

    # forward keeps abstract, should be implemented by subclasses
