from typing import Optional

import torch
from torch import Tensor, nn

from cirkit.layers.layer import Layer
from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.log_trick import log_func_exp
from cirkit.utils.type_aliases import ReparamFactory


class SumLayer(Layer):
    """The layer of sum units.

    Can be used as general sum layer or to complement sum-product layers.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,
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
        assert num_input_units == num_output_units
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
        )

        # TODO: how to annotate shapes for reparams?
        self.params = reparam((num_folds, arity, num_output_units), dim=1, mask=fold_mask)

        # TODO: should not init if reparam is composed from other reparams?
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters to default: U(0.01, 0.99) with normalization."""
        # TODO: is this still correct with reparam and fold_mask?
        for param in self.parameters():
            nn.init.uniform_(param, 0.01, 0.99)
            # TODO: pylint bug?
            # pylint: disable-next=redefined-loop-name
            param /= param.sum(dim=1, keepdim=True)  # type: ignore[misc]

    # TODO: too many `self.fold_mask is None` checks across the repo
    #       can use apply_mask method?
    def _forward_linear(self, x: Tensor) -> Tensor:
        weight = self.params() if self.fold_mask is None else self.params() * self.fold_mask
        return torch.einsum("fhk,fhkb->fkb", weight, x)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, K, B).

        Returns:
            Tensor: The output of this layer, shape (F, K, B).
        """
        return log_func_exp(x, func=self._forward_linear, dim=1, keepdim=False)

    # TODO: see commit 084a3685c6c39519e42c24a65d7eb0c1b0a1cab1 for backtrack
