from typing import Optional

import torch
from torch import Tensor, nn

from cirkit.old.layers.layer import Layer
from cirkit.old.reparams.leaf import ReparamIdentity
from cirkit.old.reparams.reparam import Reparameterization
from cirkit.old.utils.log_trick import log_func_exp
from cirkit.old.utils.type_aliases import ReparamFactory


class SumLayer(Layer):
    """The sum layer.

    TODO: currently this is only a sum for mixing, but not generic sum layer.
    """

    params: Reparameterization
    """The reparameterizaion that gives the parameters for sum units, shape (F, H, K)."""

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
            fold_mask (Optional[Tensor], optional): The mask of valid folds, shape (F, H). \
                Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        # TODO: can we lift this constraint?
        assert (
            num_input_units == num_output_units
        ), "The sum layer cannot change the number of units."
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=reparam,
        )

        # TODO: better way to handle fold_mask shape? too many None checks
        self.params = reparam(
            (num_folds, arity, num_output_units),
            dim=1,
            mask=fold_mask.unsqueeze(dim=-1) if fold_mask is not None else None,
        )

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
        # TODO: problem with batch dims at the end. any better solution?
        x = (
            x
            if self.fold_mask is None
            else x
            * self.fold_mask.view(self.fold_mask.shape + (1,) * (x.ndim - self.fold_mask.ndim))
        )
        # shape (F, H, K, *B) -> (F, K, *B)
        return torch.einsum("fhk,fhk...->fk...", self.params(), x)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, K, *B).

        Returns:
            Tensor: The output of this layer, shape (F, K, *B).
        """
        return log_func_exp(x, func=self._forward_linear, dim=1, keepdim=False)

    # TODO: see commit 084a3685c6c39519e42c24a65d7eb0c1b0a1cab1 for backtrack
