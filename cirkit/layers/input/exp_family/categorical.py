import functools
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.reparams.exp_family import ReparamEFCategorical
from cirkit.utils.type_aliases import ReparamFactory

from .exp_family import ExpFamilyLayer


class CategoricalLayer(ExpFamilyLayer):
    """The categorical distribution layer."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_vars: int,
        num_channels: int = 1,
        num_replicas: int = 1,
        num_input_units: Literal[1] = 1,
        num_output_units: int,
        arity: Literal[1] = 1,
        num_folds: Literal[0] = 0,
        fold_mask: None = None,
        reparam: ReparamFactory = ReparamEFCategorical,
        num_categories: int,
    ) -> None:
        """Init class.

        Args:
            num_vars (int): The number of variables of the circuit.
            num_channels (int, optional): The number of channels of each variable. Defaults to 1.
            num_replicas (int, optional): The number of replicas for each variable. Defaults to 1.
            num_input_units (Literal[1], optional): The number of input units, must be 1. \
                Defaults to 1.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            num_folds (Literal[0], optional): The number of folds. Should not be provided and will \
                be calculated as num_vars*num_replicas. Defaults to 0.
            fold_mask (None, optional): The mask of valid folds, must be None. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. \
                Defaults to ReparamEFCategorical.
            num_categories (int): The number of categories for categorical distribution.
        """
        assert (
            num_categories > 0
        ), "The number of categories for categorical distribution must be positive."
        super().__init__(
            num_vars=num_vars,
            num_channels=num_channels,
            num_replicas=num_replicas,
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            num_folds=num_folds,
            fold_mask=fold_mask,
            reparam=functools.partial(reparam, num_categories=num_categories),
            num_suff_stats=num_channels * num_categories,
        )
        self.num_categories = num_categories

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, D, S).
        """
        if x.is_floating_point():
            x = x.long()
        # TODO: pylint issue?
        # pylint: disable-next=not-callable
        suff_stats = F.one_hot(x, self.num_categories)  # shape (*B, D, C, cat)
        suff_stats = suff_stats.to(torch.get_default_dtype())
        return suff_stats.flatten(start_dim=-2)  # shape (*B, D, S=C*cat)

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (*B, D, C).

        Returns:
            Tensor: The natural parameters eta, shape (*B, D).
        """
        return torch.zeros(()).to(x).expand_as(x[..., 0])

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (D, K, P, S).

        Returns:
            Tensor: The log partition function A, shape (D, K, P).
        """
        return torch.zeros(()).to(eta).expand_as(eta[..., 0])

    @property
    def probs(self) -> Tensor:
        """The parameter probs of each category for categorical distribution, \
        shape (D, K, P, C, cat)."""
        # TODO: x.unflatten is not typed
        return torch.unflatten(
            torch.exp(self.params()), dim=-1, sizes=(self.num_channels, self.num_categories)
        )
