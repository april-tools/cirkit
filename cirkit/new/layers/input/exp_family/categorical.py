from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from cirkit.new.layers.input.exp_family.exp_family import ExpFamilyLayer
from cirkit.new.reparams import Reparameterization


class CategoricalLayer(ExpFamilyLayer):
    """The Categorical distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    # Disable: This __init__ is designed to have these arguments.
    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: Literal[1] = 1,
        reparam: Reparameterization,
        num_categories: int,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units, i.e. number of channels for variables.
            num_output_units (int): The number of output units.
            arity (Literal[1], optional): The arity of the layer, must be 1. Defaults to 1.
            reparam (Reparameterization): The reparameterization for layer parameters.
            num_categories (int): The number of categories for Categorical distribution.
        """
        assert (
            num_categories > 0
        ), "The number of categories for Categorical distribution must be positive."
        self.num_categories = num_categories
        self.suff_stats_shape = (num_input_units, num_categories)
        # Set self.suff_stats_shape before ExpFamilyLayer.__init__. The reparam will be set in
        # ExpFamilyLayer.__init__ to normalize dim=-1 (cat).
        super().__init__(
            num_input_units=num_input_units,
            num_output_units=num_output_units,
            arity=arity,
            reparam=reparam,
        )

    def sufficient_stats(self, x: Tensor) -> Tensor:
        """Calculate sufficient statistics T from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The sufficient statistics T, shape (*B, H, S).
        """
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete.
        # TODO: pylint issue?
        suff_stats = F.one_hot(x, self.num_categories).to(  # pylint: disable=not-callable
            torch.get_default_dtype()
        )  # shape (H, *B, K, cat).
        return suff_stats.movedim(0, -3).flatten(
            start_dim=-2
        )  # shape (H, *B, K, cat) -> (*B, H, *S=(K, cat)) -> (*B, H, S=K*cat).

    def log_base_measure(self, x: Tensor) -> Tensor:
        """Calculate log base measure log_h from input x.

        Args:
            x (Tensor): The input x, shape (H, *B, K).

        Returns:
            Tensor: The natural parameters eta, shape (*B, H).
        """
        return torch.zeros(()).to(x).expand_as(x[..., 0]).movedim(0, -1)

    def log_partition(self, eta: Tensor) -> Tensor:
        """Calculate log partition function A from natural parameters eta.

        Args:
            eta (Tensor): The natural parameters eta, shape (H, K, *S).

        Returns:
            Tensor: The log partition function A, shape (H, K).
        """
        return torch.zeros(()).to(eta).expand(eta.shape[:2])
