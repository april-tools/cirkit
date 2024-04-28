from abc import abstractmethod
from typing import Tuple, Optional

import torch
from torch import Tensor, distributions
from torch.nn import functional as F

from cirkit.backend.torch.layers import TorchInputLayer
from cirkit.backend.torch.params.base import AbstractTorchParameter


class TorchExpFamilyLayer(TorchInputLayer):
    """The abstract base class for Exponential Family distribution layers.

    Exponential Family dist:
        p(x|θ) = exp(η(θ) · T(x) + log_h(x) - A(η)).
    Ref: https://en.wikipedia.org/wiki/Exponential_family#Table_of_distributions.

    However here we directly parameterize η instead of θ:
        p(x|η) = exp(η · T(x) + log_h(x) - A(η)).
    Implemtations provide inverse mapping from η to θ.

    This implementation is fully factorized over the variables if used as multivariate, i.e., \
    equivalent to num_vars (arity) univariate EF distributions followed by a Hadamard product of \
    the same arity.
    """

    def __init__(
        self,
        *,
        num_variables: int,
        num_output_units: int,
        num_channels: int,
        log_partition: Optional[AbstractTorchParameter] = None,
    ) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels.
        """
        super().__init__(
            num_variables=num_variables,
            num_output_units=num_output_units,
            num_channels=num_channels
        )
        self.log_partition = log_partition

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.comp_space.from_log(self.log_score(x))

    @abstractmethod
    def log_score(self, x: Tensor) -> Tensor:
        ...


class TorchCategoricalLayer(TorchExpFamilyLayer):
    """The Categorical distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        *,
        num_variables: int,
        num_output_units: int,
        num_channels: int,
        num_categories: int,
        logits: AbstractTorchParameter,
        log_partition: Optional[AbstractTorchParameter] = None,
    ) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels.
            num_categories (int): The number of categories for Categorical distribution. Defaults to 2.
            logits (AbstractTorchParameter): The reparameterization for layer parameters.
            log_partition (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        assert (
            num_categories > 0
        ), "The number of categories for Categorical distribution must be positive."
        assert logits.shape == (num_variables, num_output_units, num_channels, num_categories)
        self.num_categories = num_categories
        # Set self.suff_stats_shape before ExpFamilyLayer.__init__. The reparam will be set in
        # ExpFamilyLayer.__init__ to normalize dim=-1 (cat).
        super().__init__(
            num_variables=num_variables,
            num_output_units=num_output_units,
            num_channels=num_channels,
            log_partition=log_partition
        )
        self.logits = logits

    def log_score(self, x: Tensor) -> Tensor:
        x = F.one_hot(x, self.num_categories)  # (H, *B, K, num_categories)
        return torch.einsum('cbdk,dick->bc')


class TorchGaussianLayer(TorchExpFamilyLayer):
    """The Normal distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    def __init__(
        self,
        *,
        num_variables: int,
        num_output_units: int,
        num_channels: int,
        mean: AbstractTorchParameter,
        stddev: AbstractTorchParameter,
        log_partition: Optional[AbstractTorchParameter] = None
    ) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels.
            mean (AbstractTorchParameter): The reparameterization for layer parameters.
            stddev (AbstractTorchParameter): The reparameterization for layer parameters.
            log_partition (Optional[AbstractTorchParameter]): The reparameterization for layer parameters.
        """
        super().__init__(
            num_variables=num_variables,
            num_output_units=num_output_units,
            num_channels=num_channels,
            log_partition=log_partition
        )
        self.mean = mean
        self.stddev = stddev

    def log_score(self, x: torch.Tensor) -> torch.Tensor:
        # (C, B, D)
        log_prob = distributions.Normal(loc=self.mean, scale=self.stddev).log_prob(x)
        return log_prob
