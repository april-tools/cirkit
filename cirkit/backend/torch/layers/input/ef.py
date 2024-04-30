from abc import abstractmethod
from typing import Dict, Optional

import torch
from torch import Tensor, distributions, nn
from torch.nn import functional as F

from cirkit.backend.torch.layers.input.base import TorchInputLayer
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.semiring import SemiringCls
from cirkit.backend.torch.utils import InitializerFunc


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
        num_variables: int,
        num_output_units: int,
        *,
        num_channels: int = 1,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
        """
        super().__init__(
            num_variables, num_output_units, num_channels=num_channels, semiring=semiring
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.semiring.from_lse_sum(self.log_score(x))

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
        num_variables: int,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_categories: int = 2,
        logits: AbstractTorchParameter,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels.
            num_categories (int): The number of categories for Categorical distribution. Defaults to 2.
            logits (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        assert (
            num_categories > 0
        ), "The number of categories for Categorical distribution must be positive."
        assert logits.shape == (num_variables, num_output_units, num_channels, num_categories)
        super().__init__(
            num_variables, num_output_units, num_channels=num_channels, semiring=semiring
        )
        self.num_categories = num_categories
        self.logits = logits

    @staticmethod
    def default_initializers() -> Dict[str, InitializerFunc]:
        return dict(logits=lambda t: nn.init.normal_(t, mean=0.0, std=1e-1))

    def log_score(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete.
        x = F.one_hot(x, self.num_categories)  # (C, *B, D, num_categories)
        x = x.to(torch.get_default_dtype())
        x = torch.einsum("cbdi,dkci->bk", x, self.logits())
        return x


class TorchGaussianLayer(TorchExpFamilyLayer):
    """The Normal distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    def __init__(
        self,
        num_variables: int,
        num_output_units: int,
        *,
        num_channels: int = 1,
        mean: AbstractTorchParameter,
        stddev: AbstractTorchParameter,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_variables (int): The number of variables.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
            mean (AbstractTorchParameter): The reparameterization for layer parameters.
            stddev (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        super().__init__(
            num_variables, num_output_units, num_channels=num_channels, semiring=semiring
        )
        self.mean = mean
        self.stddev = stddev

    @staticmethod
    def default_initializers() -> Dict[str, InitializerFunc]:
        return dict(
            mean=lambda t: nn.init.normal_(t, mean=0.0, std=3e-1),
            stddev=lambda t: nn.init.normal_(t, mean=0.0, std=3e-1),
        )

    def log_score(self, x: torch.Tensor) -> torch.Tensor:
        # (C, B, D)
        log_prob = distributions.Normal(loc=self.mean, scale=self.stddev).log_prob(x)
        return log_prob
