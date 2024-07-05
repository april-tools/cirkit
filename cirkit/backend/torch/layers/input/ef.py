from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import Tensor, distributions
from torch.nn import functional as F

from cirkit.backend.torch.layers.input.base import TorchInputLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring
from cirkit.utils.scope import Scope


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
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            scope (Scope): The scope the input layer is defined on.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
            num_folds (int): The number of channels. Defaults to 1.
        """
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        return self.semiring.from_lse_sum(self.log_unnormalized_likelihood(x))

    @abstractmethod
    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        ...


class TorchCategoricalLayer(TorchExpFamilyLayer):
    """The Categorical distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        num_categories: int = 2,
        probs: Optional[TorchParameter] = None,
        logits: Optional[TorchParameter] = None,
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            scope (Scope): The scope the input layer is defined on.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels.
            num_folds (int): The number of channels. Defaults to 1.
            num_categories (int): The number of categories for Categorical distribution. Defaults to 2.
            logits (TorchParameter): The reparameterization for layer parameters.
        """
        if num_categories <= 0:
            raise ValueError(
                "The number of categories for Categorical distribution must be positive"
            )
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.num_folds = num_folds
        self.num_categories = num_categories
        if not ((logits is None) ^ (probs is None)):
            raise ValueError("Exactly one between 'logits' and 'probs' must be specified")
        if logits is None:
            assert probs is not None
            if not self._valid_parameter_shape(probs):
                raise ValueError(f"The number of folds and shape of 'probs' must match the layer's")
        else:
            if not self._valid_parameter_shape(logits):
                raise ValueError(f"The number of folds and shape of 'probs' must match the layer's")
        self.probs = probs
        self.logits = logits

    def _valid_parameter_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return p.shape == (
            len(self.scope),
            self.num_output_units,
            self.num_channels,
            self.num_categories,
        )

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(num_categories=self.num_categories)
        return config

    @property
    def params(self) -> Dict[str, TorchParameter]:
        if self.logits is None:
            return dict(probs=self.probs)
        return dict(logits=self.logits)

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete
        x = F.one_hot(x, self.num_categories)  # (F, C, *B, D, num_categories)
        x = x.to(torch.get_default_dtype())
        logits = torch.log(self.probs()) if self.logits is None else self.logits()
        x = torch.einsum("fcbdi,fdkci->fbk", x, logits)
        return x


class TorchGaussianLayer(TorchExpFamilyLayer):
    """The Normal distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        mean: Optional[TorchParameter],
        stddev: Optional[TorchParameter],
        log_partition: Optional[TorchParameter] = None,
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            scope (Scope): The scope the input layer is defined on.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
            mean (AbstractTorchParameter): The reparameterization for layer parameters.
            stddev (AbstractTorchParameter): The reparameterization for layer parameters.
            num_folds (int): The number of channels. Defaults to 1.
        """
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        if not self._valid_parameters_shape(mean):
            raise ValueError(f"The number of folds and shape of 'mean' must match the layer's")
        if not self._valid_parameters_shape(stddev):
            raise ValueError(f"The number of folds and shape of 'stddev' must match the layer's")
        if log_partition is not None and not self._valid_parameters_shape(log_partition):
            raise ValueError(
                f"The number of folds and shape of 'log_partition' must match the layer's"
            )
        self.mean = mean
        self.stddev = stddev
        self.log_partition = log_partition

    def _valid_parameters_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return p.shape == (len(self.scope), self.num_output_units, self.num_channels)

    @property
    def params(self) -> Dict[str, TorchParameter]:
        params = dict(mean=self.mean, stddev=self.stddev)
        if self.log_partition is not None:
            params.update(log_partition=self.log_partition)
        return params

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        mean = self.mean().unsqueeze(dim=1)  # (F, 1, D, K, C)
        stddev = self.stddev().unsqueeze(dim=1)  # (F, 1, D, K, C)
        x = x.permute(0, 2, 3, 1).unsqueeze(dim=2)  # (F, B, D, 1, C)
        x = distributions.Normal(loc=mean, scale=stddev).log_prob(x)  # (F, B, D, K, C)
        x = torch.sum(x, dim=[2, 4])  # (F, B, K)
        if self.log_partition is not None:
            log_partition = self.log_partition()  # (F, D, K, C)
            x = x + torch.sum(log_partition, dim=[1, 3]).unsqueeze(dim=1)
        return x
