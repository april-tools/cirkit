from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, distributions, nn
from torch.nn import functional as F

from cirkit.backend.torch.layers.input.base import TorchInputLayer
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.semiring import SemiringCls
from cirkit.backend.torch.utils import InitializerFunc
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
        semiring: Optional[SemiringCls] = None,
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
        return self.semiring.from_lse_sum(self.log_score(x))

    @abstractmethod
    def log_probs(self, x: Tensor) -> Tensor:
        ...

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
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        num_categories: int = 2,
        logits: AbstractTorchParameter,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            scope (Scope): The scope the input layer is defined on.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels.
            num_folds (int): The number of channels. Defaults to 1.
            num_categories (int): The number of categories for Categorical distribution. Defaults to 2.
            logits (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        assert (
            num_categories > 0
        ), "The number of categories for Categorical distribution must be positive."
        assert logits.num_folds == num_folds
        assert logits.shape == (
            len(scope),
            num_output_units,
            num_channels,
            num_categories,
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
        self.logits = logits

    @classmethod
    def default_initializers(cls) -> Dict[str, InitializerFunc]:
        return dict(logits=lambda t: nn.init.normal_(t, mean=0.0, std=1e-1))

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(num_categories=self.num_categories)
        return config

    @property
    def params(self) -> Dict[str, AbstractTorchParameter]:
        params = super().params
        params.update(logits=self.logits)
        return params

    def _eval_forward(self, x: Tensor, logits: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete.
        x = F.one_hot(x, self.num_categories)  # (F, C, *B, D, num_categories)
        x = x.to(torch.get_default_dtype())
        x = torch.einsum("fcbdi,fdkci->fbk", x, logits)
        return x

    def log_probs(self, x: Tensor) -> Tensor:
        logits = self.logits()
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
        return self._eval_forward(x, logits) - log_z

    def log_score(self, x: Tensor) -> Tensor:
        logits = self.logits()
        return self._eval_forward(x, logits)


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
        mean: AbstractTorchParameter = None,
        stddev: AbstractTorchParameter = None,
        # mean: Optional[AbstractTorchParameter] = None,
        # stddev: Optional[AbstractTorchParameter] = None,
        log_partition: Optional[AbstractTorchParameter] = None,
        # params: Optional[TorchMultiParameters] = None,
        semiring: Optional[SemiringCls] = None,
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
        # assert params is not None or (mean is not None and stddev is not None)
        mean_stddev_shape = num_folds, len(scope), num_output_units, num_channels
        # if mean is not None and stddev is not None:
        #     assert params is None
        #     assert mean.shape == stddev.shape == mean_stddev_shape
        #     if log_partition is not None:
        #         assert log_partition.shape == (num_output_units,)
        # else:
        #     assert mean is None and stddev is None and log_partition is None
        #     assert params.shape("mean") == params.shape("stddev") == mean_stddev_shape
        #     assert params.shape("log_partition") == (num_output_units,)
        #     mean = TorchSelectorParameter(params, "mean")
        #     stddev = TorchSelectorParameter(params, "stddev")
        #     log_partition = TorchSelectorParameter(params, "log_partition")
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.mean = mean
        self.stddev = stddev
        self.log_partition = log_partition
        # self.params = params

    @classmethod
    def default_initializers(cls) -> Dict[str, InitializerFunc]:
        return dict(
            mean=lambda t: nn.init.normal_(t, mean=0.0, std=3e-1),
            stddev=lambda t: nn.init.uniform_(t, a=np.log(1e-2), b=0.0).exp_(),
        )

    @property
    def params(self) -> Dict[str, AbstractTorchParameter]:
        params = super().params
        params.update(mean=self.mean, stddev=self.stddev, log_partition=self.log_partition)
        return params

    def _get_parameters(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        # if self.params is None:
        #     log_partition = self.log_partition() if self.log_partition else None
        #     return self.mean(), self.stddev(), log_partition
        # mstd = self.params()
        # return mstd["mean"], mstd["stddev"], mstd["log_partition"]
        log_partition = self.log_partition() if self.log_partition else None
        return self.mean(), self.stddev(), log_partition

    def _eval_forward(self, x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        # x: (F, C, B, D)
        # dist_loc: (F, D, K, C)
        # dist_stddev: (F, D, K, C)
        dist = distributions.Normal(loc=loc, scale=scale)
        x = x.permute(0, 2, 3, 1).unsqueeze(dim=2)  # (F, *B, D, 1, C)
        x = dist.log_prob(x)
        x = torch.sum(x, dim=[2, 4])
        # TODO (LL):
        #  the current tensor shapes might enable faster inference,
        #  but it is honestly a mess to work with.
        #  E.g., they require frequent permutations, squeezes/unsqueezes,
        #        and reductions over obscure combinations of axes for which we
        #        do not know if they will be time/memory efficient in the end
        return x

    def log_probs(self, x: Tensor) -> Tensor:
        mean, stddev, _ = self._get_parameters()
        x = self._eval_forward(x, mean, stddev)
        return x

    def log_score(self, x: Tensor) -> Tensor:
        mean, stddev, log_partition = self._get_parameters()
        x = self._eval_forward(x, mean, stddev)
        if log_partition is not None:
            x = x + log_partition
        return x
