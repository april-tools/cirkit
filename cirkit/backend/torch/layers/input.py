from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, distributions
from torch.nn import functional as F

from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import LSESumSemiring, Semiring
from cirkit.utils.scope import Scope


class TorchInputLayer(TorchLayer, ABC):
    """The abstract base class for input layers."""

    # NOTE: We use exactly the sae interface (F, H, B, K) -> (F, B, K) for __call__ of input layers:
    #           1. Define arity(H)=num_channels(C), reusing the H dimension.
    #           2. Define num_input_units(K)=num_vars(D), which reuses the K dimension.
    #       For dimension D (variables), we should parse the input in circuit according to the
    #       scope of the corresponding region node/symbolic input layer.

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
            len(scope), num_output_units, arity=num_channels, num_folds=num_folds, semiring=semiring
        )
        self.scope = scope

    @property
    def num_variables(self) -> int:
        return self.num_input_units

    @property
    def num_channels(self) -> int:
        return self.arity

    @property
    def fold_settings(self) -> Tuple[Any, ...]:
        return self.num_variables, self.num_channels, self.num_output_units

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "num_channels": self.num_channels,
            "num_folds": self.num_folds,
        }

    @property
    def params(self) -> Dict[str, TorchParameter]:
        return {}

    def extra_repr(self) -> str:
        return (
            "  ".join(
                [
                    f"folds: {self.num_folds}",
                    f"channels: {self.num_channels}",
                    f"variables: {self.num_variables}",
                    f"output-units: {self.num_output_units}",
                ]
            )
            + "\n"
            + f"input-shape: {(self.num_folds, self.arity, -1, self.num_input_units)}"
            + "\n"
            + f"output-shape: {(self.num_folds, -1, self.num_output_units)}"
        )


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
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        x = self.log_unnormalized_likelihood(x)
        return self.semiring.map_from(x, LSESumSemiring)

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
        if len(scope) != 1:
            raise ValueError("The Gaussian layer encodes a univariate distribution")
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
        x = F.one_hot(x, self.num_categories)  # (F, C, B, 1, num_categories)
        x = x.squeeze(dim=3)  # (F, C, B, num_categories)
        x = x.to(torch.get_default_dtype())
        logits = torch.log(self.probs()) if self.logits is None else self.logits()
        x = torch.einsum("fcbi,fkci->fbk", x, logits)
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
        mean: TorchParameter,
        stddev: TorchParameter,
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
        if len(scope) != 1:
            raise ValueError("The Gaussian layer encodes a univariate distribution")
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
        return p.shape == (self.num_output_units, self.num_channels)

    @property
    def params(self) -> Dict[str, TorchParameter]:
        params = dict(mean=self.mean, stddev=self.stddev)
        if self.log_partition is not None:
            params.update(log_partition=self.log_partition)
        return params

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        mean = self.mean().unsqueeze(dim=1)  # (F, 1, K, C)
        stddev = self.stddev().unsqueeze(dim=1)  # (F, 1, K, C)
        x = x.permute(0, 2, 3, 1)  # (F, C, B, 1) -> (F, B, 1, C)
        x = distributions.Normal(loc=mean, scale=stddev).log_prob(x)  # (F, B, K, C)
        x = torch.sum(x, dim=-1)  # (F, B, K)
        if self.log_partition is not None:
            log_partition = self.log_partition()  # (F, K, C)
            x = x + torch.sum(log_partition, dim=-1).unsqueeze(dim=1)
        return x


class TorchLogPartitionLayer(TorchInputLayer):
    """The constant input layer, with no parameters."""

    # We still accept any Reparameterization instance for reparam, but it will be ignored.
    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        value: TorchParameter,
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            scope (Scope): The scope of the layer.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
            num_folds (int): The number of channels. Defaults to 1.
            value (Optional[Reparameterization], optional): Ignored. This layer has no parameters.
        """
        assert value.num_folds == num_folds
        assert value.shape == (num_output_units,)
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.value = value

    @property
    def params(self) -> Dict[str, TorchParameter]:
        params = super().params
        params.update(value=self.value)
        return params

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, B, Ki).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        value = self.value().unsqueeze(dim=1)  # (F, 1, Ko)
        # (F, Ko) -> (F, B, O)
        value = value.expand(value.shape[0], x.shape[2], value.shape[2])
        return self.semiring.map_from(value, LSESumSemiring)
