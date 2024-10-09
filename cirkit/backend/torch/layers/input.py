from abc import ABC, abstractmethod
from typing import Any, Optional

import einops as E
import torch
from torch import Tensor, distributions
from torch.nn import functional as F

from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import LSESumSemiring, Semiring, SumProductSemiring


class TorchInputLayer(TorchLayer, ABC):
    """The abstract base class for input layers."""

    # NOTE: We use exactly the safe interface (F, H, B, K) -> (F, B, K) for __call__ of input layers:
    #           1. Define arity(H)=num_channels(C), reusing the H dimension.
    #           2. Define num_input_units(K)=num_vars(D), which reuses the K dimension.
    #       For dimension D (variables), we should parse the input in circuit according to the
    #       scope of the corresponding region node/symbolic input layer.

    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_channels: int = 1,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            scope_idx: A tensor of shape (F, D), where F is the number of folds, and
                D is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape (D,) can be specified, which will be interpreted
                as a tensor of shape (1, D), i.e., with F = 1.
            num_output_units: The number of output units.
            num_channels: The number of channels. Defaults to 1.
        """
        if len(scope_idx.shape) == 1:
            scope_idx = scope_idx.unsqueeze(dim=0)
        elif len(scope_idx.shape) > 2:
            raise ValueError(f"The scope index must be a matrix, but found shape {scope_idx.shape}")
        num_folds, num_variables = scope_idx.shape
        super().__init__(
            num_variables,
            num_output_units,
            arity=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.register_buffer("_scope_idx", scope_idx)

    @property
    def scope_idx(self) -> Tensor:
        return self._scope_idx

    @property
    def num_variables(self) -> int:
        return self.num_input_units

    @property
    def num_channels(self) -> int:
        return self.arity

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return self.num_variables, self.num_channels, self.num_output_units

    @property
    def config(self) -> dict[str, Any]:
        return {
            "scope_idx": self.scope_idx,
            "num_output_units": self.num_output_units,
            "num_channels": self.num_channels,
        }

    @property
    def params(self) -> dict[str, TorchParameter]:
        return {}

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def integrate(self) -> Tensor:
        ...

    @abstractmethod
    def sample(self, num_samples: int = 1, x: Tensor | None = None) -> Tensor:
        ...

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
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_channels: int = 1,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            scope_idx: A tensor of shape (F, D), where F is the number of folds, and
                D is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape (D,) can be specified, which will be interpreted
                as a tensor of shape (1, D), i.e., with F = 1.
            num_output_units: The number of output units.
            num_channels: The number of channels. Defaults to 1.
        """
        super().__init__(
            scope_idx,
            num_output_units,
            num_channels=num_channels,
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

    def integrate(self) -> Tensor:
        log_partition = self.log_partition_function()
        return self.semiring.map_from(log_partition, LSESumSemiring)

    @abstractmethod
    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def log_partition_function(self) -> Tensor:
        ...


class TorchCategoricalLayer(TorchExpFamilyLayer):
    """The Categorical distribution layer.

    This is fully factorized down to univariate Categorical distributions.
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_categories: int = 2,
        probs: TorchParameter | None = None,
        logits: TorchParameter | None = None,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            scope_idx: A tensor of shape (F, D), where F is the number of folds, and
                D is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape (D,) can be specified, which will be interpreted
                as a tensor of shape (1, D), i.e., with F = 1.
            num_output_units: The number of output units.
            num_channels: The number of channels.
            num_categories: The number of categories for Categorical distribution. Defaults to 2.
            probs: The reparameterization for layer probs parameters.
            logits: The reparameterization for layer logits parameters.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Gaussian layer encodes a univariate distribution")
        if num_categories <= 0:
            raise ValueError(
                "The number of categories for Categorical distribution must be positive"
            )
        super().__init__(
            scope_idx,
            num_output_units,
            num_channels=num_channels,
            semiring=semiring,
        )
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
    def config(self) -> dict[str, Any]:
        config = super().config
        config.update(num_categories=self.num_categories)
        return config

    @property
    def params(self) -> dict[str, TorchParameter]:
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

    def log_partition_function(self) -> Tensor:
        if self.logits is None:
            return torch.zeros(
                size=(self.num_folds, 1, self.num_output_units), device=self.probs.device
            )
        logits = self.logits()
        return torch.sum(torch.logsumexp(logits, dim=3), dim=2).unsqueeze(dim=1)

    def sample(self, num_samples: int = 1, x: Tensor | None = None) -> Tensor:
        raise TypeError("Sampling is not implemented for Categorical layers")


class TorchBinomialLayer(TorchExpFamilyLayer):
    """The Binomial distribution layer.

    This is fully factorized down to univariate Binomial distributions.
    """

    # DISABLE: It's designed to have these arguments.
    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_channels: int = 1,
        total_count: int = 1,
        probs: Optional[TorchParameter] = None,
        logits: Optional[TorchParameter] = None,
        semiring: Optional[Semiring] = None,
    ) -> None:
        """Init class.

        Args:
            scope_idx: A tensor of shape (F, D), where F is the number of folds, and
                D is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape (D,) can be specified, which will be interpreted
                as a tensor of shape (1, D), i.e., with F = 1.
            num_output_units: The number of output units.
            num_channels: The number of channels.
            total_count: The number of trails. Defaults to 1.
            probs: The reparameterization for layer probs parameters.
            logits: The reparameterization for layer logits parameters.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Binomial layer encodes a univariate distribution")
        if total_count < 0:
            raise ValueError("The number of trials must be non-negative")
        super().__init__(
            scope_idx,
            num_output_units,
            num_channels=num_channels,
            semiring=semiring,
        )
        self.total_count = total_count
        if not ((logits is None) ^ (probs is None)):
            raise ValueError("Exactly one between 'logits' and 'probs' must be specified")
        if logits is None:
            assert probs is not None
            if not self._valid_parameter_shape(probs):
                raise ValueError(f"The number of folds and shape of 'probs' must match the layer's")
        else:
            if not self._valid_parameter_shape(logits):
                raise ValueError(
                    f"The number of folds and shape of 'logits' must match the layer's"
                )
        self.probs = probs
        self.logits = logits

    def _valid_parameter_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return p.shape == (
            self.num_output_units,
            self.num_channels,
        )

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config.update(total_count=self.total_count)
        return config

    @property
    def params(self) -> dict[str, TorchParameter]:
        if self.logits is None:
            return dict(probs=self.probs)
        return dict(logits=self.logits)

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete
        if self.logits is not None:
            dist = distributions.Binomial(self.total_count, logits=self.logits())
        else:
            dist = distributions.Binomial(self.total_count, probs=self.probs())
        x = dist.log_prob(x.transpose(1, 2)).sum(-1)
        return x

    def log_partition_function(self) -> Tensor:
        if self.logits is None:
            return torch.zeros(
                size=(self.num_folds, 1, self.num_output_units), device=self.probs.device
            )
        logits = self.logits()
        return torch.sum(torch.logsumexp(logits, dim=3), dim=2).unsqueeze(dim=1)

    def sample(self, num_samples: int = 1, x: Tensor | None = None) -> Tensor:
        logits = torch.log(self.probs()) if self.logits is None else self.logits()
        distribution = distributions.Binomial(self.total_count, logits=logits)
        samples = distribution.sample((num_samples,))  # (N, F, K, C)
        samples = E.rearrange(samples, "n f k c -> f c k n")  # (F, C, K, N)
        return samples


class TorchGaussianLayer(TorchExpFamilyLayer):
    """The Normal distribution layer.

    This is fully factorized down to univariate Gaussian distributions.
    """

    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_channels: int = 1,
        mean: TorchParameter,
        stddev: TorchParameter,
        log_partition: TorchParameter | None = None,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            scope_idx: A tensor of shape (F, D), where F is the number of folds, and
                D is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape (D,) can be specified, which will be interpreted
                as a tensor of shape (1, D), i.e., with F = 1.
            num_output_units: The number of output units.
            num_channels: The number of channels. Defaults to 1.
            mean (AbstractTorchParameter): The reparameterization for layer parameters.
            stddev (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Gaussian layer encodes a univariate distribution")
        super().__init__(
            scope_idx,
            num_output_units,
            num_channels=num_channels,
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
    def params(self) -> dict[str, TorchParameter]:
        params = dict(mean=self.mean, stddev=self.stddev)
        if self.log_partition is not None:
            params.update(log_partition=self.log_partition)
        return params

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        mean = self.mean().unsqueeze(dim=1)  # (F, 1, K, C)
        stddev = self.stddev().unsqueeze(dim=1)  # (F, 1, K, C)
        x = x.permute(0, 2, 3, 1)  # (F, C, B, 1) -> (F, B, 1, C)
        x = distributions.Normal(loc=mean, scale=stddev).log_prob(x)  # (F, B, K, C)
        x = torch.sum(x, dim=3)  # (F, B, K)
        if self.log_partition is not None:
            log_partition = self.log_partition()  # (F, K, C)
            x = x + torch.sum(log_partition, dim=2).unsqueeze(dim=1)
        return x

    def log_partition_function(self) -> Tensor:
        if self.log_partition is None:
            return torch.zeros(
                size=(self.num_folds, 1, self.num_output_units), device=self.mean.device
            )
        log_partition = self.log_partition()  # (F, K, C)
        return torch.sum(log_partition, dim=2).unsqueeze(dim=1)

    def sample(self, num_samples: int = 1, x: Tensor | None = None) -> Tensor:
        raise TypeError("Sampling is not implemented for Gaussian layers")


class TorchLogPartitionLayer(TorchInputLayer):
    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_channels: int = 1,
        num_folds: int = 1,
        value: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            scope_idx: A tensor of shape (F, D), where F is the number of folds, and
                D is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape (D,) can be specified, which will be interpreted
                as a tensor of shape (1, D), i.e., with F = 1.
            num_output_units: The number of output units.
            num_channels: The number of channels. Defaults to 1.
            value (Optional[Reparameterization], optional): Ignored. This layer has no parameters.
        """
        super().__init__(
            scope_idx,
            num_output_units,
            num_channels=num_channels,
            semiring=semiring,
        )
        assert value.num_folds == self.num_folds
        assert value.shape == (num_output_units,)
        self.value = value

    @property
    def params(self) -> dict[str, TorchParameter]:
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

    def integrate(self) -> Tensor:
        raise TypeError("Cannot integrate a layer computing a log-partition function")

    def sample(self, num_samples: int = 1, x: Tensor | None = None) -> Tensor:
        raise TypeError("Cannot sample from a layer computing a log-partition function")


# TODO: could be in backends/torch/utils, can be reused by PolyGaussian
def polyval(coeff: Tensor, x: Tensor) -> Tensor:
    """Evaluate polynomial given coefficients and point, with the shape for PolynomialLayer.

    Args:
        coeff (Tensor): The coefficients of the polynomial, shape (F, Ko, deg+1).
        x (Tensor): The point of the variable, shape (F, H, B, Ki), where H=Ki=1.

    Returns:
        Tensor: The value of the polymonial, shape (F, B, Ko).
    """
    x = x.squeeze(dim=1)  # shape (F, H=1, B, Ki=1) -> (F, B, 1).
    y = x.new_zeros(*x.shape[:-1], coeff.shape[-2])  # shape (F, B, Ko).

    # TODO: iterating over dim=2 is inefficient
    for a_n in reversed(coeff.unbind(dim=2)):  # Reverse iterator of the degree axis, shape (F, Ko).
        # a_n shape (F, Ko) -> (F, 1, Ko).
        y = torch.addcmul(a_n.unsqueeze(dim=1), x, y)  # y = a_n + x * y, by Horner's method.
    return y  # shape (F, B, Ko).


class TorchPolynomialLayer(TorchInputLayer):
    """The Polynomial layer.

    This is fully factorized down to univariate polynomials w.r.t. channels.
    """

    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_channels: int = 1,
        degree: int,
        coeff: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        """Init class.

        Args:
            scope_idx: A tensor of shape (F, D), where F is the number of folds, and
                D is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape (D,) can be specified, which will be interpreted
                as a tensor of shape (1, D), i.e., with F = 1.
            num_output_units (int): The number of output units.
            num_channels (int): The number of channels. Defaults to 1.
            degree (int): The degree of polynomial.
            coeff (AbstractTorchParameter): The reparameterization for layer parameters.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Polynomial layer encodes a univariate distribution")
        if num_channels != 1:
            raise ValueError("The Polynomial layer encodes a univariate distribution")
        super().__init__(
            scope_idx,
            num_output_units,
            num_channels=num_channels,
            semiring=semiring,
        )
        self.degree = degree
        if not self._valid_parameters_shape(coeff):
            raise ValueError("The number of folds and shape of 'coeff' must match the layer's")
        self.coeff = coeff

    def _valid_parameters_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return p.shape == (self.num_output_units, self.degree + 1)

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        return *super().fold_settings, self.degree + 1

    @property
    def config(self) -> dict[str, Any]:
        return {**super().config, "degree": self.degree}

    @property
    def params(self) -> dict[str, TorchParameter]:
        return {"coeff": self.coeff}

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H=C, B, Ki=D).

        Returns:
            Tensor: The output of this layer, shape (F, B, Ko).
        """
        coeff = self.coeff()  # shape (F, Ko, dp1)
        return self.semiring.map_from(polyval(coeff, x), SumProductSemiring)

    def integrate(self) -> Tensor:
        raise TypeError("Cannot integrate a Polynomial layer")

    def sample(self, num_samples: int = 1, x: Tensor | None = None) -> Tensor:
        raise TypeError("Cannot sample from a Polynomial layer")
