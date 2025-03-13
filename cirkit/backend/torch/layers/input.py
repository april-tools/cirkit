from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, distributions
from torch.distributions.utils import probs_to_logits

from cirkit.backend.torch.layers import TorchLayer
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import LSESumSemiring, Semiring, SumProductSemiring
from cirkit.utils.shape import comp_shape


class TorchInputLayer(TorchLayer, ABC):
    """The abstract base class for torch input layers."""

    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        semiring: Semiring | None = None,
    ) -> None:
        r"""Initialize a torch input layer.

        Args:
            scope_idx: A tensor of shape $(F, D)$, where $F$ is the number of folds, and
                $D$ is the number of variables on which the input layers in each fold are defined
                on. Alternatively, a tensor of shape $(D,)$ can be specified, which will be
                interpreted as a tensor of shape $(1, D)$, i.e., with $F = 1$.
            num_output_units: The number of output units.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the scope index is not a vector or a matrix.
        """
        if len(scope_idx.shape) == 1:
            scope_idx = scope_idx.unsqueeze(dim=0)
        elif len(scope_idx.shape) > 2:
            raise ValueError(f"The scope index must be a matrix, but found shape {scope_idx.shape}")
        num_folds, num_variables = scope_idx.shape
        super().__init__(
            num_variables,
            num_output_units,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.register_buffer("_scope_idx", scope_idx)

    @property
    def scope_idx(self) -> Tensor:
        """Retrieve the scope index tensor.

        Returns:
            The scope index tensor.
        """
        return self._scope_idx

    @property
    def num_variables(self) -> int:
        """The number of variables the input layer is defined on.

        Returns:
            The number of variables.
        """
        return self.num_input_units

    @property
    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        ...

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {}

    @property
    def fold_settings(self) -> tuple[Any, ...]:
        pshapes = [(n, p.shape) for n, p in self.params.items()]
        return self.num_variables, *self.config.items(), *pshapes

    def integrate(self) -> Tensor:
        r"""Integrate an input layer over all its variables' domain.

        Returns:
            Tensor: The tensor result of the integration, having shape $(F, K)$, where
                $F$ is the number of folds and $K$ is the number of output units.

        Raises:
            TypeError: If integration is not supported by the layer.
        """
        raise TypeError(f"Integration is not supported for layers of type {type(self)}")

    def sample(self, num_samples: int = 1) -> Tensor:
        r"""If the input layer encodes a probability distribution, then sample from it.

        Args:
            num_samples: The number of data points to sample.

        Returns:
            Tensor: The tensorized sample, having shape $(F, K, N)$, where
                $F$ is the number of folds, $K$ is the number of output units,
                and $N$ is the number of samples.

        Raises:
            TypeError: If sampling is not supported by the layer.
        """
        raise TypeError(f"Sampling is not supported for layers of type {type(self)}")

    def extra_repr(self) -> str:
        return (
            "  ".join(
                [
                    f"folds: {self.num_folds}",
                    f"variables: {self.num_variables}",
                    f"output-units: {self.num_output_units}",
                ]
            )
            + "\n"
            + f"input-shape: {(self.num_folds, self.arity, -1, self.num_input_units)}"
            + "\n"
            + f"output-shape: {(self.num_folds, -1, self.num_output_units)}"
        )


class TorchInputFunctionLayer(TorchInputLayer):
    """An input layer encoding functions defined over a non-empty set of variables."""

    def __call__(self, x: Tensor) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        r"""Invoke the forward function.

        Args:
            x: The tensor input to this layer, having shape $(F, B, D)$, where $F$
                is the number of folds, $B$ is the batch size, and $D$ is the number of variables.

        Returns:
            Tensor: The tensor output of this layer, having shape $(F, B, K)$, where $K$
                is the number of output units.
        """


class TorchConstantLayer(TorchInputLayer, ABC):
    """An input layer encoding a constant vector or, equivalently, a vector of functions
    defined over empty variable scopes.
    """

    def __init__(
        self,
        num_output_units: int,
        num_folds: int,
        *,
        semiring: Semiring | None = None,
    ) -> None:
        """Initializes a constant layer.

        Args:
            num_output_units: The number of output units.
            num_folds: The number of folds.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].
        """
        super().__init__(
            torch.empty(size=(num_folds, 0), dtype=torch.int64), num_output_units, semiring=semiring
        )

    def __call__(self, batch_size: int) -> Tensor:
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(batch_size)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, batch_size: int) -> Tensor:
        r"""Invoke the forward function.

        Args:
            batch_size: The batch size $B$ of the output tensor.

        Returns:
            Tensor: The tensor output of this layer, having shape $(F, B, K)$, where $K$
                is the number of output units, and $B$ is the batch size given as input.
        """


class TorchEmbeddingLayer(TorchInputFunctionLayer):
    r"""The embedding input layer, where each input function maps a discrete variable having
    finite support $\{0,\ldots,V-1\}$ to the corresponding entry of a $V$-th dimensional vector.
    """

    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_states: int = 2,
        weight: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        r"""Initialize an embedding input layer.

        Args:
            scope_idx: A tensor of shape $(F, D)$, where $F$ is the number of folds, and
                $D$ is the number of variables on which the input layers in each fold are defined
                on. Alternatively, a tensor of shape $(D,)$ can be specified, which will be
                interpreted as a tensor of shape $(1, D)$, i.e., with $F = 1$.
            num_output_units: The number of output units.
            num_states: The number of states $V$ each variable can assume.
            weight: The weight parameter of shape $(F, K, N)$, where $K$ is the number of output
                units, and $V$ is the number of states.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the scope contains more than one variable.
            ValueError: If the number of states $V$ is less than 2.
            ValueError: If the parameter's shape is incorrect.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Embedding layer encodes univariate functions")
        if num_states <= 1:
            raise ValueError("The number of states must be at least 2")
        super().__init__(
            scope_idx,
            num_output_units,
            semiring=semiring,
        )
        self.num_states = num_states
        if not self._valid_weight_shape(weight):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape {self._weight_shape} for 'weight', found"
                f"{weight.num_folds} and {weight.shape}, respectively"
            )
        self.weight = weight

    def _valid_weight_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return comp_shape(p.shape, self._weight_shape)

    @property
    def _weight_shape(self) -> tuple[int, ...]:
        return self.num_output_units, self.num_states

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_output_units": self.num_output_units,
            "num_states": self.num_states,
        }

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"weight": self.weight}

    def forward(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Embedding should be discrete
        x = x.squeeze(dim=2)  # (F, B)
        weight = self.weight()  # (F, B, K, N)

        # for each fold and batch retrieve the corresponding state
        idx_fold = torch.arange(self.num_folds, device=weight.device)
        idx_batch = torch.arange(weight.size(1), device=weight.device)
        x = weight[idx_fold[:, None], idx_batch, :, x]
        x = self.semiring.map_from(x, SumProductSemiring)
        return x  # (F, B, K)


class TorchExpFamilyLayer(TorchInputFunctionLayer, ABC):
    """The abstract base class for exponential family distribution layers.
    An input layer that is an exponential family distribution must define two methods.
    The first one is the ```log_unnormalized_likelihood```, used to compute the
    possibly-unnormalized log-likelihood. The second one is the ```log_partition_function```,
    used to compute the logarithm of the partition function."""

    def forward(self, x: Tensor) -> Tensor:
        x = self.log_unnormalized_likelihood(x)
        return self.semiring.map_from(x, LSESumSemiring)

    def integrate(self) -> Tensor:
        log_partition = self.log_partition_function()
        return self.semiring.map_from(log_partition, LSESumSemiring)

    @abstractmethod
    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        """Compute the (possibly unnormalized) log-likelihood of the given inputs.

        Args:
            x: The input tensor.

        Returns:
            Tensor: The (possibly unnormalized) log-likelihood as a tensor of shape $(F, K)$,
                where $F$ is the number of folds and $K$ is the number of output units.
        """

    @abstractmethod
    def log_partition_function(self) -> Tensor:
        """Compute the logarithm of the partition function of the layer.

        Returns:
            Tensor: The logarithm of the partition function as a tensor of shape $(F, K)$,
                where $F$ is the number of folds and $K$ is the number of output units.
                Note that it will be a tensor of zeros if the layer encodes already normalized
                exponential family distributions.
        """


class TorchCategoricalLayer(TorchExpFamilyLayer):
    """The Categorical distribution layer, parameterized by either probabilities or logits."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        num_categories: int = 2,
        probs: TorchParameter | None = None,
        logits: TorchParameter | None = None,
        semiring: Semiring | None = None,
    ) -> None:
        """Initialize a Categorical layer.

        Args:
            scope_idx: A tensor of shape $(F, D)$, where $F$ is the number of folds, and
                $D$ is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape $(D,)$ can be specified, which will be interpreted
                as a tensor of shape $(1, D)$, i.e., with $F = 1$.
            num_output_units: The number of output units.
            num_categories: The number of categories for Categorical distribution.
            probs: The probabilities parameter of shape $(F, K, N)$, where $K$ is the number of
                output units, and $V$ is the number of categories.
            logits: The logits parameter of shape $(F, K, N)$, where $K$ is the number of
                output units, and $V$ is the number of categories.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the scope contains more than one variable.
            ValueError: If the number of categories is negative.
            ValueError: If both the probs and logits parameters are provided, or none of them.
            ValueError: If the parameter's shape is incorrect.
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
            semiring=semiring,
        )
        self.num_categories = num_categories
        if not ((logits is None) ^ (probs is None)):
            raise ValueError("Exactly one between 'logits' and 'probs' must be specified")
        if logits is None:
            assert probs is not None
            if not self._valid_parameter_shape(probs):
                raise ValueError(
                    f"Expected number of folds {self.num_folds} "
                    f"and shape {self._probs_logits_shape} for 'probs', found "
                    f"{probs.num_folds} and {probs.shape}, respectively"
                )
        elif not self._valid_parameter_shape(logits):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape {self._probs_logits_shape} for 'logits', found "
                f"{logits.num_folds} and {logits.shape}, respectively"
            )
        self.probs = probs
        self.logits = logits

    def _valid_parameter_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return comp_shape(p.shape, self._probs_logits_shape)

    @property
    def _probs_logits_shape(self) -> tuple[int, ...]:
        return (-1, self.num_output_units, self.num_categories)

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_output_units": self.num_output_units,
            "num_categories": self.num_categories,
        }

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        if self.logits is None:
            return {"probs": self.probs}
        return {"logits": self.logits}

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Categorical should be discrete
        # x: (F, B, 1) -> (F, B)
        x = x.squeeze(dim=2)
        # logits: (F, B, K, N)
        logits = torch.log(self.probs()) if self.logits is None else self.logits()

        idx_fold = torch.arange(self.num_folds, device=logits.device)
        idx_batch = torch.arange(logits.size(1), device=logits.device)
        x = logits[idx_fold[:, None], idx_batch, :, x]
        return x  # (F, B, K)

    def log_partition_function(self) -> Tensor:
        if self.logits is None:
            return torch.zeros(
                size=(self.num_folds, 1, self.num_output_units), device=self.probs.device
            )
        logits = self.logits()
        return torch.sum(torch.logsumexp(logits, dim=3), dim=2).unsqueeze(dim=1)

    def sample(self, num_samples: int = 1) -> Tensor:
        logits = torch.log(self.probs()) if self.logits is None else self.logits()
        dist = distributions.Categorical(logits=logits)
        samples = dist.sample((num_samples,))  # (N, F, K)
        samples = samples.permute(1, 2, 0)  # (F, K, N)
        return samples


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
        total_count: int = 1,
        probs: TorchParameter | None = None,
        logits: TorchParameter | None = None,
        semiring: Semiring | None = None,
    ) -> None:
        r"""Initialize a Binomial layer.

        Args:
            scope_idx: A tensor of shape $(F, D)$, where $F$ is the number of folds, and
                $D$ is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape $(D,)$ can be specified, which will be interpreted
                as a tensor of shape $(1, D)$, i.e., with $F = 1$.
            num_output_units: The number of output units.
            total_count: The number of trials.
            probs: The probabilities parameter of shape $(F, K)$, where $K$ is the number of
                output units.
            logits: The logits parameter of shape $(F, K)$, where $K$ is the number of
                output units.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the scope contains more than one variable.
            ValueError: If the total count is not positive.
            ValueError: If both the probs and logits parameters are provided, or none of them.
            ValueError: If the parameter's shape is incorrect.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Binomial layer encodes a univariate distribution")
        if total_count < 0:
            raise ValueError("The number of trials must be non-negative")
        super().__init__(
            scope_idx,
            num_output_units,
            semiring=semiring,
        )
        self.total_count = total_count
        if not ((logits is None) ^ (probs is None)):
            raise ValueError("Exactly one between 'logits' and 'probs' must be specified")
        if logits is None:
            assert probs is not None
            if not self._valid_parameter_shape(probs):
                raise ValueError(
                    f"Expected number of folds {self.num_folds} "
                    f"and shape {self._probs_logits_shape} for 'probs', found"
                    f"{probs.num_folds} and {probs.shape}, respectively"
                )
        else:
            if not self._valid_parameter_shape(logits):
                raise ValueError(
                    f"Expected number of folds {self.num_folds} "
                    f"and shape {self._probs_logits_shape} for 'logits', found"
                    f"{logits.num_folds} and {logits.shape}, respectively"
                )
        self.probs = probs
        self.logits = logits

    def _valid_parameter_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return comp_shape(p.shape, self._probs_logits_shape)

    @property
    def _probs_logits_shape(self) -> tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_output_units": self.num_output_units,
            "total_count": self.total_count,
        }

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        if self.logits is None:
            return {"probs": self.probs}
        return {"logits": self.logits}

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        if x.is_floating_point():
            x = x.long()  # The input to Binomial should be discrete

        # (F, B, K)
        logits = (
            self.logits
            if self.logits is not None
            else probs_to_logits(self.probs(), is_binary=True)
        )
        dist = distributions.Binomial(self.total_count, logits=logits)
        return dist.log_prob(x)  # (F, B, K)

    def log_partition_function(self) -> Tensor:
        device = self.logits.device if self.logits is not None else self.probs.device
        return torch.zeros(size=(self.num_folds, 1, self.num_output_units), device=device)

    def sample(self, num_samples: int = 1) -> Tensor:
        logits = torch.log(self.probs()) if self.logits is None else self.logits()
        dist = distributions.Binomial(self.total_count, logits=logits)
        samples = dist.sample((num_samples,))  # (num_samples, F, K)
        samples = samples.permute(1, 2, 0)  # (F, K, num_samples)
        return samples


class TorchGaussianLayer(TorchExpFamilyLayer):
    """The Gaussian distribution layer. Optionally, this layer can encode unnormalized Gaussian
    distributions with the spefication of a log-partition function parameter."""

    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        mean: TorchParameter,
        stddev: TorchParameter,
        log_partition: TorchParameter | None = None,
        semiring: Semiring | None = None,
    ) -> None:
        r"""Initialize a Gaussian layer.

        Args:
            scope_idx: A tensor of shape $(F, D)$, where $F$ is the number of folds, and
                $D$ is the number of variables on which the input layers in each fold are defined on.
                Alternatively, a tensor of shape $(D,)$ can be specified, which will be interpreted
                as a tensor of shape $(1, D)$, i.e., with $F = 1$.
            num_output_units: The number of output units.
            mean: The mean parameter, having shape $(F, K)$, where $K$ is the number of
                output units.
            stddev: The standard deviation parameter, having shape $(F, K)$, where $K$ is the
                number of output units.
            log_partition: An optional parameter of shape $(F, K$, encoding the log-partition.
                function. If this is not None, then the Gaussian layer encodes unnormalized
                Gaussian likelihoods, which are then normalized with the given log-partition
                function.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the scope contains more than one variable.
            ValueError: If the mean and standard deviation parameter shapes are incorrect.
            ValueError: If the log-partition function parameter shape is incorrect.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Gaussian layer encodes a univariate distribution")
        super().__init__(
            scope_idx,
            num_output_units,
            semiring=semiring,
        )
        if not self._valid_mean_stddev_shape(mean):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape {self._mean_stddev_shape} for 'mean', found"
                f"{mean.num_folds} and {mean.shape}, respectively"
            )
        if not self._valid_mean_stddev_shape(stddev):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape {self._mean_stddev_shape} for 'stddev', found"
                f"{stddev.num_folds} and {stddev.shape}, respectively"
            )
        if log_partition is not None and not self._valid_log_partition_shape(log_partition):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape {self._log_partition_shape} for 'log_partition', found"
                f"{log_partition.num_folds} and {log_partition.shape}, respectively"
            )
        self.mean = mean
        self.stddev = stddev
        self.log_partition = log_partition

    def _valid_mean_stddev_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return comp_shape(p.shape, self._mean_stddev_shape)

    def _valid_log_partition_shape(self, log_partition: TorchParameter) -> bool:
        if log_partition.num_folds != self.num_folds:
            return False
        return comp_shape(log_partition.shape, self._log_partition_shape)

    @property
    def _mean_stddev_shape(self) -> tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def _log_partition_shape(self) -> tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def config(self) -> Mapping[str, Any]:
        return {"num_output_units": self.num_output_units}

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        params = {"mean": self.mean, "stddev": self.stddev}
        if self.log_partition is not None:
            params["log_partition"] = self.log_partition
        return params

    def log_unnormalized_likelihood(self, x: Tensor) -> Tensor:
        dist = distributions.Normal(loc=self.mean(), scale=self.stddev())

        x = dist.log_prob(x)  # (F, B, K)
        if self.log_partition is not None:
            x = x + self.log_partition()  # (F, B, K)
        return x

    def log_partition_function(self) -> Tensor:
        if self.log_partition is None:
            return torch.zeros(
                size=(self.num_folds, 1, self.num_output_units), device=self.mean.device
            )
        log_partition = self.log_partition()  # (F, K)
        return log_partition.unsqueeze(dim=1)  # (F, 1, K)

    def sample(self, num_samples: int = 1) -> Tensor:
        dist = distributions.Normal(loc=self.mean(), scale=self.stddev())
        samples = dist.sample((num_samples,))  # (N, F, K)
        samples = samples.permute(1, 2, 0)  # (F, K, N)
        return samples


class TorchConstantValueLayer(TorchConstantLayer):
    """An input layer having empty scope and computing a constant value."""

    def __init__(
        self,
        num_output_units: int,
        *,
        log_space: bool = False,
        value: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        r"""Initialize a constant value input layer.

        Args:
            num_output_units: The number of output units.
            log_space: Whether the given value is in the log-space, i.e., this constant
                layer should encode functions $\exp(x)$ rather than just x.
            value: The tensor value encoded by the layer, given by a parameter of shape $(F, K)$,
                where $F$ is the number of folds and $K$ is the numer of output units.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the number of folds of the shape of the given value is incorrect.
        """
        super().__init__(
            num_output_units,
            value.num_folds,
            semiring=semiring,
        )
        if value.num_folds != self.num_folds:
            raise ValueError(
                f"The value must have number of folds {self.num_folds}, "
                f"but found {value.num_folds}"
            )

        if not comp_shape(value.shape, (num_output_units,)):
            raise ValueError(
                f"The shape of the value must be (-1, {num_output_units}), "
                f"but found {value.shape}"
            )
        self.value = value
        self.log_space = log_space
        self._source_semiring = LSESumSemiring if log_space else SumProductSemiring

    @property
    def config(self) -> Mapping[str, Any]:
        return {"num_output_units": self.num_output_units, "log_space": self.log_space}

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"value": self.value}

    def forward(self, batch_size: int) -> Tensor:
        value = self.value()  # (F, B, Ko)

        if value.size(1) != 1 and value.size(1) != batch_size:
            raise ValueError(
                f"The batch size of the parameter ({value.size(1)})"
                f"does not match batch_size {batch_size}."
            )
        elif value.size(1) == 1 and batch_size != 1:
            # expand value to the requested batch size
            value = value.expand(-1, batch_size, *tuple(value.shape[2:]))

        return self.semiring.map_from(value, self._source_semiring)


class TorchEvidenceLayer(TorchConstantLayer):
    """The input layer computing the output of another input layer on a given observation."""

    def __init__(
        self,
        layer: TorchInputLayer,
        *,
        observation: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        r"""Initializes an evidence layer.

        Args:
            layer: The input layer on which compute the evidence of.
            observation: The observation, i.e., the input to pass to the given input layer.
                It must be a parameter of shape $(F, D)$, where $F$ is the number of folds
                of the given layer, $D$ is the number variables the given layer is defined on.
            semiring: The evaluation semiring.
                Defaults to [SumProductSemiring][cirkit.backend.torch.semiring.SumProductSemiring].

        Raises:
            ValueError: If the number of folds or the shape of the given observation is incorrect,
                with respect to the expected input shape of the given input layer.
        """
        if observation.num_folds != layer.num_folds:
            raise ValueError(
                f"The number of folds in the observation and in the layer should be the same, "
                f"but found {observation.num_folds} and {layer.num_folds} respectively"
            )
        if not comp_shape(observation.shape, (1,)):
            raise ValueError(
                f"Expected observation of shape (num_variables,), " f"but found {observation.shape}"
            )
        if observation.shape[-1] != layer.num_variables:
            raise ValueError(
                f"Expected an observation with number of variables {layer.num_variables}, "
                f"but found {observation.shape[-1]}"
            )
        super().__init__(layer.num_output_units, layer.num_folds, semiring=semiring)
        self.layer = layer
        self.observation = observation

    @property
    def config(self) -> Mapping[str, Any]:
        return {}

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"observation": self.observation}

    @property
    def sub_modules(self) -> Mapping[str, TorchInputLayer]:
        return {"layer": self.layer}

    def forward(self, batch_size: int) -> Tensor:
        obs = self.observation()  # (F, B, D)

        if obs.size(1) != 1 and obs.size(1) != batch_size:
            raise ValueError(
                f"The batch size of the observation ({obs.size(1)})"
                f"does not match batch_size {batch_size}."
            )

        x = self.layer(obs)

        if x.size(1) == 1:
            x = x.expand(x.size(0), batch_size, x.size(2))

        return x

    def sample(self, num_samples: int = 1) -> Tensor:
        if self.num_variables != 1:
            raise NotImplementedError("Sampling a multivariate Evidence layer is not implemented")
        # Sampling an evidence layer translates to return the given observation
        obs = self.observation()  # (F, D=1)
        obs = obs.unsqueeze(dim=-1)  # (F, 1, 1)
        return obs.expand(size=(-1, -1, self.num_output_units, num_samples))


class TorchPolynomialLayer(TorchInputFunctionLayer):
    """The polynomial input layer, evaluating a vector of parameterized polynomials."""

    def __init__(
        self,
        scope_idx: Tensor,
        num_output_units: int,
        *,
        degree: int,
        coeff: TorchParameter,
        semiring: Semiring | None = None,
    ) -> None:
        r"""Initialize a polynomial layer.

        Args:
            scope_idx: A tensor of shape $(F, D)$, where $F$ is the number of folds, and
                $D$ is the number of variables on which the input layers in each fold are defined
                on. Alternatively, a tensor of shape $(D,)$ can be specified, which will be
                interpreted as a tensor of shape $(1, D)$, i.e., with $F = 1$.
            num_output_units: The number of output units.
            degree: The degree of polynomial.
            coeff: The coefficient parameter, having shape $(F, K, \mathsf{degree} + 1)$, where $K$ is the number
                of output units.

        Raises:
            ValueError: If the scope contains more than one variable.
            ValueError: If the coefficients is not correct.
        """
        num_variables = scope_idx.shape[-1]
        if num_variables != 1:
            raise ValueError("The Polynomial layer encodes a univariate distribution")
        super().__init__(
            scope_idx,
            num_output_units,
            semiring=semiring,
        )
        self.degree = degree
        if not self._valid_parameters_shape(coeff):
            raise ValueError(
                f"Expected number of folds {self.num_folds} "
                f"and shape {self._coeff_shape} for 'coeff', found"
                f"{coeff.num_folds} and {coeff.shape}, respectively"
            )
        self.coeff = coeff

    def _valid_parameters_shape(self, p: TorchParameter) -> bool:
        if p.num_folds != self.num_folds:
            return False
        return p.shape == self._coeff_shape

    @property
    def _coeff_shape(self) -> tuple[int, ...]:
        return self.num_output_units, self.degree + 1

    @staticmethod
    def _polyval(coeff: Tensor, x: Tensor) -> Tensor:
        r"""Evaluate polynomial given coefficients and point, with the shape for PolynomialLayer.

        Args:
            coeff: The coefficients of the polynomial, shape $(F, K_o, \mathsf{degree} + 1)$.
            x: The point of the variable, shape $(F, H, B, K_i)$, where $H=K_i=1$.

        Returns:
            Tensor: The value of the polymonial, shape $(F, B, K_o)$.
        """
        x = x.squeeze(dim=1)  # shape (F, H=1, B, Ki=1) -> (F, B, 1).
        y = x.new_zeros(*x.shape[:-1], coeff.shape[-2])  # shape (F, B, Ko).

        # TODO: iterating over dim=2 is inefficient
        for a_n in reversed(
            coeff.unbind(dim=2)
        ):  # Reverse iterator of the degree axis, shape (F, Ko).
            # a_n shape (F, Ko) -> (F, 1, Ko).
            y = torch.addcmul(a_n.unsqueeze(dim=1), x, y)  # y = a_n + x * y, by Horner's method.
        return y  # shape (F, B, Ko).

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_output_units": self.num_output_units,
            "degree": self.degree,
        }

    @property
    def params(self) -> Mapping[str, TorchParameter]:
        return {"coeff": self.coeff}

    def forward(self, x: Tensor) -> Tensor:
        coeff = self.coeff()  # shape (F, Ko, dp1)
        return self.semiring.map_from(TorchPolynomialLayer._polyval(coeff, x), SumProductSemiring)
