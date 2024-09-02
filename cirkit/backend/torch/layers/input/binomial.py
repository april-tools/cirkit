import torch
import einops as E

from typing import Any, Dict, Optional, Tuple
from torch import Tensor, distributions
from cirkit.backend.torch.parameters.parameter import TorchParameter
from cirkit.backend.torch.semiring import Semiring
from cirkit.backend.torch.layers import TorchExpFamilyLayer
from cirkit.utils.scope import Scope
from cirkit.symbolic.parameters import TensorParameter, Parameter, ScaledSigmoidParameter
from cirkit.symbolic.layers import InputLayer, ParameterFactory
from cirkit.symbolic.initializers import NormalInitializer
from cirkit.backend.torch.compiler import TorchCompiler


class BinomialLayer(InputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        total_count: int,
        logits: Optional[Parameter] = None,
        probs: Optional[Parameter] = None,
        logits_factory: Optional[ParameterFactory] = None,
        probs_factory: Optional[ParameterFactory] = None,
    ):
        if logits is not None and probs is not None:
            raise ValueError("At most one between 'logits' and 'probs' can be specified")
        if logits_factory is not None and probs_factory is not None:
            raise ValueError(
                "At most one between 'logits_factory' and 'probs_factory' can be specified"
            )
        if total_count < 0:
            raise ValueError("The number of trials should be non negative")
        super().__init__(scope, num_output_units, num_channels)
        self.total_count = total_count
        if logits is None and probs is None:
            if logits_factory is not None:
                logits = logits_factory(self.probs_logits_shape)
            elif probs_factory is not None:
                probs = probs_factory(self.probs_logits_shape)
            else:
                logits = Parameter.from_leaf(
                    TensorParameter(*self.probs_logits_shape, initializer=NormalInitializer())
                )
        if logits is not None and logits.shape != self.probs_logits_shape:
            raise ValueError(
                f"Expected parameter shape {self.probs_logits_shape}, found {logits.shape}"
            )
        if probs is not None and probs.shape != self.probs_logits_shape:
            raise ValueError(
                f"Expected parameter shape {self.probs_logits_shape}, found {probs.shape}"
            )
        self.probs = probs
        self.logits = logits

    @property
    def probs_logits_shape(self) -> Tuple[int, ...]:
        return self.num_variables, self.num_output_units, self.num_channels

    @property
    def config(self) -> dict:
        config = super().config
        config.update(total_count=self.total_count)
        return config

    @property
    def params(self) -> Dict[str, Parameter]:
        if self.logits is None:
            return dict(probs=self.probs)
        return dict(logits=self.logits)


class TorchBinomialLayer(TorchExpFamilyLayer):
    """The Binomial distribution layer.

    This is fully factorized down to univariate Binomial distributions.
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
        total_count: int = 1,
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
            total_count (int): The number of trails. Defaults to 1.
            logits (TorchParameter): The reparameterization for layer parameters.
        """
        if total_count < 0:
            raise ValueError("The number of trials must be non-negative")
        super().__init__(
            scope,
            num_output_units,
            num_channels=num_channels,
            num_folds=num_folds,
            semiring=semiring,
        )
        self.num_folds = num_folds
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
            len(self.scope),
            self.num_output_units,
            self.num_channels,
        )

    @property
    def config(self) -> Dict[str, Any]:
        config = super().config
        config.update(total_count=self.total_count)
        return config

    @property
    def params(self) -> Dict[str, TorchParameter]:
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

    def sample_forward(self, num_samples: int, x: Optional[Tensor] = None) -> Tensor:
        if len(self.scope) > 1:
            raise NotImplementedError("Multivariate Binomial sampling is not implemented yet!")

        logits = torch.log(self.probs()) if self.logits is None else self.logits()
        distribution = distributions.Binomial(self.total_count, logits=logits)

        samples = distribution.sample((num_samples,))  # (N, F, D = 1, K, C)
        samples = E.rearrange(samples[..., 0, :, :], "n f k c -> f c k n")  # (F, C, K, N)
        return samples

    def extended_forward(self, x: Tensor) -> Tensor:
        return self.log_unnormalized_likelihood(x)


def compile_binomial_layer(compiler: "TorchCompiler", sl: BinomialLayer) -> TorchBinomialLayer:
    if sl.logits is None:
        probs = compiler.compile_parameter(sl.probs)
        logits = None
    else:
        probs = None
        logits = compiler.compile_parameter(sl.logits)
    return TorchBinomialLayer(
        sl.scope,
        sl.num_output_units,
        num_channels=sl.num_channels,
        total_count=sl.total_count,
        probs=probs,
        logits=logits,
        semiring=compiler.semiring,
    )


def binomial_layer_factory(scope: Scope, num_units: int, num_channels: int) -> BinomialLayer:
    return BinomialLayer(
        scope,
        num_units,
        num_channels,
        total_count=255,
        probs_factory=lambda shape: Parameter.from_sequence(
            TensorParameter(*shape, initializer=NormalInitializer(0.0, 1.0)),
            ScaledSigmoidParameter(shape, vmin=1e-5, vmax=1.0),
        ),
    )
