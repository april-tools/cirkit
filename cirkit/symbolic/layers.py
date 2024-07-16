from abc import ABC
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, cast

from cirkit.symbolic.initializers import Initializer, NormalInitializer
from cirkit.symbolic.parameters import (
    Parameter,
    Parameterization,
    ScaledSigmoidParameter,
    TensorParameter,
)
from cirkit.utils.scope import Scope

AbstractLayerOperator = IntEnum  # TODO: switch to StrEnum (>=py3.11) or better alternative


class LayerOperation(AbstractLayerOperator):
    """Types of Symbolic operations on layers."""

    def _generate_next_value_(self, start: int, count: int, last_values: list) -> int:
        return -(
            count + 1
        )  # Enumerate negative integers as the user can extend them with non-negative ones

    INTEGRATION = auto()
    DIFFERENTIATION = auto()
    MULTIPLICATION = auto()
    CONJUGATION = auto()


class Layer(ABC):
    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
    ):
        self.scope = scope
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

    @property
    def params(self) -> Dict[str, Parameter]:
        return {}


class InputLayer(Layer):
    def __init__(self, scope: Scope, num_output_units: int, num_channels: int = 1):
        super().__init__(scope, len(scope), num_output_units, num_channels)

    @property
    def num_variables(self) -> int:
        return self.num_input_units

    @property
    def num_channels(self) -> int:
        return self.arity

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "num_channels": self.num_channels,
        }


class CategoricalLayer(InputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        num_categories: int = 2,
        logits: Optional[Parameter] = None,
        probs: Optional[Parameter] = None,
        parameterization: Optional[Parameterization] = None,
        initializer: Optional[Initializer] = None,
    ):
        if logits is not None and probs is not None:
            raise ValueError("At most one between 'logits' and 'probs' can be specified")
        super().__init__(scope, num_output_units, num_channels)
        self.num_categories = num_categories
        if logits is None and probs is None:
            if initializer is None:
                initializer = NormalInitializer()
            logits = TensorParameter(
                len(scope),
                num_output_units,
                num_channels,
                num_categories,
                initializer=initializer,
            )
            if parameterization is None:
                logits = Parameter.from_leaf(logits)
            else:
                logits = parameterization(logits)
        elif logits is not None:
            assert logits.shape == (len(scope), num_output_units, num_channels, num_categories)
        elif probs is not None:
            assert probs.shape == (len(scope), num_output_units, num_channels, num_categories)
        self.probs = probs
        self.logits = logits

    @property
    def config(self) -> dict:
        config = super().config
        config.update(num_categories=self.num_categories)
        return config

    @property
    def params(self) -> Dict[str, Parameter]:
        if self.logits is None:
            return dict(probs=self.probs)
        return dict(logits=self.logits)


class GaussianLayer(InputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        mean: Optional[Parameter] = None,
        stddev: Optional[Parameter] = None,
        log_partition: Optional[Parameter] = None,
        mean_parameterization: Optional[Parameterization] = None,
        mean_initializer: Optional[Initializer] = None,
        stddev_parameterization: Optional[Parameterization] = None,
        stddev_initializer: Optional[Initializer] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        assert (mean is None and stddev is None) or (
            mean is not None and stddev is not None
        ), "Either both 'mean' and 'variance' has to be specified or none of them"
        if mean is None and stddev is None:
            if mean_initializer is None:
                mean_initializer = NormalInitializer()
            if stddev_initializer is None:
                stddev_initializer = NormalInitializer()
            mean = TensorParameter(
                self.num_variables, num_output_units, num_channels, initializer=mean_initializer
            )
            stddev = TensorParameter(
                self.num_variables, num_output_units, num_channels, initializer=stddev_initializer
            )
            if mean_parameterization is None:
                mean = Parameter.from_leaf(mean)
            else:
                mean = mean_parameterization(mean)
            if stddev_parameterization is None:
                stddev = Parameter.from_unary(
                    ScaledSigmoidParameter(stddev.shape, vmin=1e-5, vmax=1.0), stddev
                )
            else:
                stddev = stddev_parameterization(stddev)
        else:
            assert mean.shape == (len(scope), num_output_units, num_channels)
            assert stddev.shape == (len(scope), num_output_units, num_channels)
        self.mean = mean
        self.stddev = stddev
        self.log_partition = log_partition

    @property
    def params(self) -> Dict[str, Parameter]:
        params = dict(mean=self.mean, stddev=self.stddev)
        if self.log_partition is not None:
            params.update(log_partition=self.log_partition)
        return params


class LogPartitionLayer(InputLayer):
    def __init__(self, scope: Scope, num_output_units: int, num_channels: int, value: Parameter):
        assert value.shape == (num_output_units,)
        super().__init__(scope, num_output_units, num_channels)
        self.value = value

    @property
    def params(self) -> Dict[str, Parameter]:
        params = super().params
        params.update(value=self.value)
        return params


class ProductLayer(Layer, ABC):
    """The abstract base class for Symbolic product layers."""

    def __init__(self, scope: Scope, num_input_units: int, num_output_units: int, arity: int = 2):
        super().__init__(scope, num_input_units, num_output_units, arity)

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "arity": self.arity,
        }


class HadamardLayer(ProductLayer):
    """The Symbolic Hadamard product layer."""

    def __init__(self, scope: Scope, num_input_units: int, arity: int = 2):
        super().__init__(
            scope, num_input_units, HadamardLayer.num_prod_units(num_input_units), arity=arity
        )

    @staticmethod
    def num_prod_units(in_num_units: int) -> int:
        return in_num_units


class KroneckerLayer(ProductLayer):
    """The Symbolic Kronecker product layer."""

    def __init__(self, scope: Scope, num_input_units: int, arity: int = 2):
        super().__init__(
            scope,
            num_input_units,
            KroneckerLayer.num_prod_units(num_input_units, arity),
            arity=arity,
        )

    @staticmethod
    def num_prod_units(in_num_units: int, arity: int) -> int:
        return cast(int, in_num_units**arity)


class SumLayer(Layer, ABC):
    """The abstract base class for Symbolic sum layers."""


class DenseLayer(SumLayer):
    """The Symbolic dense sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        weight: Optional[Parameter] = None,
        parameterization: Optional[Parameterization] = None,
        initializer: Optional[Initializer] = None,
    ):
        super().__init__(scope, num_input_units, num_output_units, arity=1)
        if weight is None:
            if initializer is None:
                initializer = NormalInitializer()
            weight = TensorParameter(num_output_units, num_input_units, initializer=initializer)
            if parameterization is None:
                weight = Parameter.from_leaf(weight)
            else:
                weight = parameterization(weight)
        else:
            assert weight.shape == (num_output_units, num_input_units)
        self.weight = weight

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
        }

    @property
    def params(self) -> Dict[str, Parameter]:
        return dict(weight=self.weight)


class MixingLayer(SumLayer):
    """The Symbolic mixing sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        arity: int,
        weight: Optional[Parameter] = None,
        parameterization: Optional[Parameterization] = None,
        initializer: Optional[Initializer] = None,
    ):
        super().__init__(scope, num_units, num_units, arity)
        if weight is None:
            if initializer is None:
                initializer = NormalInitializer()
            weight = TensorParameter(num_units, arity, initializer=initializer)
            if parameterization is None:
                weight = Parameter.from_leaf(weight)
            else:
                weight = parameterization(weight)
        else:
            assert weight.shape == (num_units, arity)
        self.weight = weight

    @property
    def config(self) -> Dict[str, Any]:
        return {"scope": self.scope, "num_units": self.num_input_units, "arity": self.arity}

    @property
    def params(self) -> Dict[str, Parameter]:
        return dict(weight=self.weight)


class IndexLayer(SumLayer):
    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        indices: List[int],
    ):
        assert num_output_units == len(indices)
        assert 0 <= min(indices) and max(indices) < num_input_units
        super().__init__(scope, num_input_units, num_output_units, arity=1)
        self.indices = indices

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "indices": self.indices,
        }
