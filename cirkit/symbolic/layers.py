from abc import ABC
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Tuple, cast

from cirkit.symbolic.initializers import NormalInitializer
from cirkit.symbolic.parameters import (
    Parameter,
    ParameterFactory,
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
        num_categories: int,
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
        if num_categories < 2:
            raise ValueError("At least two categories must be specified")
        super().__init__(scope, num_output_units, num_channels)
        self.num_categories = num_categories
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
        return self.num_variables, self.num_output_units, self.num_channels, self.num_categories

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


class GaussianLayer(InputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        mean: Optional[Parameter] = None,
        stddev: Optional[Parameter] = None,
        log_partition: Optional[Parameter] = None,
        mean_factory: Optional[ParameterFactory] = None,
        stddev_factory: Optional[ParameterFactory] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        if mean is None:
            if mean_factory is None:
                mean = Parameter.from_leaf(
                    TensorParameter(*self.mean_stddev_shape, initializer=NormalInitializer())
                )
            else:
                mean = mean_factory(self.mean_stddev_shape)
        if stddev is None:
            if stddev_factory is None:
                stddev = Parameter.from_unary(
                    ScaledSigmoidParameter(self.mean_stddev_shape, vmin=1e-5, vmax=1.0),
                    TensorParameter(*self.mean_stddev_shape, initializer=NormalInitializer()),
                )
            else:
                stddev = stddev_factory(self.mean_stddev_shape)
        if mean.shape != self.mean_stddev_shape:
            raise ValueError(
                f"Expected parameter shape {self.mean_stddev_shape}, found {mean.shape}"
            )
        if stddev.shape != self.mean_stddev_shape:
            raise ValueError(
                f"Expected parameter shape {self.mean_stddev_shape}, found {stddev.shape}"
            )
        if log_partition is not None and log_partition.shape != self.log_partition_shape:
            raise ValueError(
                f"Expected parameter shape {self.log_partition_shape}, found {log_partition.shape}"
            )
        self.mean = mean
        self.stddev = stddev
        self.log_partition = log_partition

    @property
    def mean_stddev_shape(self) -> Tuple[int, ...]:
        return self.num_variables, self.num_output_units, self.num_channels

    @property
    def log_partition_shape(self) -> Tuple[int, ...]:
        return self.num_variables, self.num_output_units, self.num_channels

    @property
    def params(self) -> Dict[str, Parameter]:
        params = dict(mean=self.mean, stddev=self.stddev)
        if self.log_partition is not None:
            params.update(log_partition=self.log_partition)
        return params


class LogPartitionLayer(InputLayer):
    def __init__(self, scope: Scope, num_output_units: int, num_channels: int, value: Parameter):
        super().__init__(scope, num_output_units, num_channels)
        if value.shape != self.value_shape:
            raise ValueError(f"Expected parameter shape {self.value_shape}, found {value.shape}")
        self.value = value

    @property
    def value_shape(self) -> Tuple[int, ...]:
        return (self.num_output_units,)

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
        weight_factory: Optional[ParameterFactory] = None,
    ):
        super().__init__(scope, num_input_units, num_output_units, arity=1)
        if weight is None:
            if weight_factory is None:
                weight = Parameter.from_leaf(
                    TensorParameter(*self.weight_shape, initializer=NormalInitializer())
                )
            else:
                weight = weight_factory(self.weight_shape)
        if weight.shape != self.weight_shape:
            raise ValueError(f"Expected parameter shape {self.weight_shape}, found {weight.shape}")
        self.weight = weight

    @property
    def weight_shape(self) -> Tuple[int, ...]:
        return self.num_output_units, self.num_input_units

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
        weight_factory: Optional[ParameterFactory] = None,
    ):
        super().__init__(scope, num_units, num_units, arity)
        if weight is None:
            if weight_factory is None:
                weight = Parameter.from_leaf(
                    TensorParameter(*self.weight_shape, initializer=NormalInitializer())
                )
            else:
                weight = weight_factory(self.weight_shape)
        if weight.shape != self.weight_shape:
            raise ValueError(f"Expected parameter shape {self.weight_shape}, found {weight.shape}")
        self.weight = weight

    @property
    def weight_shape(self) -> Tuple[int, ...]:
        return self.num_input_units, self.arity

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
