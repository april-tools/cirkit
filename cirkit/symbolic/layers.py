from abc import ABC, abstractmethod
from enum import IntEnum, auto
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, cast

from cirkit.symbolic.params import (
    AbstractParameter,
    Parameter,
    Parameterization,
    ScaledSigmoidParameter,
    SoftmaxParameter,
)
from cirkit.utils.scope import Scope

AbstractLayerOperator = IntEnum  # TODO: switch to StrEnum (>=py3.11) or better alternative


class LayerOperation(AbstractLayerOperator):
    """Types of Symolic operations on layers."""

    def _generate_next_value_(self, start: int, count: int, last_values: list) -> int:
        return -(
            count + 1
        )  # Enumerate negative integers as the user can extend them with non-negative ones

    INTEGRATION = auto()
    DIFFERENTIATION = auto()
    MULTIPLICATION = auto()


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
    def hparams(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

    @property
    def learnable_params(self) -> Dict[str, AbstractParameter]:
        return {}


class PlaceholderParameter(AbstractParameter):
    def __init__(self, layer: Layer, name: str):
        assert name in layer.learnable_params
        self.layer = layer
        self.name = name

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.layer.learnable_params[self.name].shape


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
    def hparams(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "num_channels": self.num_channels,
        }


class ExpFamilyLayer(InputLayer, ABC):
    def __init__(self, scope: Scope, num_output_units: int, num_channels: int):
        super().__init__(scope, num_output_units, num_channels)

    @abstractmethod
    def sufficient_statistics_shape(self) -> Tuple[int, ...]:
        ...


class CategoricalLayer(ExpFamilyLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        num_categories: int = 2,
        probs: Optional[AbstractParameter] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        self.num_categories = num_categories
        if probs is None:
            probs = SoftmaxParameter(
                Parameter(self.num_variables, num_output_units, num_channels, num_categories),
                axis=-1,
            )
        self.probs = probs

    @property
    def sufficient_statistics_shape(self) -> Tuple[int, ...]:
        return self.num_channels, self.num_categories

    @property
    def hparams(self) -> dict:
        hparams = super().hparams
        hparams.update(num_categories=self.num_categories)
        return hparams

    @property
    def learnable_params(self) -> Dict[str, AbstractParameter]:
        return dict(probs=self.probs)


class NormalLayer(ExpFamilyLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        mean: Optional[AbstractParameter] = None,
        variance: Optional[AbstractParameter] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        assert (mean is None and variance is None) or (
            mean is not None and variance is not None
        ), "Either both 'mean' and 'variance' has to be specified or none of them"
        if mean is None and variance is None:
            mean = Parameter(self.num_variables, num_output_units, num_channels)
            variance = ScaledSigmoidParameter(
                Parameter(self.num_variables, num_output_units, num_channels), vmin=1e-5, vmax=1.0
            )
        self.mean = mean
        self.variance = variance

    @property
    def sufficient_statistics_shape(self) -> Tuple[int, ...]:
        return self.num_channels, 2

    @property
    def learnable_params(self) -> Dict[str, AbstractParameter]:
        return dict(mean=self.mean, variance=self.variance)


class ConstantLayer(InputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int = 1,
        value: Optional[float] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        self.value = value if value is not None else 0

    @property
    def hparams(self) -> Dict[str, Any]:
        hparams = super().hparams
        hparams.update(value=self.value)
        return hparams


class ProductLayer(Layer, ABC):
    """The abstract base class for Symolic product layers."""

    def __init__(self, scope: Scope, num_input_units: int, arity: int = 2):
        num_output_units = ProductLayer.num_prod_units(num_input_units, arity)
        super().__init__(scope, num_input_units, num_output_units, arity)

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "arity": self.arity,
        }

    @staticmethod
    @abstractmethod
    def num_prod_units(num_units: int, arity: int) -> int:
        ...


class HadamardLayer(ProductLayer):
    """The Symolic Hadamard product layer."""

    @staticmethod
    def num_prod_units(in_num_units: int, arity: int) -> int:
        return in_num_units


class KroneckerLayer(ProductLayer):
    """The Symolic Kronecker product layer."""

    @staticmethod
    def num_prod_units(in_num_units: int, arity: int) -> int:
        return cast(int, in_num_units**arity)


class SumLayer(Layer, ABC):
    """The abstract base class for Symolic sum layers."""


class DenseLayer(SumLayer):
    """The Symolic dense sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        weight: Optional[AbstractParameter] = None,
        weight_param: Optional[Parameterization] = None,
    ):
        super().__init__(scope, num_input_units, num_output_units, arity=1)
        if weight is None:
            weight = Parameter(num_output_units, num_input_units)
            if weight_param is not None:
                weight = weight_param(weight)
        self.weight = weight

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
        }

    @property
    def learnable_params(self) -> Dict[str, AbstractParameter]:
        return dict(weight=self.weight)


class MixingLayer(SumLayer):
    """The Symolic mixing sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        arity: int,
        weight: Optional[AbstractParameter] = None,
        weight_param: Optional[Parameterization] = None,
    ):
        super().__init__(scope, num_units, num_units, arity)
        if weight is None:
            weight = Parameter(num_units, arity)
            if weight_param is not None:
                weight = weight_param(weight)
        self.weight = weight

    @property
    def hparams(self) -> Dict[str, Any]:
        return {"scope": self.scope, "num_units": self.num_input_units, "arity": self.arity}

    @property
    def learnable_params(self) -> Dict[str, AbstractParameter]:
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
    def hparams(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "indices": self.indices,
        }
