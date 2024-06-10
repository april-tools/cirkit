from abc import ABC
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, cast

from cirkit.symbolic.parameters import Parameter, Parameterization, TensorParameter
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
        logits_param: Optional[Parameterization] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        self.num_categories = num_categories
        if logits is None:
            logits = TensorParameter(len(scope), num_output_units, num_channels, num_categories)
            logits = Parameter.from_leaf(logits) if logits_param is None else logits_param(logits)
        else:
            assert logits.shape == (len(scope), num_output_units, num_channels, num_categories)
        self.logits = logits

    @property
    def config(self) -> dict:
        config = super().config
        config.update(num_categories=self.num_categories)
        return config

    @property
    def params(self) -> Dict[str, Parameter]:
        params = super().params
        params.update(logits=self.logits)
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
    """The abstract base class for Symolic product layers."""

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
    """The Symolic Hadamard product layer."""

    def __init__(self, scope: Scope, num_input_units: int, arity: int = 2):
        super().__init__(
            scope, num_input_units, HadamardLayer.num_prod_units(num_input_units), arity=arity
        )

    @staticmethod
    def num_prod_units(in_num_units: int) -> int:
        return in_num_units


class KroneckerLayer(ProductLayer):
    """The Symolic Kronecker product layer."""

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
    """The abstract base class for Symolic sum layers."""


class DenseLayer(SumLayer):
    """The Symolic dense sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        weight: Optional[Parameter] = None,
        weight_param: Optional[Parameterization] = None,
    ):
        super().__init__(scope, num_input_units, num_output_units, arity=1)
        if weight is None:
            weight = TensorParameter(num_output_units, num_input_units)
            weight = Parameter.from_leaf(weight) if weight_param is None else weight_param(weight)
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
    """The Symolic mixing sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        arity: int,
        weight: Optional[Parameter] = None,
        weight_param: Optional[Parameterization] = None,
    ):
        super().__init__(scope, num_units, num_units, arity)
        if weight is None:
            weight = TensorParameter(num_units, arity)
            weight = Parameter.from_leaf(weight) if weight_param is None else weight_param(weight)
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
