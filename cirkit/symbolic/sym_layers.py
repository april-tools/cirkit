from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple, cast

from cirkit.symbolic.sym_params import AbstractSymParameter, SymParameter
from cirkit.utils.scope import Scope

AbstractSymLayerOperator = IntEnum  # TODO: switch to StrEnum (>=py3.11) or better alternative


class SymLayerOperator(AbstractSymLayerOperator):
    """Types of Symolic operations on layers."""

    def _generate_next_value_(self, start: int, count: int, last_values: list) -> int:
        return -(
            count + 1
        )  # Enumerate negative integers as the user can extend them with non-negative ones

    INTEGRATION = auto()
    DIFFERENTIATION = auto()
    KRONECKER = auto()


class SymLayer(ABC):
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
    def learnable_params(self) -> Dict[str, AbstractSymParameter]:
        return {}


class SymParameterPlaceholder(AbstractSymParameter):
    def __init__(self, layer: SymLayer, name: str):
        self.layer = layer
        self.name = name


class SymInputLayer(SymLayer):
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


class SymExpFamilyLayer(ABC, SymInputLayer):
    def __init__(self, scope: Scope, num_output_units: int, num_channels: int):
        super().__init__(scope, len(scope), num_output_units, num_channels)

    @abstractmethod
    def sufficient_statistics_shape(self) -> Tuple[int, ...]:
        ...


class SymCategoricalLayer(SymExpFamilyLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        num_categories: int = 2,
        weight: Optional[SymParameter] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        self.num_categories = num_categories
        if weight is None:
            weight = SymParameter(
                self.num_variables, num_output_units, num_channels, num_categories
            )
        self.weight = weight

    @property
    def sufficient_statistics_shape(self) -> Tuple[int, ...]:
        return self.num_channels, self.num_categories

    @property
    def hparams(self) -> dict:
        hparams = super().hparams
        hparams.update(num_categories=self.num_categories)
        return hparams

    @property
    def learnable_params(self) -> Dict[str, AbstractSymParameter]:
        return dict(weight=self.weight)


class SymNormalLayer(SymExpFamilyLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        mean: Optional[SymParameter] = None,
        variance: Optional[SymParameter] = None,
    ):
        super().__init__(scope, num_output_units, num_channels)
        assert (mean is None and variance is None) or (
            mean is not None and variance is not None
        ), "Either both 'mean' and 'variance' has to be specified or none of them"
        if mean is None and variance is None:
            mean = SymParameter(self.num_variables, num_output_units, num_channels)
            variance = SymParameter(self.num_variables, num_output_units, num_channels)
        self.mean = mean
        self.variance = variance

    @property
    def sufficient_statistics_shape(self) -> Tuple[int, ...]:
        return self.num_channels, 2

    @property
    def learnable_params(self) -> Dict[str, AbstractSymParameter]:
        return dict(mean=self.mean, variance=self.variance)


class SymConstantLayer(SymInputLayer):
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


class SymProdLayer(ABC, SymLayer):
    """The abstract base class for Symolic product layers."""

    def __init__(self, scope: Scope, num_input_units: int, arity: int = 2):
        num_output_units = SymProdLayer.num_prod_units(num_input_units, arity)
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


class SymHadamardLayer(SymProdLayer):
    """The Symolic Hadamard product layer."""

    @staticmethod
    def num_prod_units(in_num_units: int, arity: int) -> int:
        return in_num_units


class SymKroneckerLayer(SymProdLayer):
    """The Symolic Kronecker product layer."""

    @staticmethod
    def num_prod_units(in_num_units: int, arity: int) -> int:
        return cast(int, in_num_units**arity)


class SymSumLayer(ABC, SymLayer):
    """The abstract base class for Symolic sum layers."""


class SymDenseLayer(SymSumLayer):
    """The Symolic dense sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        weight: Optional[SymParameter] = None,
    ):
        super().__init__(scope, num_input_units, num_output_units, arity=1)
        if weight is None:
            weight = SymParameter(num_output_units, num_input_units)
        self.weight = weight

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
        }

    @property
    def learnable_params(self) -> Dict[str, AbstractSymParameter]:
        return dict(weight=self.weight)


class SymMixingLayer(SymSumLayer):
    """The Symolic mixing sum layer."""

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        arity: int,
        weight: Optional[SymParameter] = None,
    ):
        super().__init__(scope, num_units, num_units, arity)
        if weight is None:
            weight = SymParameter(num_units, arity)
        self.weight = weight

    @property
    def hparams(self) -> Dict[str, Any]:
        return {"scope": self.scope, "num_units": self.num_input_units, "arity": self.arity}

    @property
    def learnable_params(self) -> Dict[str, AbstractSymParameter]:
        return dict(weight=self.weight)
