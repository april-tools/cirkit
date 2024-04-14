from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple, cast, Type, Callable

from cirkit.symbolic.symb_params import SymbParameter, AbstractSymbParameter
from cirkit.utils import Scope


AbstractSymbLayerOperator = IntEnum  # TODO: switch to StrEnum (>=py3.11) or better alternative


class SymbLayerOperator(AbstractSymbLayerOperator):
    """Types of symbolic operations on layers."""
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> int:
        return -(count + 1)  # Enumerate negative integers as the user can extend them with non-negative ones

    NOP = auto()
    INTEGRATION = auto()
    DIFFERENTIATION = auto()
    KRONECKER = auto()


@dataclass(frozen=True)
class SymbLayerOperation:
    """The symbolic operation applied on a SymbLayer."""

    operator: AbstractSymbLayerOperator
    operands: Tuple["SymbLayer", ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


ParamFactoryFunction = Callable[[Tuple[int, ...]], SymbParameter]


class SymbLayer(ABC):
    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        operation: Optional[SymbLayerOperation] = None
    ):
        self.scope = scope
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity
        self.operation = operation

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            'scope': self.scope,
            'num_input_units': self.num_input_units,
            'num_output_units': self.num_output_units,
            'arity': self.arity
        }

    @property
    def params(self) -> Dict[str, AbstractSymbParameter]:
        return {}


class SymbInputLayer(SymbLayer):
    def __init__(
        self,
        scope: Scope,
        num_variables: int,
        num_output_units: int,
        num_channels: int = 1,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_variables, num_output_units, num_channels, operation=operation)

    @property
    def num_variables(self) -> int:
        return self.num_input_units

    @property
    def num_channels(self) -> int:
        return self.arity

    @property
    def hparams(self) -> Dict[str, Any]:
        return {
            'scope': self.scope,
            'num_variables': self.num_variables,
            'num_output_units': self.num_output_units,
            'num_channels': self.num_channels
        }


class SymbExpFamilyLayer(SymbInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_variables: int,
        num_output_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_variables, num_output_units, num_channels, operation=operation)


class SymbCategoricalLayer(SymbExpFamilyLayer):
    def __init__(
        self,
        scope: Scope,
        num_variables: int,
        num_output_units: int,
        num_channels: int,
        num_categories: int = 2,
        param_factory: ParamFactoryFunction = SymbParameter,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_variables, num_output_units, num_channels, operation=operation)
        self.num_categories = num_categories

    @property
    def hparams(self) -> dict:
        hparams = super().hparams
        hparams.update(num_categories=self.num_categories)
        return hparams


class SymbConstantLayer(SymbInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_variables: int,
        num_output_units: int,
        num_channels: int = 1,
        operation: Optional[SymbLayerOperation] = None,
        value: Optional[float] = None,
    ):
        assert (
            operation is not None or value is not None
        ), "Eiether 'operation' or 'value' must be specified to construct a constant layer"
        super().__init__(scope, num_variables, num_output_units, num_channels, operation=operation)
        self.value = value

    @property
    def hparams(self) -> Dict[str, Any]:
        hparams = super().hparams
        hparams.update(value=self.value)
        return hparams


class SymbProdLayer(ABC, SymbLayer):
    """The abstract base class for symbolic product layers."""

    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        arity: int = 2,
        operation: Optional[SymbLayerOperation] = None,
    ):
        num_output_units = SymbProdLayer.num_prod_units(num_input_units, arity)
        super().__init__(scope, num_input_units, num_output_units, arity, operation=operation)

    @staticmethod
    @abstractmethod
    def num_prod_units(num_units: int, arity: int) -> int:
        ...


class SymbHadamardLayer(SymbProdLayer):
    """The symbolic Hadamard product layer."""

    @staticmethod
    def num_prod_units(in_num_units: int, arity: int) -> int:
        return in_num_units


class SymbKroneckerLayer(SymbProdLayer):
    """The symbolic Kronecker product layer."""

    @staticmethod
    def num_prod_units(in_num_units: int, arity: int) -> int:
        return cast(int, in_num_units**arity)


class SymbSumLayer(ABC, SymbLayer):
    """The abstract base class for symbolic sum layers."""


class SymbDenseLayer(SymbSumLayer):
    """The symbolic dense sum layer."""
    def __init__(
            self,
            scope: Scope,
            num_input_units: int,
            num_output_units: int,
            param_factory: ParamFactoryFunction = SymbParameter,
            operation: Optional[SymbLayerOperation] = None
    ):
        super().__init__(scope, num_input_units, num_output_units, arity=1, operation=operation)
        self.weight = param_factory((num_output_units, num_input_units))

    @property
    def params(self) -> Dict[str, AbstractSymbParameter]:
        return dict(weight=self.weight)


class SymbMixingLayer(SymbSumLayer):
    """The symbolic mixing sum layer."""
    def __init__(
            self,
            scope: Scope,
            num_units: int,
            arity: int,
            param_factory: ParamFactoryFunction = SymbParameter,
            operation: Optional[SymbLayerOperation] = None
    ):
        super().__init__(scope, num_units, num_units, arity, operation=operation)
        self.weight = param_factory((num_units, arity))

    @property
    def params(self) -> Dict[str, AbstractSymbParameter]:
        return dict(weight=self.weight)
