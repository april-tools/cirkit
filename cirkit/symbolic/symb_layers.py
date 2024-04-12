from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, cast

from cirkit.utils import Scope


class SymbLayerOperator(Enum):
    """Types of symbolic operations on layers."""

    INTEGRATION = auto()
    DIFFERENTIATION = auto()
    MULTIPLICATION = auto()


@dataclass(frozen=True)
class SymbLayerOperation:
    """The symbolic operation applied on a SymbLayer."""

    operator: SymbLayerOperator
    operands: Tuple["SymbLayer", ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


class SymbLayer(ABC):
    def __init__(
        self, scope: Scope, num_units: int, operation: Optional[SymbLayerOperation] = None
    ):
        self.scope = scope
        self.num_units = num_units
        self.operation = operation

    @property
    def hparams(self) -> dict:
        return {}


class SymbInputLayer(SymbLayer):
    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_units, operation=operation)
        self.num_channels = num_channels

    @property
    def hparams(self) -> dict:
        return dict(num_units=self.num_units, num_channels=self.num_channels)


class SymbExpFamilyLayer(SymbInputLayer):
    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_units, num_channels, operation=operation)


class SymbCategoricalLayer(SymbExpFamilyLayer):
    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
        num_categories: int = 2,
    ):
        super().__init__(scope, num_units, num_channels, operation=operation)
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
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
        value: Optional[float] = None,
    ):
        assert (
            operation is not None or value is not None
        ), "Eiether 'operation' or 'value' must be specified to construct a constant layer"
        super().__init__(scope, num_units, num_channels, operation=operation)
        self.value = value

    @property
    def hparams(self) -> dict:
        hparams = super().hparams
        hparams.update(value=self.value)
        return hparams


class SymbProdLayer(ABC, SymbLayer):
    """The abstract base class for symbolic product layers."""

    def __init__(
        self,
        scope: Scope,
        in_num_units: int,
        arity: int,
        operation: Optional[SymbLayerOperation] = None,
    ):
        num_units = SymbProdLayer.num_prod_units(in_num_units, arity)
        super().__init__(scope, num_units, operation=operation)

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


class SymbMixingLayer(SymbSumLayer):
    """The symbolic mixing sum layer."""
