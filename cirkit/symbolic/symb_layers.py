from abc import ABC, abstractmethod
from typing import Optional, List, cast

from cirkit.utils import Scope
from cirkit.symbolic.symb_op import SymbLayerOperation


class SymbLayer(ABC):
    """The abstract base class for symbolic layers in symbolic circuits."""

    scope: Scope
    num_units: int
    operation: Optional[SymbLayerOperation]
    inputs: List["SymbLayer"]
    outputs: List["SymbLayer"]  # reverse reference of inputs

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        operation: Optional[SymbLayerOperation] = None,
        inputs: Optional[List["SymbLayer"]] = None,
    ):
        self.scope = scope
        self.num_units = num_units
        self.operation = operation
        self.inputs = inputs if inputs is not None else []
        for sl in inputs:
            sl.outputs.append(self)

    def __repr__(self) -> str:
        ...

    @property
    def arity(self) -> int:
        return len(self.inputs)

    @property
    def kwargs(self) -> dict:
        return {}


class SymbInputLayer(SymbLayer):
    """The (abstract???) base class for symbolic input layers."""

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        num_channels: int,
        operation: Optional[SymbLayerOperation] = None,
    ):
        super().__init__(scope, num_units, operation=operation, inputs=[])
        self.num_channels = num_channels


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
    ...


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
    def kwargs(self) -> dict:
        return dict(value=self.value)


class SymbProdLayer(ABC, SymbLayer):
    """The abstract base class for symbolic product layers."""

    @staticmethod
    @abstractmethod
    def num_prod_units(num_units: int, arity: int) -> int:
        ...


class SymbHadamardLayer(SymbProdLayer):
    """The symbolic Hadamard product layer."""

    @staticmethod
    def num_prod_units(num_input_units: int, arity: int) -> int:
        return num_input_units


class SymbKroneckerLayer(SymbProdLayer):
    """The symbolic Kronecker product layer."""

    @staticmethod
    def num_prod_units(num_input_units: int, arity: int) -> int:
        return cast(int, num_input_units ** arity)


class SymbSumLayer(ABC, SymbLayer):
    """The abstract base class for symbolic sum layers."""


class SymbDenseLayer(SymbSumLayer):
    """The symbolic dense sum layer."""


class SymbMixingLayer(SymbSumLayer):
    """The symbolic mixing sum layer."""
