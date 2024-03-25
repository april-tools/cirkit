from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Mapping, Tuple
from typing_extensions import TypeAlias  # FUTURE: in typing from 3.10, deprecated in 3.12

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    from cirkit.newer.symbolic.layers.symb_layer import SymbLayer
    from cirkit.newer.symbolic.symb_circuit import SymbCircuit


class SymbOperator(Enum):
    """Types of symbolic operations on circuits/layers."""

    NONE = auto()
    INTEGRATION = auto()
    DIFFERENTIATION = auto()
    MULTIPLICATION = auto()


# TODO: do we need anything more than int?
SymbOpMetadata: TypeAlias = Mapping[str, int]
"""The metadata for symbolic operations."""


# We add frozen=True because we expect this should be immutable.
# FUTURE: kw_only=True in 3.10
@dataclass(frozen=True)
class SymbLayerOperation:
    """The symbolic operation applied on a SymbLayer."""

    operator: SymbOperator = SymbOperator.NONE
    operands: Tuple["SymbLayer", ...] = ()
    metadata: SymbOpMetadata = field(default_factory=dict)


# We add frozen=True because we expect this should be immutable.
# FUTURE: kw_only=True in 3.10
@dataclass(frozen=True)
class SymbCircuitOperation:
    """The symbolic operation applied on a SymbCircuit."""

    operator: SymbOperator = SymbOperator.NONE
    operands: Tuple["SymbCircuit", ...] = ()
    metadata: SymbOpMetadata = field(default_factory=dict)
