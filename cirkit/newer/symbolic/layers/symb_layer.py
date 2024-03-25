from abc import ABC
from typing import List, Optional

from cirkit.newer.symbolic.symb_op import SymbLayerOperation
from cirkit.newer.utils import Scope


class SymbLayer(ABC):
    """The abstract base class for symbolic layers in symbolic circuits."""

    scope: Scope
    inputs: List["SymbLayer"]
    outputs: List["SymbLayer"]  # reverse reference of inputs
    arity: int  # length of inputs
    num_units: int
    construct_op: SymbLayerOperation
    concrete_layer: Optional["Layer"]  # ref to the corresponding Layer

    def __repr__(self) -> str:
        ...
