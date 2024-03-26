from abc import ABC
from typing import List, Optional

from cirkit.newer.symbolic.symb_op import SymbLayerOperation
from cirkit.newer.utils import Scope


class SymbLayer(ABC):
    """The abstract base class for symbolic layers in symbolic circuits."""

    scope: Scope
    num_units: int
    operator: Optional[SymbLayerOperation]
    inputs: List["SymbLayer"]
    outputs: List["SymbLayer"]  # reverse reference of inputs

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        operator: Optional[SymbLayerOperation] = None,
        inputs: Optional[List["SymbLayer"]] = None,
    ):
        self.scope = scope
        self.num_units = num_units
        self.operator = operator
        self.inputs = inputs if inputs is not None else []
        for sl in inputs:
            sl.outputs.append(self)

    def __repr__(self) -> str:
        ...

    @property
    def arity(self) -> int:
        return len(self.inputs)
