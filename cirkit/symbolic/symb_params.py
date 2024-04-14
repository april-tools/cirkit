from abc import ABC
from typing import Tuple


class AbstractSymbParameter(ABC):
    ...


class SymbParameter(AbstractSymbParameter):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self._shape = tuple(shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class SymbParameterUnary(ABC, AbstractSymbParameter):
    def __init__(self, opd: AbstractSymbParameter) -> None:
        super().__init__()
        self.opd = opd


class SymbParameterBinary(ABC, AbstractSymbParameter):
    def __init__(self, lhs: AbstractSymbParameter, rhs: AbstractSymbParameter) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs


class SymbParameterReduce(ABC, AbstractSymbParameter):
    def __init__(self, *opds: AbstractSymbParameter):
        super().__init__()
        self.opds = list(opds)


class SymbParameterNormalize(ABC, SymbParameterUnary):
    def __init__(self, opd: AbstractSymbParameter, axis: int = -1):
        super().__init__(opd)
        self.axis = axis


class SymbHadamard(SymbParameterBinary):
    ...


class SymbKronecker(SymbParameterBinary):
    ...


class SymbSoftmax(SymbParameterNormalize):
    def __init__(self, opd: AbstractSymbParameter, axis: int = -1):
        super().__init__(opd, axis)
        self.axis = axis


class SymbLogSoftmax(SymbParameterNormalize):
    def __init__(self, opd: AbstractSymbParameter, axis: int = -1):
        super().__init__(opd, axis)
        self.axis = axis


class SymbScaleSigmoid(SymbParameterUnary):
    def __init__(self, opd: AbstractSymbParameter, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__(opd)
        self.vmin = vmin
        self.vmax = vmax


class SymbSoftplus(SymbParameterUnary):
    ...


class SymbConcat(SymbParameterReduce):
    def __init__(self, *opds: AbstractSymbParameter, axis: int = -1):
        super().__init__(*opds)
        self.axis = axis


class SymbStack(SymbParameterReduce):
    def __init__(self, *opds: AbstractSymbParameter, axis: int = -1):
        super().__init__(*opds)
        self.axis = axis
