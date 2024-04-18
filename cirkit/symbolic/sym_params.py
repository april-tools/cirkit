from abc import ABC
from typing import Tuple


class AbstractSymParameter(ABC):
    ...


class SymParameter(AbstractSymParameter):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self._shape = tuple(shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class SymParameterUnary(ABC, AbstractSymParameter):
    def __init__(self, opd: AbstractSymParameter) -> None:
        super().__init__()
        self.opd = opd


class SymParameterBinary(ABC, AbstractSymParameter):
    def __init__(self, lhs: AbstractSymParameter, rhs: AbstractSymParameter) -> None:
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs


class SymParameterReduce(ABC, AbstractSymParameter):
    def __init__(self, *opds: AbstractSymParameter):
        super().__init__()
        self.opds = list(opds)


class SymHadamard(SymParameterBinary):
    ...


class SymKronecker(SymParameterBinary):
    ...


class SymConcat(SymParameterReduce):
    def __init__(self, *opds: AbstractSymParameter, axis: int = -1):
        super().__init__(*opds)
        self.axis = axis


class SymStack(SymParameterReduce):
    def __init__(self, *opds: AbstractSymParameter, axis: int = -1):
        super().__init__(*opds)
        self.axis = axis


class SymParameterNormalize(ABC, SymParameterUnary):
    def __init__(self, opd: AbstractSymParameter, axis: int = -1):
        super().__init__(opd)
        self.axis = axis


class SymSoftplus(SymParameterUnary):
    ...


class SymScaleSigmoid(SymParameterUnary):
    def __init__(self, opd: AbstractSymParameter, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__(opd)
        self.vmin = vmin
        self.vmax = vmax


class SymSoftmax(SymParameterNormalize):
    def __init__(self, opd: AbstractSymParameter, axis: int = -1):
        super().__init__(opd, axis)
        self.axis = axis


class SymLogSoftmax(SymParameterNormalize):
    def __init__(self, opd: AbstractSymParameter, axis: int = -1):
        super().__init__(opd, axis)
        self.axis = axis
