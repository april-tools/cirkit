from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, Dict, Tuple


class AbstractParameter(ABC):
    @cached_property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def hparams(self) -> Dict[str, Any]:
        return {}


class Parameter(AbstractParameter):
    def __init__(self, *shape: int):
        self._shape = tuple(shape)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class OpParameter(ABC, AbstractParameter):
    ...


Parameterization = Callable[[AbstractParameter], OpParameter]


class UnaryOpParameter(ABC, OpParameter):
    def __init__(self, opd: AbstractParameter) -> None:
        self.opd = opd


class BinaryOpParameter(ABC, OpParameter):
    def __init__(self, lhs: AbstractParameter, rhs: AbstractParameter) -> None:
        self.lhs = lhs
        self.rhs = rhs


class HadamardParameter(AbstractParameter):
    def __init__(self, lhs: AbstractParameter, rhs: AbstractParameter) -> None:
        assert lhs.shape == rhs.shape
        self.lhs = lhs
        self.rhs = rhs

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.lhs.shape


class KroneckerParameter(AbstractParameter):
    def __init__(self, lhs: AbstractParameter, rhs: AbstractParameter) -> None:
        assert len(lhs.shape) == len(rhs.shape)
        assert lhs.shape[:-1] == rhs.shape[:-1]
        self.lhs = lhs
        self.rhs = rhs

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return *self.lhs.shape[:-1], self.lhs.shape[-1] * self.rhs.shape[-1]


class EntrywiseOpParameter(ABC, AbstractParameter):
    def __init__(self, opd: AbstractParameter):
        super().__init__(opd)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.opd.shape


class ReduceOpParameter(ABC, UnaryOpParameter):
    def __init__(self, opd: AbstractParameter, axis: int = -1):
        super().__init__(opd)
        axis = axis if axis >= 0 else axis + len(opd.shape)
        assert 0 <= axis < len(opd.shape)
        self.axis = axis

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return *self.opd.shape[: self.axis], *self.opd.shape[self.axis + 1]

    @property
    def hparams(self) -> Dict[str, Any]:
        return dict(axis=self.axis)


class ExponentialParameter(EntrywiseOpParameter):
    ...


class SoftplusParameter(EntrywiseOpParameter):
    ...


class ScaledSigmoidParameter(EntrywiseOpParameter):
    def __init__(self, opd: AbstractParameter, vmin: float, vmax: float):
        super().__init__(opd)
        self.vmin = vmin
        self.vmax = vmax

    @property
    def hparams(self) -> Dict[str, Any]:
        return dict(vmin=self.vmin, vmax=self.vmax)


class SigmoidParameter(ScaledSigmoidParameter):
    def __init__(self, opd: AbstractParameter):
        super().__init__(opd, vmin=0.0, vmax=0.0)


class LogSoftmaxParameter(ReduceOpParameter):
    ...


class SoftmaxParameter(ReduceOpParameter):
    ...
