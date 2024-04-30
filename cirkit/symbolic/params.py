from abc import ABC, abstractmethod
from functools import cached_property
from numbers import Number
from typing import Any, Callable, Dict, Tuple


class AbstractParameter(ABC):
    @cached_property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def config(self) -> Dict[str, Any]:
        return {}


class Parameter(AbstractParameter):
    def __init__(self, *shape: int):
        self._shape = tuple(shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class ConstantParameter(AbstractParameter):
    def __init__(self, shape: Tuple[int, ...], value: Number):
        super().__init__()
        self.value = value
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class OpParameter(AbstractParameter, ABC):
    ...


Parameterization = Callable[[AbstractParameter], OpParameter]


class UnaryOpParameter(OpParameter, ABC):
    def __init__(self, opd: AbstractParameter) -> None:
        self.opd = opd


class BinaryOpParameter(OpParameter, ABC):
    def __init__(self, opd1: AbstractParameter, opd2: AbstractParameter) -> None:
        self.opd1 = opd1
        self.opd2 = opd2


class HadamardParameter(BinaryOpParameter):
    def __init__(self, opd1: AbstractParameter, opd2: AbstractParameter) -> None:
        assert opd1.shape == opd2.shape
        super().__init__(opd1, opd2)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.opd1.shape


class KroneckerParameter(BinaryOpParameter):
    def __init__(self, opd1: AbstractParameter, opd2: AbstractParameter) -> None:
        assert len(opd1.shape) == len(opd2.shape)
        super().__init__(opd1, opd2)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.opd1.shape[i] * self.opd2.shape[i] for i in range(len(self.opd1.shape)))


class OuterProductParameter(BinaryOpParameter):
    def __init__(self, opd1: AbstractParameter, opd2: AbstractParameter, axis: int = -1) -> None:
        super().__init__(opd1, opd2)
        assert len(opd1.shape) == len(opd2.shape)
        axis = axis if axis >= 0 else axis + len(opd1.shape)
        assert 0 <= axis < len(opd1.shape)
        assert opd1.shape[:axis] == opd2.shape[:axis]
        assert opd1.shape[axis + 1 :] == opd2.shape[axis + 1 :]
        self.p1 = opd1
        self.p2 = opd2
        self.axis = axis

    @property
    def config(self) -> Dict[str, Any]:
        return dict(axis=self.axis)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        cross_dim = self.p1.shape[self.axis] * self.p2.shape[self.axis]
        return *self.p1.shape[: self.axis], cross_dim, *self.p1.shape[self.axis + 1 :]


class OuterSumParameter(BinaryOpParameter):
    def __init__(self, opd1: AbstractParameter, opd2: AbstractParameter, axis: int = -1) -> None:
        super().__init__(opd1, opd2)
        assert len(opd1.shape) == len(opd2.shape)
        axis = axis if axis >= 0 else axis + len(opd1.shape)
        assert 0 <= axis < len(opd1.shape)
        self.p1 = opd1
        self.p2 = opd2
        self.axis = axis

    @property
    def config(self) -> Dict[str, Any]:
        return dict(axis=self.axis)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        cross_dim = self.p1.shape[self.axis] * self.p2.shape[self.axis]
        return *self.p1.shape[: self.axis], cross_dim, *self.p1.shape[self.axis + 1 :]


class EntrywiseOpParameter(UnaryOpParameter, ABC):
    def __init__(self, opd: AbstractParameter):
        super().__init__(opd)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.opd.shape


class ReduceOpParameter(UnaryOpParameter, ABC):
    def __init__(self, opd: AbstractParameter, axis: int = -1):
        super().__init__(opd)
        axis = axis if axis >= 0 else axis + len(opd.shape)
        assert 0 <= axis < len(opd.shape)
        self.axis = axis

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return *self.opd.shape[: self.axis], *self.opd.shape[self.axis + 1 :]

    @property
    def config(self) -> Dict[str, Any]:
        return dict(axis=self.axis)


class EntrywiseReduceOpParameter(EntrywiseOpParameter, ABC):
    def __init__(self, opd: AbstractParameter, axis: int = -1):
        super().__init__(opd)
        axis = axis if axis >= 0 else axis + len(opd.shape)
        assert 0 <= axis < len(opd.shape)
        self.axis = axis

    @property
    def config(self) -> Dict[str, Any]:
        return dict(axis=self.axis)


class ExpParameter(EntrywiseOpParameter):
    ...


class LogParameter(EntrywiseOpParameter):
    ...


class SoftplusParameter(EntrywiseOpParameter):
    ...


class ScaledSigmoidParameter(EntrywiseOpParameter):
    def __init__(self, opd: AbstractParameter, vmin: float, vmax: float):
        super().__init__(opd)
        self.vmin = vmin
        self.vmax = vmax

    @property
    def config(self) -> Dict[str, Any]:
        return dict(vmin=self.vmin, vmax=self.vmax)


class SigmoidParameter(ScaledSigmoidParameter):
    def __init__(self, opd: AbstractParameter):
        super().__init__(opd, vmin=0.0, vmax=0.0)


class ReduceSumParameter(ReduceOpParameter):
    ...


class ReduceProductParameter(ReduceOpParameter):
    ...


class ReduceLSEParameter(ReduceOpParameter):
    ...


class LogSoftmaxParameter(EntrywiseReduceOpParameter):
    ...


class SoftmaxParameter(EntrywiseReduceOpParameter):
    ...


class MeanNormalProduct(AbstractParameter):
    def __init__(
        self,
        mean1: AbstractParameter,
        mean2: AbstractParameter,
        variance1: AbstractParameter,
        variance2: AbstractParameter,
    ):
        assert mean1.shape[0] == mean2.shape[0] == variance1.shape[0] == variance2.shape[0]
        assert mean1.shape[2] == mean2.shape[2] == variance1.shape[2] == variance2.shape[2]
        assert mean1.shape[1] == variance1.shape[1] and mean2.shape[1] == variance2.shape[1]
        self.mean1 = mean1
        self.mean2 = mean2
        self.variance1 = variance1
        self.variance2 = variance2

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return self.mean1.shape[0], self.mean1.shape[1] * self.mean2.shape[1], self.mean1.shape[2]


class VarianceNormalProduct(AbstractParameter):
    def __init__(self, variance1: AbstractParameter, variance2: AbstractParameter):
        assert variance1.shape[0] == variance2.shape[0]
        assert variance1.shape[2] == variance2.shape[2]
        self.variance1 = variance1
        self.variance2 = variance2

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.variance1.shape[0],
            self.variance1.shape[1] * self.variance2.shape[1],
            self.variance1.shape[2],
        )


class PartitionGaussianProduct(AbstractParameter):
    def __init__(
        self,
        mean1: AbstractParameter,
        mean2: AbstractParameter,
        variance1: AbstractParameter,
        variance2: AbstractParameter,
    ):
        assert mean1.shape[0] == mean2.shape[0] == variance1.shape[0] == variance2.shape[0]
        assert mean1.shape[2] == mean2.shape[2] == variance1.shape[2] == variance2.shape[2]
        assert mean1.shape[1] == variance1.shape[1] and mean2.shape[1] == variance2.shape[1]
        self.mean1 = mean1
        self.mean2 = mean2
        self.variance1 = variance1
        self.variance2 = variance2

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return (self.mean1.shape[1] * self.mean2.shape[1],)
