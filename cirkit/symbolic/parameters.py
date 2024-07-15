import operator
from abc import ABC, abstractmethod
from copy import copy as shallowcopy
from functools import cached_property, reduce
from itertools import chain
from numbers import Number
from typing import Any, Callable, Dict, Optional, Tuple, Union, final

from cirkit.symbolic.initializers import ConstantInitializer, Initializer
from cirkit.utils.algorithms import RootedDiAcyclicGraph, topologically_process_nodes


class ParameterNode(ABC):
    @abstractmethod
    def __copy__(self) -> "ParameterNode":
        ...

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def config(self) -> Dict[str, Any]:
        return {}


class ParameterLeaf(ParameterNode, ABC):
    ...


class TensorParameter(ParameterLeaf):
    def __init__(self, *shape: int, initializer: Initializer, learnable: bool = True):
        super().__init__()
        self._shape = tuple(shape)
        self.initializer = initializer
        self.learnable = learnable

    def __copy__(self) -> "TensorParameter":
        cls = self.__class__
        return cls(*self._shape, **self.config)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def config(self) -> Dict[str, Any]:
        return dict(initializer=self.initializer, learnable=self.learnable)


class ConstantParameter(TensorParameter):
    def __init__(self, *shape: int, value: Number = 0.0):
        super().__init__(*shape, initializer=ConstantInitializer(value), learnable=False)
        self.value = value

    def __copy__(self) -> "ConstantParameter":
        cls = self.__class__
        return cls(*self._shape, value=self.value)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(value=self.value)


@final
class ReferenceParameter(ParameterLeaf):
    def __init__(self, parameter: TensorParameter):
        super().__init__()
        self._parameter = parameter

    def __copy__(self) -> "ReferenceParameter":
        cls = self.__class__
        return cls(self._parameter)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._parameter.shape

    def deref(self) -> TensorParameter:
        return self._parameter


class ParameterOp(ParameterNode, ABC):
    def __init__(self, *in_shape: Tuple[int, ...], **kwargs):
        self.in_shapes = in_shape

    def __copy__(self) -> "ParameterOp":
        cls = self.__class__
        return cls(*self.in_shapes, **self.config)


class UnaryParameterOp(ParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...]):
        super().__init__(in_shape)


class BinaryParameterOp(ParameterOp, ABC):
    def __init__(self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...]):
        super().__init__(in_shape1, in_shape2)


class EntrywiseOpParameter(UnaryParameterOp, ABC):
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]


class ReduceOpParameter(UnaryParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...], *, axis: int = -1):
        assert 0 <= axis < len(in_shape)
        super().__init__(in_shape)
        self.axis = axis if axis >= 0 else axis + len(in_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return *self.in_shapes[0][: self.axis], *self.in_shapes[0][self.axis + 1 :]

    @property
    def config(self) -> Dict[str, Any]:
        return dict(axis=self.axis)


class EntrywiseReduceOpParameter(EntrywiseOpParameter, ABC):
    def __init__(self, in_shape: Tuple[int, ...], *, axis: int = -1):
        super().__init__(in_shape)
        axis = axis if axis >= 0 else axis + len(in_shape)
        assert 0 <= axis < len(in_shape)
        self.axis = axis

    @property
    def config(self) -> Dict[str, Any]:
        return dict(axis=self.axis)


class SumParameter(BinaryParameterOp):
    def __init__(self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...]) -> None:
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]


class HadamardParameter(BinaryParameterOp):
    def __init__(self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...]):
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]


class KroneckerParameter(BinaryParameterOp):
    def __init__(self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...]):
        assert len(in_shape1) == len(in_shape2)
        super().__init__(in_shape1, in_shape2)

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(
            self.in_shapes[0][i] * self.in_shapes[1][i] for i in range(len(self.in_shapes[0]))
        )


class OuterOpParameter(BinaryParameterOp):
    def __init__(
        self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...], *, axis: int = -1
    ) -> None:
        assert len(in_shape1) == len(in_shape2)
        axis = axis if axis >= 0 else axis + len(in_shape1)
        assert 0 <= axis < len(in_shape1)
        assert in_shape1[:axis] == in_shape1[:axis]
        assert in_shape1[axis + 1 :] == in_shape1[axis + 1 :]
        super().__init__(in_shape1, in_shape2)
        self.axis = axis

    @property
    def shape(self) -> Tuple[int, ...]:
        cross_dim = self.in_shapes[0][self.axis] * self.in_shapes[1][self.axis]
        return *self.in_shapes[0][: self.axis], cross_dim, *self.in_shapes[0][self.axis + 1 :]

    @property
    def config(self) -> Dict[str, Any]:
        return dict(axis=self.axis)


class OuterProductParameter(OuterOpParameter):
    ...


class OuterSumParameter(OuterOpParameter):
    ...


class ExpParameter(EntrywiseOpParameter):
    ...


class LogParameter(EntrywiseOpParameter):
    ...


class SquareParameter(EntrywiseOpParameter):
    ...


class SoftplusParameter(EntrywiseOpParameter):
    ...


class SigmoidParameter(EntrywiseOpParameter):
    ...


class ScaledSigmoidParameter(EntrywiseOpParameter):
    def __init__(self, in_shape: Tuple[int, ...], vmin: float, vmax: float):
        super().__init__(in_shape)
        self.vmin = vmin
        self.vmax = vmax

    @property
    def config(self) -> Dict[str, Any]:
        return dict(vmin=self.vmin, vmax=self.vmax)


class ClampParameter(EntrywiseOpParameter):
    """Exp reparameterization."""

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        assert vmin is not None or vmax is not None
        super().__init__(in_shape)
        self.vmin = vmin
        self.vmax = vmax

    @property
    def config(self) -> Dict[str, Any]:
        config = dict()
        if self.vmin is not None:
            config.update(vmin=self.vmin)
        if self.vmax is not None:
            config.update(vmax=self.vmax)
        return config


class ReduceSumParameter(ReduceOpParameter):
    ...


class ReduceProductParameter(ReduceOpParameter):
    ...


class ReduceLSEParameter(ReduceOpParameter):
    ...


class SoftmaxParameter(EntrywiseReduceOpParameter):
    ...


class LogSoftmaxParameter(EntrywiseReduceOpParameter):
    ...


class Parameter(RootedDiAcyclicGraph[ParameterNode]):
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.output.shape

    @classmethod
    def from_leaf(cls, p: ParameterLeaf) -> "Parameter":
        return Parameter([p], {}, [p], topologically_ordered=True)

    @classmethod
    def from_sequence(cls, p: Union[ParameterLeaf, "Parameter"], *ns: ParameterNode) -> "Parameter":
        if isinstance(p, ParameterLeaf):
            p = Parameter.from_leaf(p)
        nodes = p.nodes + list(ns)
        in_nodes = dict(p.nodes_inputs)
        for i, n in enumerate(ns):
            in_nodes[n] = [ns[i - 1]] if i - 1 >= 0 else [p.output]
        return Parameter(
            nodes, in_nodes, [ns[-1]], topologically_ordered=p.is_topologically_ordered
        )

    @classmethod
    def from_nary(cls, n: ParameterOp, *ps: Union[ParameterLeaf, "Parameter"]) -> "Parameter":
        ps = tuple(Parameter.from_leaf(p) if isinstance(p, ParameterLeaf) else p for p in ps)
        p_nodes = list(chain.from_iterable(p.nodes for p in ps)) + [n]
        in_nodes = reduce(operator.ior, (p.nodes_inputs for p in ps), {})
        in_nodes[n] = list(p.output for p in ps)
        topologically_ordered = all(p.is_topologically_ordered for p in ps)
        return Parameter(
            p_nodes,
            in_nodes,
            [n],
            topologically_ordered=topologically_ordered,
        )

    @classmethod
    def from_unary(cls, n: UnaryParameterOp, p: Union[ParameterLeaf, "Parameter"]) -> "Parameter":
        return Parameter.from_sequence(p, n)

    @classmethod
    def from_binary(
        cls,
        n: BinaryParameterOp,
        p1: Union[ParameterLeaf, "Parameter"],
        p2: Union[ParameterLeaf, "Parameter"],
    ) -> "Parameter":
        return Parameter.from_nary(n, p1, p2)

    def copy(self) -> "Parameter":
        # Build a new symbolic parameter's computational graph, by coping nodes.
        def replace_copy(n: ParameterNode) -> ParameterNode:
            return shallowcopy(n)

        return self._process_nodes(replace_copy)

    def ref(self) -> "Parameter":
        # Build a new symbolic parameter's computational graph, where the parameter tensors
        # become references to the tensors of the 'self' parameter's computational graph.
        # All the other nodes are new objects.
        def replace_ref_or_copy(n: ParameterNode) -> ParameterNode:
            return ReferenceParameter(n) if isinstance(n, TensorParameter) else shallowcopy(n)

        return self._process_nodes(replace_ref_or_copy)

    def _process_nodes(self, process_fn: Callable[[ParameterNode], ParameterNode]) -> "Parameter":
        nodes, in_nodes, outputs = topologically_process_nodes(
            self.topological_ordering(), self.outputs, process_fn, incomings_fn=self.node_inputs
        )
        return Parameter(nodes, in_nodes, outputs, topologically_ordered=True)


Parameterization = Callable[[TensorParameter], Parameter]


class GaussianProductMean(ParameterOp):
    def __init__(
        self, in_gaussian1_shape: Tuple[int, ...], in_gaussian2_shape: Tuple[int, ...]
    ) -> None:
        assert (
            in_gaussian1_shape[0] == in_gaussian2_shape[0]
            and in_gaussian1_shape[2] == in_gaussian2_shape[2]
        )
        super().__init__(in_gaussian1_shape, in_gaussian2_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0],
            self.in_shapes[0][1] * self.in_shapes[1][1],
            self.in_shapes[0][2],
        )


class GaussianProductStddev(BinaryParameterOp):
    def __init__(
        self, in_gaussian1_shape: Tuple[int, ...], in_gaussian2_shape: Tuple[int, ...]
    ) -> None:
        assert (
            in_gaussian1_shape[0] == in_gaussian2_shape[0]
            and in_gaussian1_shape[2] == in_gaussian2_shape[2]
        )
        super().__init__(in_gaussian1_shape, in_gaussian2_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0],
            self.in_shapes[0][1] * self.in_shapes[1][1],
            self.in_shapes[0][2],
        )


class GaussianProductLogPartition(ParameterOp):
    def __init__(
        self, in_gaussian1_shape: Tuple[int, ...], in_gaussian2_shape: Tuple[int, ...]
    ) -> None:
        assert (
            in_gaussian1_shape[0] == in_gaussian2_shape[0]
            and in_gaussian1_shape[2] == in_gaussian2_shape[2]
        )
        super().__init__(in_gaussian1_shape, in_gaussian2_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0],
            self.in_shapes[0][1] * self.in_shapes[1][1],
            self.in_shapes[0][2],
        )
