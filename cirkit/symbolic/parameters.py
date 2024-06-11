from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy as shallowcopy
from functools import cached_property
from numbers import Number
from typing import Any, Callable, Dict, List, Tuple, final, Union

from cirkit.utils.algorithms import RootedDiAcyclicGraph


class ParameterNode(ABC):
    @abstractmethod
    def __copy__(self) -> "ParameterLeaf":
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
    def __init__(self, *shape: int, learnable: bool = True):
        super().__init__()
        self._shape = tuple(shape)
        self.learnable = learnable

    def __copy__(self) -> "TensorParameter":
        cls = self.__class__
        return cls(*self._shape, learnable=self.learnable)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def config(self) -> Dict[str, Any]:
        return dict(learnable=self.learnable)


class ConstantParameter(TensorParameter):
    def __init__(self, *shape: int, value: Number = 0.0):
        super().__init__(*shape, learnable=False)
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
    def __init__(
        self,
        nodes: List[ParameterNode],
        in_nodes: Dict[ParameterNode, List[ParameterNode]],
        out_nodes: Dict[ParameterNode, List[ParameterNode]],
        *,
        topologically_ordered: bool = False,
    ) -> None:
        super().__init__(nodes, in_nodes, out_nodes, topologically_ordered=topologically_ordered)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.output.shape

    @classmethod
    def from_leaf(cls, p: ParameterLeaf) -> "Parameter":
        return Parameter([p], {}, {}, topologically_ordered=True)

    @classmethod
    def from_sequence(cls, p: Union[ParameterLeaf, "Parameter"], *ns: ParameterNode) -> "Parameter":
        if isinstance(p, ParameterLeaf):
            p = Parameter.from_leaf(p)
        nodes = p.nodes + list(ns)
        in_nodes = dict(p.nodes_inputs)
        out_nodes = dict(p.nodes_outputs)
        for i, n in enumerate(ns):
            in_nodes[n] = [ns[i - 1]] if i - 1 >= 0 else [p.output]
            out_nodes[n] = [ns[i + 1]] if i + 1 < len(ns) else []
        out_nodes[p.output] = [ns[0]]
        return Parameter(
            nodes, in_nodes, out_nodes, topologically_ordered=p.is_topologically_ordered
        )

    @classmethod
    def from_unary(cls, p: Union[ParameterLeaf, "Parameter"], n: UnaryParameterOp) -> "Parameter":
        return Parameter.from_sequence(p, n)

    @classmethod
    def from_binary(cls, p1: Union[ParameterLeaf, "Parameter"], p2: Union[ParameterLeaf, "Parameter"], n: BinaryParameterOp) -> "Parameter":
        if isinstance(p1, ParameterLeaf):
            p1 = Parameter.from_leaf(p1)
        if isinstance(p2, ParameterLeaf):
            p2 = Parameter.from_leaf(p2)
        p_nodes = p1.nodes + p2.nodes + [n]
        in_nodes = {**p1.nodes_inputs, **p2.nodes_inputs}
        out_nodes = {**p1.nodes_outputs, **p2.nodes_outputs}
        in_nodes[n] = [p1.output, p2.output]
        out_nodes[p1.output] = [n]
        out_nodes[p2.output] = [n]
        return Parameter(
            p_nodes,
            in_nodes,
            out_nodes,
            topologically_ordered=p1.is_topologically_ordered and p2.is_topologically_ordered,
        )

    def copy(self) -> "Parameter":
        # Build a new symbolic parameter's computational graph, by coping nodes.
        def replace_copy(p: ParameterNode) -> ParameterNode:
            return shallowcopy(p)

        return self._process_nodes(replace_copy)

    def ref(self) -> "Parameter":
        # Build a new symbolic parameter's computational graph, where the parameter tensors
        # become references to the tensors of the 'self' parameter's computational graph.
        # All the other nodes are new objects.
        def replace_ref_or_copy(p: ParameterNode) -> ParameterNode:
            return ReferenceParameter(p) if isinstance(p, TensorParameter) else shallowcopy(p)

        return self._process_nodes(replace_ref_or_copy)

    def _process_nodes(self, process_fn: Callable[[ParameterNode], ParameterNode]) -> "Parameter":
        nodes_map = {}
        in_nodes = {}
        out_nodes = defaultdict(list)
        for p in self.topological_ordering():
            new_p = process_fn(p)
            nodes_map[p] = new_p
            in_new_nodes = [nodes_map[in_p] for in_p in self.node_inputs(p)]
            in_nodes[new_p] = in_new_nodes
            for in_p in in_new_nodes:
                out_nodes[in_p].append(new_p)
        nodes = [nodes_map[p] for p in nodes_map.keys()]
        return Parameter(nodes, in_nodes, out_nodes, topologically_ordered=True)


Parameterization = Callable[[TensorParameter], Parameter]
