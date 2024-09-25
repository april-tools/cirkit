from abc import ABC, abstractmethod
from collections import ChainMap
from functools import cached_property
from itertools import chain
from numbers import Number
from typing import List, Optional, Protocol, Tuple, Union

import numpy as np

from cirkit.symbolic.dtypes import DataType, dtype_value
from cirkit.symbolic.initializers import ConstantTensorInitializer, Initializer
from cirkit.utils.algorithms import RootedDiAcyclicGraph


class ParameterNode(ABC):
    """The abstract parameter node class. A parameter node is a node in the computational
    graph that computes parameters. See [Parameter][cirkit.symbolic.parameters.Parameter]
    for more details."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Retrieves the shape of the output of the parameter node.

        Returns:
            The shape of the output.
        """


class ParameterInput(ParameterNode, ABC):
    """The abstract parameter input class. A parameter input is a parameter node in the
    computational graph that comptues parameter that does __not__ have inputs. See
    [Parameter][cirkit.symbolic.parameters.Parameter] for more details."""


class TensorParameter(ParameterInput):
    """A symbolic tensor parameter is an object storing information about a dense
    tensor parameter, i.e., its shape, its initialization method, whether it is
    learnable (or if it is a constant tensor), and its data type. Note that the
    symbolic tensor parmater does __not__ allocate any tensor, but only stores
    the mentioned information about it."""

    def __init__(
        self,
        *shape: int,
        initializer: Initializer,
        learnable: bool = True,
        dtype: DataType = DataType.REAL,
    ):
        """Initializes a symbolic tensor parameter.

        Args:
            *shape: The shape of the tensor parameter.
            initializer: The initializer object.
            learnable: Whether the tensor parameter is learnable or not.
            dtype: The data type.

        Raises:
            ValueError: If the shape contains dimensions that are not positive.
        """
        super().__init__()
        if any(d <= 0 for d in shape):
            raise ValueError(f"The given shape {shape} is not valid")
        self._shape = tuple(shape)
        self.initializer = initializer
        self.learnable = learnable
        self.dtype = dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape


class ConstantParameter(TensorParameter):
    def __init__(self, *shape: int, value: Union[Number, np.ndarray] = 0.0):
        initializer = ConstantTensorInitializer(value)
        super().__init__(
            *shape,
            initializer=initializer,
            learnable=False,
            dtype=dtype_value(value),
        )
        self.value = value


class ParameterOp(ParameterNode, ABC):
    def __init__(self, *in_shape: Tuple[int, ...], **kwargs):
        self.in_shapes = in_shape


class UnaryParameterOp(ParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...]):
        super().__init__(in_shape)


class BinaryParameterOp(ParameterOp, ABC):
    def __init__(self, in_shape1: Tuple[int, ...], in_shape2: Tuple[int, ...]):
        super().__init__(in_shape1, in_shape2)


class EntrywiseParameterOp(UnaryParameterOp, ABC):
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.in_shapes[0]


class ReduceParameterOp(UnaryParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...], *, axis: int = -1):
        assert 0 <= axis < len(in_shape)
        super().__init__(in_shape)
        self.axis = axis if axis >= 0 else axis + len(in_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return *self.in_shapes[0][: self.axis], *self.in_shapes[0][self.axis + 1 :]


class EntrywiseReduceParameterOp(EntrywiseParameterOp, ABC):
    def __init__(self, in_shape: Tuple[int, ...], *, axis: int = -1):
        super().__init__(in_shape)
        axis = axis if axis >= 0 else axis + len(in_shape)
        assert 0 <= axis < len(in_shape)
        self.axis = axis


class IndexParameter(UnaryParameterOp):
    def __init__(self, in_shape: Tuple[int, ...], *, indices: List[int], axis: int = -1):
        super().__init__(in_shape)
        axis = axis if axis >= 0 else axis + len(in_shape)
        assert 0 <= axis < len(in_shape)
        assert all(0 <= i < in_shape[axis] for i in indices)
        self.indices = indices
        self.axis = axis

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            *self.in_shapes[0][: self.axis],
            len(self.indices),
            *self.in_shapes[0][self.axis + 1 :],
        )


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


class OuterParameterOp(BinaryParameterOp):
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


class OuterProductParameter(OuterParameterOp):
    ...


class OuterSumParameter(OuterParameterOp):
    ...


class ExpParameter(EntrywiseParameterOp):
    ...


class LogParameter(EntrywiseParameterOp):
    ...


class SquareParameter(EntrywiseParameterOp):
    ...


class SoftplusParameter(EntrywiseParameterOp):
    ...


class SigmoidParameter(EntrywiseParameterOp):
    ...


class ScaledSigmoidParameter(EntrywiseParameterOp):
    def __init__(self, in_shape: Tuple[int, ...], vmin: float, vmax: float):
        super().__init__(in_shape)
        self.vmin = vmin
        self.vmax = vmax


class ClampParameter(EntrywiseParameterOp):
    """Clamp reparameterization."""

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


class ConjugateParameter(EntrywiseParameterOp):
    ...


class ReduceSumParameter(ReduceParameterOp):
    ...


class ReduceProductParameter(ReduceParameterOp):
    ...


class ReduceLSEParameter(ReduceParameterOp):
    ...


class SoftmaxParameter(EntrywiseReduceParameterOp):
    ...


class LogSoftmaxParameter(EntrywiseReduceParameterOp):
    ...


class Parameter(RootedDiAcyclicGraph[ParameterNode]):
    """The symbolic parameter computational graph. A symbolic parameter is a computational graph
    consisting of symbolic nodes, which represent how to compute a tensor parameter."""

    @property
    def shape(self) -> Tuple[int, ...]:
        """Retrieves the shape of the output tensor.

        Returns:
            The shape of the output of the computational graph.
        """
        return self.output.shape

    @classmethod
    def from_leaf(cls, p: ParameterInput) -> "Parameter":
        """Constructs a parameter from a leaf symbolic node only.

        Args:
            p: The symbolic parameter input.

        Returns:
            A symbolic parameter encapsulating the given parameter input.
        """
        return Parameter([p], {}, [p])

    @classmethod
    def from_sequence(
        cls, p: Union[ParameterInput, "Parameter"], *ns: ParameterNode
    ) -> "Parameter":
        """Constructs a parameter from a composition of symbolic parameter nodes.

        Args:
            p: The entry point of the sequence, which can be either a symbolic parameter
                input or another symbolic parameter.
            *ns: A sequence of symbolic parameter nodes.

        Returns:
            A symbolic parameter that encodes the composition of the symbolic parameter nodes,
                starting from the given entry point of the sequence.
        """
        if isinstance(p, ParameterInput):
            p = Parameter.from_leaf(p)
        nodes = list(p.nodes) + list(ns)
        in_nodes = dict(p.nodes_inputs)
        for i, n in enumerate(ns):
            in_nodes[n] = [ns[i - 1]] if i - 1 >= 0 else [p.output]
        return Parameter(nodes, in_nodes, [ns[-1]])

    @classmethod
    def from_nary(cls, n: ParameterOp, *ps: Union[ParameterInput, "Parameter"]) -> "Parameter":
        """Constructs a parameter by using a parameter operation node and by specifying its inputs.

        Args:
            n: The parameter operation node.
            *ps: A sequence of symbolic parameter input nodes or parameters.

        Returns:
            A symbolic parameter that encodes the application of the given parameter operation node
                to the outputs given by the symbolic parameter input nodes or parameters.
        """
        ps = tuple(Parameter.from_leaf(p) if isinstance(p, ParameterInput) else p for p in ps)
        p_nodes = list(chain.from_iterable(p.nodes for p in ps)) + [n]
        in_nodes = dict(ChainMap(*(p.nodes_inputs for p in ps)))
        in_nodes[n] = list(p.output for p in ps)
        return Parameter(p_nodes, in_nodes, [n])

    @classmethod
    def from_unary(cls, n: UnaryParameterOp, p: Union[ParameterInput, "Parameter"]) -> "Parameter":
        """Constructs a parameter by using a unary parameter operation node and by specifying its
        inputs.

        Args:
            n: The unary parameter operation node.
            p: The symbolic parameter input node, or another parameter.

        Returns:
            A symbolic parameter that encodes the application of the given parameter operation
                node to the output given by the symbolic parameter input node or parameter.
        """
        return Parameter.from_sequence(p, n)

    @classmethod
    def from_binary(
        cls,
        n: BinaryParameterOp,
        p1: Union[ParameterInput, "Parameter"],
        p2: Union[ParameterInput, "Parameter"],
    ) -> "Parameter":
        """Constructs a parameter by using a binary parameter operation node and by specifying
        its inputs.

        Args:
            n: The binary parameter operation node.
            p1: The first symbolic parameter input node, or another parameter.
            p2: The second symbolic parameter input node, or another parameter.

        Returns:
            A symbolic parameter that encodes the application of the given parameter operation
                node to the two outputs given by the symbolic parameter inputs or parameters.
        """
        return Parameter.from_nary(n, p1, p2)


class ParameterFactory(Protocol):
    """A factory that constucts symbolic parameter given a shape."""

    def __call__(self, shape: Tuple[int, ...]) -> Parameter:
        """Constructs a symbolic parameter given the parameter shape.

        Args:
            shape: The shape.

        Returns:
            A parameter whose output shape is equal to the given shape.
        """


class GaussianProductMean(ParameterOp):
    def __init__(
        self, in_gaussian1_shape: Tuple[int, ...], in_gaussian2_shape: Tuple[int, ...]
    ) -> None:
        assert in_gaussian1_shape[1] == in_gaussian2_shape[1]
        super().__init__(in_gaussian1_shape, in_gaussian2_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0] * self.in_shapes[1][0],
            self.in_shapes[0][1],
        )


class GaussianProductStddev(BinaryParameterOp):
    def __init__(
        self, in_gaussian1_shape: Tuple[int, ...], in_gaussian2_shape: Tuple[int, ...]
    ) -> None:
        assert in_gaussian1_shape[1] == in_gaussian2_shape[1]
        super().__init__(in_gaussian1_shape, in_gaussian2_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0] * self.in_shapes[1][0],
            self.in_shapes[0][1],
        )


class GaussianProductLogPartition(ParameterOp):
    def __init__(
        self, in_gaussian1_shape: Tuple[int, ...], in_gaussian2_shape: Tuple[int, ...]
    ) -> None:
        assert in_gaussian1_shape[1] == in_gaussian2_shape[1]
        super().__init__(in_gaussian1_shape, in_gaussian2_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return (
            self.in_shapes[0][0] * self.in_shapes[1][0],
            self.in_shapes[0][1],
        )
