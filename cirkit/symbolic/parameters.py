from abc import ABC, abstractmethod
from collections import ChainMap
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from functools import cached_property
from itertools import chain
from numbers import Number
from typing import Any, Protocol, Union

import numpy as np

from cirkit.symbolic.dtypes import DataType, dtype_value
from cirkit.symbolic.initializers import ConstantTensorInitializer, Initializer
from cirkit.utils.algorithms import RootedDiAcyclicGraph, topologically_process_nodes


class ParameterNode(ABC):
    """The abstract parameter node class. A parameter node is a node in the computational
    graph that computes parameters. See [Parameter][cirkit.symbolic.parameters.Parameter]
    for more details."""

    def __copy__(self) -> "ParameterNode":
        """The shallow copy operation of a parameter node.

        Returns:
            The copy of the parameter node.
        """
        cls = self.__class__
        return cls(**self.config)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Retrieves the shape of the output of the parameter node.

        Returns:
            The shape of the output.
        """

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Retrieves the configuration of the parameter node, i.e., a dictionary mapping
        hyperparameters of the parameter node to their values. The hyperparameter names must
        match the argument names in the ```__init__``` method.

        Returns:
            Dict[str, Any]: A dictionary from hyperparameter names to their value.
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
            ValueError: If the shape is empty or contains dimensions that are not positive.
            ValueError: If the initializer does not allow the parameter shape.
        """
        super().__init__()
        if len(shape) < 1 or any(d <= 0 for d in shape):
            raise ValueError(
                f"The shape {shape} must be non-empty and have positive dimension sizes"
            )
        if not initializer.allows_shape(shape):
            raise ValueError(f"The shape {shape} is not valid for the initializer {initializer}")
        self._shape = shape
        self.initializer = initializer
        self.learnable = learnable
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def config(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "initializer": self.initializer,
            "learnable": self.learnable,
            "dtype": self.dtype,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self._shape}, "
            f"learnable={self.learnable}, "
            f"dtype={self.dtype}, "
            f"initializer={self.initializer}"
            ")"
        )


class ConstantParameter(TensorParameter):
    """A symbolic tensor parameter representing a constant tensor, i.e.,
    a tensor that is not learnable.
    """

    def __init__(self, *shape: int, value: Number | np.ndarray = 0.0):
        """Initializes a constant symbolic parameter.

        Args:
            *shape: The shape of the parameter.
            value: The values stored in the parameter. It can be either a number or
                a Numpy array having the same shape of the given one.

        Raises:
            ValueError: If the given value is a Numpy array having a different shape
                than the given one.
        """
        if isinstance(value, np.ndarray) and value.shape != shape:
            raise ValueError("The shape of the Numpy array is not equal to the given shape")
        initializer = ConstantTensorInitializer(value)
        super().__init__(
            *shape,
            initializer=initializer,
            learnable=False,
            dtype=dtype_value(value),
        )
        self.value = value

    @property
    def config(self) -> dict[str, Any]:
        return {"shape": self.shape, "value": self.value}


class GateFunctionParameter(ParameterInput):
    """A symbolic parameter whose value is computed by an externally-provided function.
    For example, this function can be the evaluation of a neural network.
    """

    def __init__(self, *shape: int, name: str, index: int):
        """Initialize a symbolic function parameter.

        Args:
            shape: The shape of the parameter tensor, which is specified by the combination
                of the arguments "function", "name" and "index" below.
            name: The gate function name. This is a string identifier of the function that
                will output a mapping between parameter tensor identifiers and their value.
            index: An index that selects the parameter tensor specified by the name.
                That is, this index will be used to slice the parameter tensor
                along the first dimension.

        Raises:
            ValueError: If the shape is not valid.
            ValueError: If the index is a negative integer.
        """
        if len(shape) < 1 or any(d <= 0 for d in shape):
            raise ValueError(
                f"The shape {shape} must be non-empty and have positive dimension sizes"
            )
        if index is not None and index < 0:
            raise ValueError("The index must be a non-negative integer")
        super().__init__()
        self._shape = shape
        self._name = name
        self._index = index

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def name(self) -> str:
        return self._name

    @property
    def index(self) -> int:
        return self._index

    @property
    def config(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "name": self.name,
            "index": self.index,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"name='{self.name}', "
            f"index={self.index}"
            ")"
        )


class ReferenceParameter(ParameterInput):
    """A symbolic reference parameter representing a symbolic link to a tensor parameter."""

    def __init__(self, parameter: TensorParameter):
        """Initializes a reference parameter.

        Args:
            parameter: The tensor parameter to point to.
        """
        super().__init__()
        self._parameter = parameter

    @property
    def shape(self) -> tuple[int, ...]:
        return self._parameter.shape

    @property
    def config(self) -> dict[str, Any]:
        return {"parameter": self._parameter}

    def deref(self) -> TensorParameter:
        """Dereference the pointer.

        Returns:
            The tensor paramter being pointed to.
        """
        return self._parameter


class ParameterOp(ParameterNode, ABC):
    """A symbolic parameter operator, i.e., an inner node of the symbolic parameter
    computational graph.
    """

    def __init__(self, *in_shapes: tuple[int, ...]):
        """Initializes a symbolic parameter operator.

        Args:
            *in_shapes: A sequence of the shapes of each input. The length of this
                sequence is the arity of the parameter operator.
        """
        self._in_shapes = in_shapes

    @property
    def in_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Retrieves the shapes of the inputs to the parameter operator.

        Returns:
            A sequence of shapes, one for each input.
        """
        return self._in_shapes


class UnaryParameterOp(ParameterOp, ABC):
    """A symbolic parameter operator that represents a unary operation."""

    def __init__(self, in_shape: tuple[int, ...]):
        """Initializes a symbolic unary parameter operator.

        Args:
            in_shape: The shape of the input.
        """
        super().__init__(in_shape)

    @property
    def in_shape(self) -> tuple[int, ...]:
        """Retrieves the shape of the input.

        Returns:
            The shape of the input.
        """
        return self._in_shapes[0]

    @property
    def config(self) -> dict[str, Any]:
        return {"in_shape": self.in_shape}


class BinaryParameterOp(ParameterOp, ABC):
    """A symbolic parameter operator that represents a binary operation."""

    def __init__(self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...]):
        """Initializes a symbolic binary parameter operator.

        Args:
            in_shape1: The shape of the first input.
            in_shape2: The shape of the second input.
        """
        super().__init__(in_shape1, in_shape2)

    @property
    def in_shape1(self) -> tuple[int, ...]:
        """Retrieves the shape of the first input.

        Returns:
            The shape of the first input.
        """
        return self._in_shapes[0]

    @property
    def in_shape2(self) -> tuple[int, ...]:
        """Retrieves the shape of the second input.

        Returns:
            The shape of the second input.
        """
        return self._in_shapes[1]

    @property
    def config(self) -> dict[str, Any]:
        return {"in_shape1": self.in_shape1, "in_shape2": self.in_shape2}


class EntrywiseParameterOp(UnaryParameterOp, ABC):
    """A symbolic parameter operator that represents a unary function
    that is applied entrywise to the input parameter tensor.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape


class ReduceParameterOp(UnaryParameterOp, ABC):
    """A symbolic parameter operator that represents a unary function
    that is a reduction along one dimension of the input parameter tensor.
    """

    def __init__(self, in_shape: tuple[int, ...], *, axis: int = -1):
        """Initializes a symbolic reduce parameter operator.

        Args:
            in_shape: The shape of the input.
            axis: The axis of the input being reduced.
        """
        axis = axis if axis >= 0 else axis + len(in_shape)
        super().__init__(in_shape)
        self._axis = axis

    @property
    def axis(self) -> int:
        """Retrieves the axis of the input being reduced.

        Returns:
            The axis dimension.
        """
        return self._axis

    @property
    def shape(self) -> tuple[int, ...]:
        return *self.in_shape[: self.axis], *self.in_shape[self.axis + 1 :]

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["axis"] = self.axis
        return config


class EntrywiseReduceParameterOp(EntrywiseParameterOp, ABC):
    """A symbolic parameter operator that represents a unary function
    that is applied entrywise to the input parameter tensor,
    and this unary function is obtained by reducing one input dimension.
    """

    def __init__(self, in_shape: tuple[int, ...], *, axis: int = -1):
        """Initializes an entrywise reduce parameter operator.

        Args:
            in_shape: The shape of the input.
            axis: The axis of the input being reduced.
        """
        super().__init__(in_shape)
        axis = axis if axis >= 0 else axis + len(in_shape)
        assert 0 <= axis < len(in_shape)
        self._axis = axis

    @property
    def axis(self) -> int:
        """Retrieves the axis of the input being reduced.

        Returns:
            The axis dimension.
        """
        return self._axis

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["axis"] = self.axis
        return config


class IndexParameter(UnaryParameterOp):
    """A symbolic parameter operator that indexes an input parameter tensor
    along one axis.
    """

    def __init__(self, in_shape: tuple[int, ...], *, indices: list[int], axis: int = -1):
        """Initializes a symbolc index parameter.

        Args:
            in_shape: The shape of the input.
            indices: The indices to index.
            axis: The axis of the input tensor being indexed.
        """
        super().__init__(in_shape)
        axis = axis if axis >= 0 else axis + len(in_shape)
        assert 0 <= axis < len(in_shape)
        assert all(0 <= i < in_shape[axis] for i in indices)
        self._indices = indices
        self._axis = axis

    @property
    def indices(self) -> list[int]:
        """Retrieves the indices.

        Returns:
            The indices.
        """
        return self._indices

    @property
    def axis(self) -> int:
        """Retrieves the axis being indexed.

        Returns:
            The axis.
        """
        return self._axis

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            *self.in_shape[: self.axis],
            len(self.indices),
            *self.in_shape[self.axis + 1 :],
        )

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["indices"] = self.indices
        config["axis"] = self.axis
        return config


class SumParameter(BinaryParameterOp):
    """A symbolic parameter operator representing the element-wise sum of its inputs."""

    def __init__(self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...]) -> None:
        """Initializes a symbolic sum parameter operator.

        Args:
            in_shape1: The shape of the first input.
            in_shape2: The shape of the second input.
        """
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape1


class HadamardParameter(BinaryParameterOp):
    """A symbolic parameter operator representing the element-wise product of its inputs."""

    def __init__(self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...]):
        """Initializes a symbolic Hadamard parameter operator.

        Args:
            in_shape1: The shape of the first input.
            in_shape2: The shape of the second input.
        """
        assert in_shape1 == in_shape2
        super().__init__(in_shape1, in_shape2)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape1


class KroneckerParameter(BinaryParameterOp):
    """A symbolic parameter operator representing the Kronecker product of its inputs."""

    def __init__(self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...]):
        """Initializes a symbolic Kronecker parameter operator.

        Args:
            in_shape1: The shape of the first input.
            in_shape2: The shape of the second input.
        """
        assert len(in_shape1) == len(in_shape2)
        super().__init__(in_shape1, in_shape2)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.in_shape1[i] * self.in_shape2[i] for i in range(len(self.in_shape1)))


class OuterParameterOp(BinaryParameterOp):
    """A symbolic parameter operator computing a function applied over all possible combinations
    of entries along one axis of the input tensors.
    """

    def __init__(self, in_shape1: tuple[int, ...], in_shape2: tuple[int, ...], *, axis: int = -1):
        """Initializes a symbolic parameter outer operator.

        Args:
            in_shape1: The shape of the first input.
            in_shape2: The shape of the second input.
            axis: The axis of both inputs along which the function is applied.
        """
        assert len(in_shape1) == len(in_shape2)
        axis = axis if axis >= 0 else axis + len(in_shape1)
        assert 0 <= axis < len(in_shape1)
        assert in_shape1[:axis] == in_shape1[:axis]
        assert in_shape1[axis + 1 :] == in_shape1[axis + 1 :]
        super().__init__(in_shape1, in_shape2)
        self._axis = axis

    @property
    def axis(self) -> int:
        """Retrieves the axis.

        Returns:
            The axis.
        """
        return self._axis

    @property
    def shape(self) -> tuple[int, ...]:
        cross_dim = self.in_shape1[self.axis] * self.in_shape2[self.axis]
        return *self.in_shape1[: self.axis], cross_dim, *self.in_shape1[self.axis + 1 :]

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["axis"] = self.axis
        return config


class OuterProductParameter(OuterParameterOp):
    """A symbolic parameter operator represeting the outer product of two parameter tensors
    along one axis.
    """


class OuterSumParameter(OuterParameterOp):
    """A symbolic parameter operator represeting the outer sum of two parameter tensors
    along one axis.
    """


class ExpParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the exponential of a parameter tensor."""


class LogParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the logarithm of a parameter tensor."""


class SquareParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the entry-wise square of a parameter
    tensor.
    """


class SoftplusParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the entry-wise application of
    the softplus function to a parameter tensor.
    """


class SigmoidParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the entry-wise application of
    the sigmoid function to a parameter tensor.
    """


class ScaledSigmoidParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the entry-wise application of
    the rescaled sigmoid function to a parameter tensor.
    """

    def __init__(self, in_shape: tuple[int, ...], vmin: float, vmax: float):
        """Initializes a symbolic scaled sigmoid parameter operator.

        Args:
            in_shape: The shape of the input.
            vmin: The minimum output value.
            vmax: The maximum output value
        """
        assert vmin < vmax
        super().__init__(in_shape)
        self._vmin = vmin
        self._vmax = vmax

    @property
    def vmin(self) -> float:
        """Retrieves the minimum output value.

        Returns:
            The minimum value.
        """
        return self._vmin

    @property
    def vmax(self) -> float:
        """Retrieves the minimum output value.

        Returns:
            The maximum value.
        """
        return self._vmax

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["vmin"] = self.vmin
        config["vmax"] = self.vmax
        return config


class ClampParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the element-wise clamping operator."""

    def __init__(
        self,
        in_shape: tuple[int, ...],
        *,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """Initializes a symbolic clamp parameter operator. At least one between
            the minimum and maximum value must be specified.

        Args:
            in_shape: The shape of the input.
            vmin: The minimum value to clamp to. If it is None, then clamping for
                the minimum value is not performed.
            vmax: The maximum value to clamp to. If it is None, then clamping for
                the maximum value is not performed.
        """
        assert vmin is not None or vmax is not None
        super().__init__(in_shape)
        self._vmin = vmin
        self._vmax = vmax

    @property
    def vmin(self) -> float:
        """Retrieves the minimum clamping value.

        Returns:
            The minimum clamping value.
        """
        return self._vmin

    @property
    def vmax(self) -> float:
        """Retrieves the maximum clamping value.

        Returns:
            The maximum clamping value.
        """
        return self._vmax

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["vmin"] = self.vmin
        config["vmax"] = self.vmax
        return config


class ConjugateParameter(EntrywiseParameterOp):
    """A symbolic parameter operator representing the element-wise complex conjugation
    operation of its input.
    """


class ReduceSumParameter(ReduceParameterOp):
    """A symbolic parameter operator representing the sum reduction along one dimension."""


class ReduceProductParameter(ReduceParameterOp):
    """A symbolic parameter operator representing the product reduction along one dimension."""


class ReduceLSEParameter(ReduceParameterOp):
    """A symbolic parameter operator representing the LogSumExp reduction along one dimension."""


class SoftmaxParameter(EntrywiseReduceParameterOp):
    """A symbolic parameter operator representing the application of the Softmax function
    along one dimension.
    """


class LogSoftmaxParameter(EntrywiseReduceParameterOp):
    """A symoblic parameter operator representing the application of the LogSoftmax function
    along one dimension.
    """


class MixingWeightParameter(UnaryParameterOp):
    r"""The symbolic mixing weights parameter node, which takes as input a matrix $\mathbf{V}$
    of shape $(K, H)$, where $K$ is the number of units and $H$ is the arity of a
    [SumLayer][cirkit.symbolic.layers.SumLayer], and returns a matrix $\mathbf{W}$ of shape
    $(K, K * H)$, where $\mathbf{W}$ is the column-wise concatenation of matrices
    $\{ \mathrm{diag}(\mathbf{v}_{:i}) \}_{i=1}^H$, $\mathbf{v}_{:i}$ denotes the $i$-th column
    of $\mathbf{V} and $\mathrm{diag}$ transforms a vector to a diagonal matrix.

    This parameter node is used in
    [mixing_weight_factory][cirkit.symbolic.parameters.mixing_weight_factory] as to parameterize
    a sum layer encoding a weighted combination of its input vectors.
    """

    def __init__(self, in_shape: tuple[int, ...]):
        if len(in_shape) != 2:
            raise ValueError(f"Expected shape (num_units, arity), but found {in_shape}")
        super().__init__(in_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.in_shape[0], self.in_shape[0] * self.in_shape[1]


class GaussianProductMean(ParameterOp):
    """A symbolic parameter operator computing the mean of the product of two Gaussians,
    given the means and standard deviations of the input Gaussians. Note that we assume
    Gaussians being univariate.
    """

    def __init__(
        self,
        in_mean1_shape: tuple[int, ...],
        in_stddev1_shape: tuple[int, ...],
        in_mean2_shape: tuple[int, ...],
        in_stddev2_shape: tuple[int, ...],
    ):
        """Initializes a symbolic Gaussian product mean, given the shape of the input means
            and standard deviations.

        Args:
            in_mean1_shape: The shape of the mean of the first univariate Gaussians.
            in_stddev1_shape: The shape of the standard deviations of the first
                univariate Gaussians.
            in_mean2_shape: The shape of the mean of the second univariate Gaussians.
            in_stddev2_shape: The shape of the standard deviations of the second
                univariate Gaussians.
        """
        assert in_mean1_shape == in_stddev1_shape
        assert in_mean2_shape == in_stddev2_shape
        super().__init__(in_mean1_shape, in_stddev1_shape, in_mean2_shape, in_stddev2_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.in_shapes[0][0] * self.in_shapes[2][0],)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "in_mean1_shape": self.in_shapes[0],
            "in_stddev1_shape": self.in_shapes[1],
            "in_mean2_shape": self.in_shapes[2],
            "in_stddev2_shape": self.in_shapes[3],
        }


class GaussianProductStddev(BinaryParameterOp):
    """A symbolic parameter operator computing the standard deviation of the product of
    two Gaussians, given the standard deviations of the input Gaussians.
    """

    def __init__(self, in_stddev1_shape: tuple[int, ...], in_stddev2_shape: tuple[int, ...]):
        """Initializes a symbolic Gaussian product standard deviation,
            given the shape of the input standard deviations.

        Args:
            in_stddev1_shape: The shape of the standard deviations of the first
                univariate Gaussians.
            in_stddev2_shape: The shape of the standard deviations of the second
                univariate Gaussians.
        """
        super().__init__(in_stddev1_shape, in_stddev2_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.in_shapes[0][0] * self.in_shapes[1][0],)

    @property
    def config(self) -> dict[str, Any]:
        return {"in_stddev1_shape": self.in_shapes[0], "in_stddev2_shape": self.in_shapes[1]}


class GaussianProductLogPartition(ParameterOp):
    """A symbolic parameter operator computing the log partition function of the product of
    two Gaussians, given the means and standard deviations of the input Gaussians.
    """

    def __init__(
        self,
        in_mean1_shape: tuple[int, ...],
        in_stddev1_shape: tuple[int, ...],
        in_mean2_shape: tuple[int, ...],
        in_stddev2_shape: tuple[int, ...],
    ):
        """Initializes a symbolic Gaussian product log partition function,
            given the shape of the input means and standard deviations.

        Args:
            in_mean1_shape: The shape of the mean of the first univariate Gaussians.
            in_stddev1_shape: The shape of the standard deviations of the first
                univariate Gaussians.
            in_mean2_shape: The shape of the mean of the second univariate Gaussians.
            in_stddev2_shape: The shape of the standard deviations of the second
                univariate Gaussians.
        """
        assert in_mean1_shape == in_stddev1_shape
        assert in_mean2_shape == in_stddev2_shape
        super().__init__(in_mean1_shape, in_stddev1_shape, in_mean2_shape, in_stddev2_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.in_shapes[0][0] * self.in_shapes[2][0],)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "in_mean1_shape": self.in_shapes[0],
            "in_stddev1_shape": self.in_shapes[1],
            "in_mean2_shape": self.in_shapes[2],
            "in_stddev2_shape": self.in_shapes[3],
        }


class PolynomialProduct(BinaryParameterOp):
    """A symbolic parameter operator representing the coefficients of a polynomial resulting
    from the product of two polynomial.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            self.in_shape1[0] * self.in_shape2[0],  # dim Ko
            self.in_shape1[1] + self.in_shape2[1] - 1,  # dim deg+1
        )


class PolynomialDifferential(UnaryParameterOp):
    """A symbolic parameter operator representing the coefficients of a polynomial resulting
    from the differentiation of a polynomial.
    """

    def __init__(self, in_shape: tuple[int, ...], *, order: int = 1):
        """Initializes a symbolic polynomial differential coefficients.

        Args:
            in_shape: The shape of the coefficients of the input polynomial.
            order: The differentiation order.

        Raises:
            ValuerError: if the differentiation order is not a positive integer.
        """
        if order <= 0:
            raise ValueError("The order of differentiation must be positive.")
        super().__init__(in_shape)
        self.order = order

    @property
    def shape(self) -> tuple[int, ...]:
        # if dp1>order, i.e., deg>=order, then diff, else const 0.
        return (
            self.in_shape[0],
            self.in_shape[1] - self.order if self.in_shape[1] > self.order else 1,
        )

    @property
    def config(self) -> dict[str, Any]:
        config = super().config
        config["order"] = self.order
        return config


class Parameter(RootedDiAcyclicGraph[ParameterNode]):
    """The symbolic parameter computational graph. A symbolic parameter is a computational graph
    consisting of symbolic nodes, which represent how to compute a tensor parameter."""

    def __init__(
        self,
        nodes: Sequence[ParameterNode],
        in_nodes: Mapping[ParameterNode, Sequence[ParameterNode]],
        outputs: Sequence[ParameterNode],
    ):
        super().__init__(nodes, in_nodes, outputs)

        # Check the computational graph is consistent w.r.t.
        # the input and output shapes of each computational node
        for node in self.nodes:
            node_ins = self.node_inputs(node)
            if isinstance(node, ParameterInput):
                if len(node_ins):
                    raise ValueError(
                        f"{node}: found an input parameter node with {len(node_ins)} other inputs, "
                        "but expected none"
                    )
                continue
            assert isinstance(node, ParameterOp)
            if len(node.in_shapes) != len(node_ins):
                raise ValueError(
                    f"{node}: expected number of inputs {len(node.in_shapes)}, "
                    f"but found {len(node_ins)}"
                )
            node_ins_shapes = tuple(n.shape for n in node_ins)
            if node.in_shapes != node_ins_shapes:
                raise ValueError(
                    f"{node}: expected input shapes {node.in_shapes}, "
                    f"but found {node_ins_shapes}"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        """Retrieves the shape of the output tensor.

        Returns:
            The shape of the output of the computational graph.
        """
        return self.output.shape

    @classmethod
    def from_input(cls, p: ParameterInput) -> "Parameter":
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
            p = Parameter.from_input(p)
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
        ps = tuple(Parameter.from_input(p) if isinstance(p, ParameterInput) else p for p in ps)
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

    def ref(self) -> "Parameter":
        """Constructs a shallow copy of the parameter, where the tensor parameters
            are replace with reference parameters to them.

        Returns:
            A shallow copy of the parameter nodes, with the exception that tensor parameter nodes
                are replaced with symbolic references to them.
        """

        # Build a new symbolic parameter's computational graph, where the parameter tensors
        # become references to the tensors of the 'self' parameter's computational graph.
        # All the other nodes are new objects.
        def _replace_ref_or_copy(n: ParameterNode) -> ParameterNode:
            return ReferenceParameter(n) if isinstance(n, TensorParameter) else copy(n)

        return self._process_nodes(_replace_ref_or_copy)

    def _process_nodes(self, process_fn: Callable[[ParameterNode], ParameterNode]) -> "Parameter":
        # Process all the nodes by following the topological ordering and using a function
        nodes, in_nodes, outputs = topologically_process_nodes(
            self.topological_ordering(), self.outputs, process_fn, incomings_fn=self.node_inputs
        )
        return Parameter(nodes, in_nodes, outputs)

    def __repr__(self) -> str:
        return f"{Parameter.__name__}(shape={self.shape})"


class ParameterFactory(Protocol):
    """A factory that constucts symbolic parameter given a shape."""

    def __call__(self, shape: tuple[int, ...]) -> Parameter:
        """Constructs a symbolic parameter given the parameter shape.

        Args:
            shape: The shape.

        Returns:
            A parameter whose output shape is equal to the given shape.
        """


def mixing_weight_factory(shape: tuple[int, ...], *, param_factory: ParameterFactory) -> Parameter:
    r"""Construct the parameters of a [sum layer][cirkit.symbolic.layers.SumLayer] with
    arity > 1 such that it encodes a linear combination of the input vectors it receives.
    A sum layer with this semantics is also referred to as "mixing layer" in some papers
    (see references below).
    A mixing layer is parameterized by a parameter matrix $\mathbf{W}\in\bbR^{K\times H}$,
    where $K$ is the number of sum units, and H is the number of input vectors; and it computes
    $\sum_{i=1}^H \mathbf{w}_{:i} \mathbf{x}_i$. This function firstly constructs $\mathbf{W}$ given
    a parameter factory, and then reshapes it into a $K\times KH$ parameter matrix for a sum layer
    that mimics a mixing layer.

    References:
        * R. Peharz et al. (2020), Einsum Networks: Fast and Scalable Learning of
            Tractable Probabilistic Circuits
        * Loconte et al. (2024), What is the Relationship between Tensor Factorizations
            and Circuits (and How Can We Exploit it)?

    Args:
        shape: The shape of the parameter. It must be (num_units, arity * num_units), where
            num_units is the number of sum units or, equivalently the size of the input vectors,
            and arity is the number of them.
        param_factory: The parameter factory used to construct the mixing weights of shape
            (num_units, arity).

    Returns:
        Parameter: A symbolic parameter.

    Raises:
        ValueError: If the given shape is not of the form (num_units, arity * num_units).
    """
    if len(shape) != 2 or shape[1] % shape[0]:
        raise ValueError(f"Expected shape (num_units, arity * num_units), but found {shape}")
    num_units = shape[0]
    arity = shape[1] // num_units
    mixing_weights_shape = num_units, arity
    return Parameter.from_unary(
        MixingWeightParameter(mixing_weights_shape), param_factory(mixing_weights_shape)
    )
