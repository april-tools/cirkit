from abc import ABC, abstractmethod
from enum import IntEnum, auto
from typing import Optional, Tuple, cast

from cirkit.symbolic.initializers import NormalInitializer
from cirkit.symbolic.parameters import (
    Parameter,
    ParameterFactory,
    ScaledSigmoidParameter,
    SoftmaxParameter,
    TensorParameter,
)
from cirkit.utils.scope import Scope


class LayerOperator(IntEnum):
    """The avaliable symbolic operators defined over layers."""

    INTEGRATION = auto()
    """The integration operator defined over input layers."""
    DIFFERENTIATION = auto()
    """The differentiation operator defined over layers."""
    MULTIPLICATION = auto()
    """The multiplication (Kronecker product) operator defined over layers."""
    CONJUGATION = auto()
    """The conjugation opereator defined over sum and input layers."""


class Layer(ABC):
    """The symbolic layer class. A symbolic layer consists of useful metadata of input, product
    and sum layers."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
    ):
        """Initializes a symbolic layer.

        Args:
            num_input_units: The number of units in each input layer.
            num_output_units: The number of output units, i.e., the number of computational units
                in this layer.
            arity: The arity of the layer, i.e., the number of input layers to this layer.

        Raises:
            ValueError: If the number of input units, output units or the arity are not positvie.
        """
        if num_input_units < 0:
            raise ValueError("The number of input units should be non-negative")
        if num_output_units <= 0:
            raise ValueError("The number of output units should be positive")
        if arity <= 0:
            raise ValueError("The arity should be positive")
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity

    def __copy__(self) -> "Layer":
        return self.copy()

    @abstractmethod
    def copy(self) -> "Layer":
        """Creates a _shallow_ copy of the layer, i.e., a copy where the symbolic parameters
        are copied by reference, thus effectively creating a symbolic parameter reference between
        the new layer and the layer being copied.

        Returns:
            A shallow copy of the layer.
        """
        ...


class InputLayer(Layer, ABC):
    """The symbolic input layer class."""

    def __init__(self, scope: Scope, num_output_units: int, num_channels: int = 1):
        """Initializes a symbolic input layer.

        Args:
            scope: The variables scope of the layer.
            num_output_units: The number of input units in the layer.
            num_channels: The number of channels for each variable in the scope. Defaults to 1.

        Raises:
            ValueError: If the number of outputs or the number of channels are not positive.
        """
        if num_output_units <= 0:
            raise ValueError("The number of output units should be positive")
        if num_channels <= 0:
            raise ValueError("The number of channels should be positive")
        super().__init__(len(scope), num_output_units, num_channels)
        self.scope = scope

    @property
    def num_variables(self) -> int:
        """The number of variables modelled by the input layer.

        Returns:
            int: The number of variables in the scope.
        """
        return self.num_input_units

    @property
    def num_channels(self) -> int:
        """The number of channels per variable modelled by the input layer.

        Returns:
            int: The number of channels per variable.
        """
        return self.arity


class ConstantLayer(InputLayer, ABC):
    """The symbolic layer computing a constant vector, i.e., it does not depend on any variable."""

    def __init__(self, num_output_units: int):
        """Initializes a symbolic constant layer.

        Args:
            num_output_units: The number of input units in the layer.
        """
        super().__init__(Scope([]), num_output_units)


class EvidenceLayer(ConstantLayer):
    """The symbolic layer computing the output of an input layer given by a complete observation.
    The only parameter of an evidence layer is a complete observation of the variables."""

    def __init__(self, layer: InputLayer, *, observation: Parameter):
        """Initializes a symbolic evidence layer.

        Args:
            layer: The symbolic input layer to condition, i.e., to evaluate on the observation.
            observation: The observation stored as a parameter that outputs a constant (i.e.,
                non-learnable) tensor of shape (C, D), where D is the number of variable the
                symbolic input layer is defined on, and C is the number of channels per variable.

        Raises:
            ValueError: If the observation parameter shape has not two dimensions, or if the
                number of its channels (resp. variables) does not match the number of channels
                (resp. variables) of the symbolic input layer.
        """
        if len(observation.shape) != 2:
            raise ValueError(
                f"Expected observation of shape (num_channels, num_variables), "
                f"but found {observation.shape}"
            )
        num_channels, num_variables = observation.shape
        if num_channels != layer.num_channels:
            raise ValueError(
                f"Expected an observation with number of channels {layer.num_channels}, "
                f"but found {num_channels}"
            )
        if num_variables != layer.num_variables:
            raise ValueError(
                f"Expected an observation with number of variables {layer.num_variables}, "
                f"but found {num_variables}"
            )
        super().__init__(layer.num_output_units)
        self.layer = layer
        self.observation = observation

    def copy(self) -> "EvidenceLayer":
        return EvidenceLayer(self.layer, observation=self.observation)


class CategoricalLayer(InputLayer):
    """A symbolic Categorical layer, which is parameterized either by
    probabilities (yielding a normalized Categorical distribution) or by
    logits (yielding an unnormalized Categorical distribution)."""

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int = 1,
        *,
        num_categories: int,
        logits: Optional[Parameter] = None,
        probs: Optional[Parameter] = None,
        logits_factory: Optional[ParameterFactory] = None,
        probs_factory: Optional[ParameterFactory] = None,
    ):
        """Initializes a Categorical layer.

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of Categorical units in the layer.
            num_channels: The number of channels per variable.
            num_categories: The number of categories for each variable and channel.
            logits: The logits parameter of shape (K, C, N), where K is the number of output units,
                C is the number of channels, and N is the number of categories. If it is None,
                then either the probabilities parameter is used (if it is not None) or a
                probabilities parameter parameterized by a
                [SoftmaxParameter][cirkit.symbolic.parameters.SoftmaxParameter].
            probs: The probabilities parameter of shape (K, C, N) (see logits parameter
                description). If it is None, then the logits parameter must be specified.
            logits_factory: A factory used to construct the logits parameter, if neither logits nor
                probabilities are given.
            probs_factory: A factory used to construct the probabilities parameter, if neither
                logits nor probabilities nor the logits parameter factory are given.
        """
        if len(scope) != 1:
            raise ValueError("The Categorical layer encodes a univariate distribution")
        if logits is not None and probs is not None:
            raise ValueError("At most one between 'logits' and 'probs' can be specified")
        if logits_factory is not None and probs_factory is not None:
            raise ValueError(
                "At most one between 'logits_factory' and 'probs_factory' can be specified"
            )
        if num_categories < 2:
            raise ValueError("At least two categories must be specified")
        super().__init__(scope, num_output_units, num_channels)
        self.num_categories = num_categories
        if logits is None and probs is None:
            if logits_factory is not None:
                logits = logits_factory(self._probs_logits_shape)
            elif probs_factory is not None:
                probs = probs_factory(self._probs_logits_shape)
            else:  # Defaults to probs with softmax parameterization
                probs = Parameter.from_unary(
                    SoftmaxParameter(self._probs_logits_shape),
                    TensorParameter(*self._probs_logits_shape, initializer=NormalInitializer()),
                )
        if logits is not None and logits.shape != self._probs_logits_shape:
            raise ValueError(
                f"Expected parameter shape {self._probs_logits_shape}, found {logits.shape}"
            )
        if probs is not None and probs.shape != self._probs_logits_shape:
            raise ValueError(
                f"Expected parameter shape {self._probs_logits_shape}, found {probs.shape}"
            )
        self.probs = probs
        self.logits = logits

    @property
    def _probs_logits_shape(self) -> Tuple[int, ...]:
        return self.num_output_units, self.num_channels, self.num_categories

    def copy(self) -> "CategoricalLayer":
        return CategoricalLayer(
            self.scope,
            self.num_output_units,
            self.num_channels,
            num_categories=self.num_categories,
            logits=self.logits,
            probs=self.probs,
        )


class GaussianLayer(InputLayer):
    """A symbolic Gaussian layer, which is parameterized by mean and standard deviations.
    Optionally, it can represent an unnormalized Gaussian layer by specifying the log partition
    function."""

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        *,
        mean: Optional[Parameter] = None,
        stddev: Optional[Parameter] = None,
        log_partition: Optional[Parameter] = None,
        mean_factory: Optional[ParameterFactory] = None,
        stddev_factory: Optional[ParameterFactory] = None,
    ):
        """Initializes a Gaussian layer.

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of Gaussian units in the layer.
            num_channels: The number of channels per variable.
            mean: The mean parameter of shape (K, C), where K is the number of output units, and
                C is the number of channels. If it is None, then a default symbolic parameter will
                be instantiated with a
                [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer] as
                symbolic initializer.
            stddev: The standard deviation parameter of shape (K, C), where K is the number of
                output units, and C is the number of channels. If it is None, then a default
                symbolic parameter will be instantiated with a
                [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer] as
                symbolic initializer, which is then re-parameterized to be positve using a
                [ScaledSigmoidParameter][cirkit.symbolic.parameters.ScaledSigmoidParameter].
            mean: A factory used to construct the mean parameter, if it is not specified.
            stddev: A factory used to construct the standard deviation parameter, if it is not
                specified.
        """
        if len(scope) != 1:
            raise ValueError("The Gaussian layer encodes a univariate distribution")
        super().__init__(scope, num_output_units, num_channels)
        if mean is None:
            if mean_factory is None:
                mean = Parameter.from_leaf(
                    TensorParameter(*self._mean_stddev_shape, initializer=NormalInitializer())
                )
            else:
                mean = mean_factory(self._mean_stddev_shape)
        if stddev is None:
            if stddev_factory is None:
                stddev = Parameter.from_unary(
                    ScaledSigmoidParameter(self._mean_stddev_shape, vmin=1e-5, vmax=1.0),
                    TensorParameter(*self._mean_stddev_shape, initializer=NormalInitializer()),
                )
            else:
                stddev = stddev_factory(self._mean_stddev_shape)
        if mean.shape != self._mean_stddev_shape:
            raise ValueError(
                f"Expected parameter shape {self._mean_stddev_shape}, found {mean.shape}"
            )
        if stddev.shape != self._mean_stddev_shape:
            raise ValueError(
                f"Expected parameter shape {self._mean_stddev_shape}, found {stddev.shape}"
            )
        if log_partition is not None and log_partition.shape != self._log_partition_shape:
            raise ValueError(
                f"Expected parameter shape {self._log_partition_shape}, found {log_partition.shape}"
            )
        self.mean = mean
        self.stddev = stddev
        self.log_partition = log_partition

    @property
    def _mean_stddev_shape(self) -> Tuple[int, ...]:
        return self.num_output_units, self.num_channels

    @property
    def _log_partition_shape(self) -> Tuple[int, ...]:
        return self.num_output_units, self.num_channels

    def copy(self) -> "GaussianLayer":
        return GaussianLayer(
            self.scope,
            self.num_output_units,
            self.num_channels,
            mean=self.mean,
            stddev=self.stddev,
            log_partition=self.log_partition,
        )


class PolynomialLayer(InputLayer):
    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        *,
        degree: int,
        coeff: Optional[Parameter] = None,
        coeff_factory: Optional[ParameterFactory] = None,
    ):
        if len(scope) != 1:
            raise ValueError("The Polynomial layer encodes a univariate distribution")
        if num_channels != 1:
            raise ValueError("The Polynomial layer encodes a univariate distribution")
        super().__init__(scope, num_output_units, num_channels)
        self.degree = degree
        if coeff is None:
            if coeff_factory is None:
                coeff = Parameter.from_leaf(
                    TensorParameter(*self._coeff_shape, initializer=NormalInitializer())
                )
            else:
                coeff = coeff_factory(self._coeff_shape)
        if coeff.shape != self._coeff_shape:
            raise ValueError(f"Expected parameter shape {self._coeff_shape}, found {coeff.shape}")
        self.coeff = coeff

    @property
    def _coeff_shape(self) -> Tuple[int, ...]:
        return self.num_output_units, self.degree + 1

    def copy(self) -> "PolynomialLayer":
        return PolynomialLayer(
            self.scope,
            self.num_output_units,
            self.num_channels,
            degree=self.degree,
            coeff=self.coeff,
        )


class LogPartitionLayer(ConstantLayer):
    """A symbolic layer computing a log-partition function."""

    def __init__(self, num_output_units: int, *, value: Parameter):
        """Initializes a Log Partition layer.

        Args:
            num_output_units: The number of output log partition functions.
            value: The symbolic parameter representing the log partition value.
                This symbolic paramater should have shape (K,), where K is the number of
                output units.
        """
        super().__init__(num_output_units)
        if value.shape != self._value_shape:
            raise ValueError(f"Expected parameter shape {self._value_shape}, found {value.shape}")
        self.value = value

    @property
    def _value_shape(self) -> Tuple[int, ...]:
        return (self.num_output_units,)

    def copy(self) -> "LogPartitionLayer":
        return LogPartitionLayer(self.num_output_units, value=self.value)


class ProductLayer(Layer, ABC):
    """The abstract base class for symbolic product layers."""

    def __init__(self, num_input_units: int, num_output_units: int, arity: int = 2):
        """Initializes a product layer.

        Args:
            num_input_units: The number of units in each input layer.
            num_output_units: The number of product units in the product layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        super().__init__(num_input_units, num_output_units, arity)


class HadamardLayer(ProductLayer):
    """The symbolic element-wise product (or Hadamard) layer. This layer computes the element-wise
    product of the vectors given in output by some input layers. Therefore, the number of product
    units in the layer is equal to the number of units in each input layer."""

    def __init__(self, num_input_units: int, arity: int = 2):
        """Initializes a Hadamard product layer.

        Args:
            num_input_units: The number of units in each input layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        super().__init__(num_input_units, num_input_units, arity=arity)

    def copy(self) -> "HadamardLayer":
        return HadamardLayer(self.num_input_units, arity=self.arity)


class KroneckerLayer(ProductLayer):
    """The symbolic outer product (or Kronecker) layer. This layer computes the outer
    product of the vectors given in output by some input layers. Therefore, the number of product
    units in the layer is equal to the product of the number of units in each input layer.
    Note that the output of a Kronecker layer is a vector."""

    def __init__(self, num_input_units: int, arity: int = 2):
        """Initializes a Kronecker product layer.

        Args:
            num_input_units: The number of units in each input layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        super().__init__(
            num_input_units,
            cast(int, num_input_units**arity),
            arity=arity,
        )

    def copy(self) -> "KroneckerLayer":
        return KroneckerLayer(self.num_input_units, arity=self.arity)


class SumLayer(Layer, ABC):
    """The abstract base class for symbolic sum layers."""


class DenseLayer(SumLayer):
    """The symbolic dense layer. A dense layer computes a matrix-by-vector product W * u,
    where W is a SxK matrix and u is the output vector of length K of another layer."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        weight: Optional[Parameter] = None,
        weight_factory: Optional[ParameterFactory] = None,
    ):
        """Initializes a dense layer.

        Args:
            num_input_units: The number of units of the input layer.
            num_output_units: The number of sum units in the dense layer.
            weight: The symbolic weight matrix parameter, having shape (S, K), where S is the
                number of output units and K is the number of input units. It can be None.
            weight_factory: A factory that constructs the symbolic weight matrix parameter,
                if the given weight is None. If this factory is also None, then a weight
                parameter with [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer]
                as initializer will be instantiated.
        """
        super().__init__(num_input_units, num_output_units, arity=1)
        if weight is None:
            if weight_factory is None:
                weight = Parameter.from_leaf(
                    TensorParameter(*self._weight_shape, initializer=NormalInitializer())
                )
            else:
                weight = weight_factory(self._weight_shape)
        if weight.shape != self._weight_shape:
            raise ValueError(f"Expected parameter shape {self._weight_shape}, found {weight.shape}")
        self.weight = weight

    @property
    def _weight_shape(self) -> Tuple[int, ...]:
        return self.num_output_units, self.num_input_units

    def copy(self) -> "DenseLayer":
        return DenseLayer(self.num_input_units, self.num_output_units, weight=self.weight)


class MixingLayer(SumLayer):
    """The symbolic mixing sum layer. A mixing layer takes N layers as inputs, where each one
    outputs a K-dimensional vector, and computes a weighted sum over them. Therefore, the
    output of a mixing layer is also a K-dimensional vector."""

    def __init__(
        self,
        num_units: int,
        arity: int,
        weight: Optional[Parameter] = None,
        weight_factory: Optional[ParameterFactory] = None,
    ):
        """Initializes a mixing layer.

        Args:
            num_units: The number of units in each of the input layers.
            arity: The arity of the layer, i.e., the number of input layers to the mixing layer.
            weight: The symbolic weight matrix parameter, having shape (K, R), where K is the
                number of units and R is the arity. It can be None.
            weight_factory: A factory that constructs the symbolic weight matrix parameter,
                if the given weight is None. If this factory is also None, then a weight
                parameter with [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer]
                as initializer will be instantiated.
        """
        super().__init__(num_units, num_units, arity)
        if weight is None:
            if weight_factory is None:
                weight = Parameter.from_leaf(
                    TensorParameter(*self._weight_shape, initializer=NormalInitializer())
                )
            else:
                weight = weight_factory(self._weight_shape)
        if weight.shape != self._weight_shape:
            raise ValueError(f"Expected parameter shape {self._weight_shape}, found {weight.shape}")
        self.weight = weight

    @property
    def num_units(self) -> int:
        return self.num_input_units

    @property
    def _weight_shape(self) -> Tuple[int, ...]:
        return self.num_input_units, self.arity

    def copy(self) -> "MixingLayer":
        return MixingLayer(self.num_units, self.arity, weight=self.weight)
