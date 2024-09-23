from abc import ABC
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple, cast

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
    """The multiplication operator defined over layers."""
    CONJUGATION = auto()
    """The conjugation opereator defined over sum and input layers."""


class Layer(ABC):
    """The symbolic layer class. A symbolic layer consists of useful metadata of input, product
    and sum layers. A layer that specializes this class must specify two property methods:
        1. config(self) -> Dict[str, Any]: A dictionary mapping the non-parameter arguments to
            the ```__init__``` method to the corresponding values, e.g., the arity.
        2. params(self) -> Dict[str, Parameter]: A dictionary mapping the parameter arguments
            the ```__init__``` method to the corresponding symbolic parameter, e.g., the mean and
            standard deviations symbolic parameters in a
            [GaussianLayer][cirkit.symbolic.layers.GaussianLayer].
    """

    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
    ):
        """Initializes a symbolic layer.

        Args:
            scope: The variables scope of the layer.
            num_input_units: The number of units in each input layer.
            num_output_units: The number of output units, i.e., the number of computational units
                in this layer.
            arity: The arity of the layer, i.e., the number of input layers to this layer.
        """
        self.scope = scope
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity

    @property
    def config(self) -> Dict[str, Any]:
        """Retrieves the configuration of the layer, i.e., a dictionary mapping hyperparameters
        of the layer to their values. The hyperparameter names must match the argument names in
        the ```__init__``` method.

        Returns:
            Dict[str, Any]: A dictionary from hyperparameter names to their value.
        """
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
        }

    @property
    def params(self) -> Dict[str, Parameter]:
        """Retrieve the symbolic parameters of the layer, i.e., a dictionary mapping the names of
        the symbolic parameters to the actual symbolic parameter instance. The parameter names must
        match the argument names in the```__init__``` method.

        Returns:
            Dict[str, Parameter]: A dictionary from parameter names to the corresponding symbolic
                parameter instance.
        """
        return {}


class InputLayer(Layer):
    """The symbolic input layer class."""

    def __init__(self, scope: Scope, num_output_units: int, num_channels: int = 1):
        """Initializes a symbolic input layer.

        Args:
            scope: The variables scope of the layer.
            num_output_units: The number of input units in the layer.
            num_channels: The number of channels for each variable in the scope. Defaults to 1.
        """
        super().__init__(scope, len(scope), num_output_units, num_channels)

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

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "num_channels": self.num_channels,
        }


class CategoricalLayer(InputLayer):
    """A symbolic Categorical layer, which is parameterized either by
    probabilities (yielding a normalized Categorical distribution) or by
    logits (yielding an unnormalized Categorical distribution)."""

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        num_channels: int,
        num_categories: int,
        *,
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

    @property
    def config(self) -> dict:
        config = super().config
        config.update(num_categories=self.num_categories)
        return config

    @property
    def params(self) -> Dict[str, Parameter]:
        if self.logits is None:
            return {"probs": self.probs}
        return {"logits": self.logits}


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

    @property
    def params(self) -> Dict[str, Parameter]:
        params = {"mean": self.mean, "stddev": self.stddev}
        if self.log_partition is not None:
            params.update(log_partition=self.log_partition)
        return params


class LogPartitionLayer(InputLayer):
    """A symbolic layer computing a log-partition function."""

    def __init__(self, scope: Scope, num_output_units: int, num_channels: int, *, value: Parameter):
        """Initializes a Log Partition layer.

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of output log partition functions.
            num_channels: The number of channels per variable.
            value: The symbolic parameter representing the log partition value.
        """
        super().__init__(scope, num_output_units, num_channels)
        if value.shape != self._value_shape:
            raise ValueError(f"Expected parameter shape {self._value_shape}, found {value.shape}")
        self.value = value

    @property
    def _value_shape(self) -> Tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def params(self) -> Dict[str, Parameter]:
        params = super().params
        params.update(value=self.value)
        return params


class ProductLayer(Layer, ABC):
    """The abstract base class for symbolic product layers."""

    def __init__(self, scope: Scope, num_input_units: int, num_output_units: int, arity: int = 2):
        """Initializes a product layer.

        Args:
            scope: The variables scope of the layer.
            num_input_units: The number of units in each input layer.
            num_output_units: The number of product units in the product layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        super().__init__(scope, num_input_units, num_output_units, arity)

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "arity": self.arity,
        }


class HadamardLayer(ProductLayer):
    """The symbolic element-wise product (or Hadamard) layer. This layer computes the element-wise
    product of the vectors given in output by some input layers. Therefore, the number of product
    units in the layer is equal to the number of units in each input layer."""

    def __init__(self, scope: Scope, num_input_units: int, arity: int = 2):
        """Initializes a Hadamard product layer.

        Args:
            scope: The variables scope of the layer.
            num_input_units: The number of units in each input layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        super().__init__(scope, num_input_units, num_input_units, arity=arity)


class KroneckerLayer(ProductLayer):
    """The symbolic outer product (or Kronecker) layer. This layer computes the outer
    product of the vectors given in output by some input layers. Therefore, the number of product
    units in the layer is equal to the product of the number of units in each input layer.
    Note that the output of a Kronecker layer is a vector."""

    def __init__(self, scope: Scope, num_input_units: int, arity: int = 2):
        """Initializes a Kronecker product layer.

        Args:
            scope: The variables scope of the layer.
            num_input_units: The number of units in each input layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        super().__init__(
            scope,
            num_input_units,
            cast(int, num_input_units**arity),
            arity=arity,
        )


class SumLayer(Layer, ABC):
    """The abstract base class for symbolic sum layers."""


class DenseLayer(SumLayer):
    """The symbolic dense layer. A dense layer computes a matrix-by-vector product W * u,
    where W is a SxK matrix and u is the output vector of length K of another layer."""

    def __init__(
        self,
        scope: Scope,
        num_input_units: int,
        num_output_units: int,
        weight: Optional[Parameter] = None,
        weight_factory: Optional[ParameterFactory] = None,
    ):
        """Initializes a dense layer.

        Args:
            scope: The variables scope of the layer.
            num_input_units: The number of units of the input layer.
            num_output_units: The number of sum units in the dense layer.
            weight: The symbolic weight matrix parameter, having shape (S, K), where S is the
                number of output units and K is the number of input units. It can be None.
            weight_factory: A factory that constructs the symbolic weight matrix parameter,
                if the given weight is None. If this factory is also None, then a weight
                parameter with [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer]
                as initializer will be instantiated.
        """
        super().__init__(scope, num_input_units, num_output_units, arity=1)
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

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
        }

    @property
    def params(self) -> Dict[str, Parameter]:
        return {"weight": self.weight}


class MixingLayer(SumLayer):
    """The symbolic mixing sum layer. A mixing layer takes N layers as inputs, where each one
    outputs a K-dimensional vector, and computes a weighted sum over them. Therefore, the
    output of a mixing layer is also a K-dimensional vector."""

    def __init__(
        self,
        scope: Scope,
        num_units: int,
        arity: int,
        weight: Optional[Parameter] = None,
        weight_factory: Optional[ParameterFactory] = None,
    ):
        """Initializes a mixing layer.

        Args:
            scope: The variables scope of the layer.
            num_units: The number of units in each of the input layers.
            arity: The arity of the layer, i.e., the number of input layers to the mixing layer.
            weight: The symbolic weight matrix parameter, having shape (K, R), where K is the
                number of units and R is the arity. It can be None.
            weight_factory: A factory that constructs the symbolic weight matrix parameter,
                if the given weight is None. If this factory is also None, then a weight
                parameter with [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer]
                as initializer will be instantiated.
        """
        super().__init__(scope, num_units, num_units, arity)
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
        return self.num_input_units, self.arity

    @property
    def config(self) -> Dict[str, Any]:
        return {"scope": self.scope, "num_units": self.num_input_units, "arity": self.arity}

    @property
    def params(self) -> Dict[str, Parameter]:
        return {"weight": self.weight}
