from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, auto
from typing import Any, Callable, cast

from cirkit.symbolic.initializers import NormalInitializer
from cirkit.symbolic.parameters import (
    Parameter,
    ParameterFactory,
    ScaledSigmoidParameter,
    SigmoidParameter,
    SoftmaxParameter,
    TensorParameter,
)
from cirkit.utils.label import LayerLabel
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
    and sum layers. A layer that specializes this class must specify two property methods:
        1. config(self) -> Mapping[str, Any]: A dictionary mapping the non-parameter arguments to
            the ```__init__``` method to the corresponding values, e.g., the arity.
        2. params(self) -> Mapping[str, Parameter]: A dictionary mapping the parameter arguments
            the ```__init__``` method to the corresponding symbolic parameter, e.g., the mean and
            standard deviations symbolic parameters in a
            [GaussianLayer][cirkit.symbolic.layers.GaussianLayer].
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        label: LayerLabel | None = None,
    ):
        """Initializes a symbolic layer.

        Args:
            num_input_units: The number of units in each input layer.
            num_output_units: The number of output units, i.e., the number of computational units
                in this layer.
            arity: The arity of the layer, i.e., the number of input layers to this layer.
            label: The label of this layer.

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
        self._label = label

    @property
    def label(self) -> LayerLabel | None:
        """Retrieves the label of the layer.

        Returns:
            LayerLabel | None: The label of the layer or None, if it is not set.
        """
        return self._label

    @property
    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        """Retrieves the configuration of the layer, i.e., a dictionary mapping hyperparameters
        of the layer to their values. The hyperparameter names must match the argument names in
        the ```__init__``` method.

        Returns:
            Mapping[str, Any]: A dictionary from hyperparameter names to their value.
        """

    @property
    def params(self) -> Mapping[str, Parameter]:
        """Retrieve the symbolic parameters of the layer, i.e., a dictionary mapping the names of
        the symbolic parameters to the actual symbolic parameter instance. The parameter names must
        match the argument names in the```__init__``` method.

        Returns:
            Mapping[str, Parameter]: A dictionary from parameter names to the corresponding symbolic
                parameter instance.
        """
        return {}

    def copy(self, *, params: Mapping[str, Parameter] | None = None) -> "Layer":
        """Creates a _shallow_ copy of the layer, i.e., a copy where the symbolic
        parameters are copied by reference. If some parameters are specified, then
        these are replaced when performing the copy by reference.

        Args:
            params: The parameters that must be replaced when performing the copy.
                It can be None. If it is None, then this method defaults to a _shallow_
                copy of the layer.

        Returns:
            A shallow copy of the layer.
        """
        if params is None:
            return type(self)(**self.config, **self.params)
        updated_params = dict(self.params)
        updated_params.update(params)
        return type(self)(**self.config, **updated_params)

    def copyref(self) -> "Layer":
        """Creates a _reference_ copy of the layer, i.e., a shallow copy where the symbolic
        parameters are re-instantiated and where the tensor parameters are replaced to
        reference parameters, thus effectively creating a symbolic parameter sharing between
        the new layer and the layer being copied by _reference_.

        Returns:
            A reference copy of the layer, with reference to the parameters.
        """
        ref_params = {pname: pgraph.ref() for pname, pgraph in self.params.items()}
        return type(self)(**self.config, **ref_params)

    def __repr__(self) -> str:
        config_repr = ", ".join(f"{k}={v}" for k, v in self.config.items())
        params_repr = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (
            f"{self.__class__.__name__}("
            f"num_input_units={self.num_input_units}, "
            f"num_output_units={self.num_output_units}, "
            f"arity={self.arity}, "
            f"label={self._label},"
            f"config=({config_repr}), "
            f"params=({params_repr})"
            ")"
        )


class InputLayer(Layer, ABC):
    """The symbolic input layer class."""

    def __init__(self, scope: Scope, num_output_units: int, label: LayerLabel | None = None):
        """Initializes a symbolic input layer.

        Args:
            scope: The variables scope of the layer.
            num_output_units: The number of input units in the layer.
            label: The label of this layer.

        Raises:
            ValueError: If the number of outputs is not positive.
        """
        if num_output_units <= 0:
            raise ValueError("The number of output units should be positive")
        super().__init__(len(scope), num_output_units, label=label)
        self.scope = scope

    @property
    def num_variables(self) -> int:
        """The number of variables modelled by the input layer.

        Returns:
            int: The number of variables in the scope.
        """
        return self.num_input_units

    def __repr__(self) -> str:
        config_repr = ", ".join(f"{k}={v}" for k, v in self.config.items())
        params_repr = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return (
            f"{self.__class__.__name__}("
            f"scope={self.scope}, "
            f"num_output_units={self.num_output_units}, "
            f"config=({config_repr}), "
            f"params=({params_repr})"
            ")"
        )


class ConstantLayer(InputLayer, ABC):
    """The symbolic layer computing a constant vector, i.e., it does not depend on any variable."""

    def __init__(self, num_output_units: int, label: LayerLabel | None = None):
        """Initializes a symbolic constant layer.

        Args:
            num_output_units: The number of input units in the layer.
            label: The label of this input layer.
        """
        super().__init__(Scope([]), num_output_units, label=label)


class EvidenceLayer(ConstantLayer):
    """The symbolic layer computing the output of an input layer given by a complete observation.
    The only parameter of an evidence layer is a complete observation of the variables."""

    def __init__(self, layer: InputLayer, *, observation: Parameter):
        r"""Initializes a symbolic evidence layer.

        Args:
            layer: The symbolic input layer to condition, i.e., to evaluate on the observation.
            observation: The observation stored as a parameter that outputs a constant (i.e.,
                non-learnable) tensor of shape $(D,)$, where $D$ is the number of variable the
                symbolic input layer is defined on.

        Raises:
            ValueError: If the observation parameter shape has not two dimensions.
        """
        if len(observation.shape) != 1:
            raise ValueError(
                f"Expected observation of shape (num_variables,), " f"but found {observation.shape}"
            )
        if observation.shape[0] != layer.num_variables:
            raise ValueError(
                f"Expected an observation with number of variables {layer.num_variables}, "
                f"but found {observation.shape[0]}"
            )
        super().__init__(layer.num_output_units)
        self.layer = layer
        self.observation = observation

    @property
    def config(self) -> Mapping[str, Any]:
        return {"layer": self.layer}

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"observation": self.observation}


class EmbeddingLayer(InputLayer):
    r"""A symbolic Embedding layer, which is parameterized by as many embedding matrices as
    the number of variables. Each embedding matrix has size $M\times N$, where $M$ is the number
    of output units of the layer, and $N$ is the number of states each variable can assume.
    """

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        label: LayerLabel | None = None,
        num_states: int = 2,
        weight: Parameter | None = None,
        weight_factory: ParameterFactory | None = None,
    ):
        r"""Initializes an Embedding layer.

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of Categorical units in the layer.
            num_states: The number of categories for each variable.
            weight: The weight parameter of shape $(K, N)$, where $K$ is the number of output
                units, and $N$ is the number of states. If it is None, then either the weight
                factory is used (if it is not None) or a weight parameter is initialized.
            weight_factory: A factory used to construct the weight parameter,
                if it is not given
        """
        if len(scope) != 1:
            raise ValueError("The Embedding layer encodes univariate functions")
        if num_states <= 1:
            raise ValueError("The number of states must be at least 2")
        super().__init__(scope, num_output_units)
        self.num_states = num_states
        if weight is None:
            if weight_factory is None:
                weight = Parameter.from_input(
                    TensorParameter(*self._weight_shape, initializer=NormalInitializer())
                )
            else:
                weight = weight_factory(self._weight_shape)
        if weight.shape != self._weight_shape:
            raise ValueError(f"Expected parameter shape {self._weight_shape}, found {weight.shape}")
        self.weight = weight

    @property
    def _weight_shape(self) -> tuple[int, ...]:
        return (
            self.num_output_units,
            self.num_states,
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "num_states": self.num_states,
            "label": self.label,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"weight": self.weight}


class CategoricalLayer(InputLayer):
    """A symbolic Categorical layer, which is parameterized either by
    probabilities (yielding a normalized Categorical distribution) or by
    logits (yielding an unnormalized Categorical distribution)."""

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        label: LayerLabel | None = None,
        num_categories: int,
        logits: Parameter | None = None,
        probs: Parameter | None = None,
        logits_factory: ParameterFactory | None = None,
        probs_factory: ParameterFactory | None = None,
    ):
        r"""Initializes a Categorical layer.

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of Categorical units in the layer.
            num_categories: The number of categories for each variable.
            logits: The logits parameter of shape $(K, N)$, where $K$ is the number of output
                units, and $N$ is the number of categories. If it is None, then either the
                probabilities parameter is used (if it is not None) or a probabilities parameter
                parameterized by a [SoftmaxParameter][cirkit.symbolic.parameters.SoftmaxParameter].
            probs: The probabilities parameter of shape $(K, N)$ (see logits parameter
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
        super().__init__(scope, num_output_units, label=label)
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
    def _probs_logits_shape(self) -> tuple[int, ...]:
        return self.num_output_units, self.num_categories

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "num_categories": self.num_categories,
            "label": self.label,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        if self.logits is None:
            return {"probs": self.probs}
        return {"logits": self.logits}


class BinomialLayer(InputLayer):
    """A symbolic Binomial layer, which is parameterized either by
    probabilities (yielding a normalized Binomial distribution) or by
    logits (yielding an unnormalized Binomial distribution)."""

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        label: LayerLabel | None = None,
        total_count: int = 2,
        logits: Parameter | None = None,
        probs: Parameter | None = None,
        logits_factory: ParameterFactory | None = None,
        probs_factory: ParameterFactory | None = None,
    ):
        r"""Initializes a Binomial layer.

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of Categorical units in the layer.
            total_count: The number of total counts for each variable.
            logits: The logits parameter of shape $(K,)$, where $K$ is the number of output
                units. If it is None, then either the probabilities parameter is used
                (if it is not None) or a probabilities parameter parameterized by a
                [SigmoidParameter][cirkit.symbolic.parameters.SigmoidParameter].
            probs: The probabilities parameter of shape $(K,)$ (see logits parameter
                description). If it is None, then the logits parameter must be specified.
            logits_factory: A factory used to construct the logits parameter, if neither logits nor
                probabilities are given.
            probs_factory: A factory used to construct the probabilities parameter, if neither
                logits nor probabilities nor the logits parameter factory are given.
        """
        if logits is not None and probs is not None:
            raise ValueError("At most one between 'logits' and 'probs' can be specified")
        if logits_factory is not None and probs_factory is not None:
            raise ValueError(
                "At most one between 'logits_factory' and 'probs_factory' can be specified"
            )
        if total_count < 0:
            raise ValueError("The number of trials should be non-negative")
        super().__init__(scope, num_output_units)
        self.total_count = total_count
        if logits is None and probs is None:
            if logits_factory is not None:
                logits = logits_factory(self._probs_logits_shape)
            elif probs_factory is not None:
                probs = probs_factory(self._probs_logits_shape)
            else:  # Defaults to probs with sigmoid parameterization
                probs = Parameter.from_unary(
                    SigmoidParameter(self._probs_logits_shape),
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
    def _probs_logits_shape(self) -> tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def config(self) -> dict:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "total_count": self.total_count,
            "label": self.label,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
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
        *,
        label: LayerLabel | None = None,
        mean: Parameter | None = None,
        stddev: Parameter | None = None,
        log_partition: Parameter | None = None,
        mean_factory: ParameterFactory | None = None,
        stddev_factory: ParameterFactory | None = None,
    ):
        r"""Initializes a Gaussian layer.

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of Gaussian units in the layer.
            mean: The mean parameter of shape $(K)$, where $K$ is the number of output units.
                If it is None, then a default symbolic parameter will be instantiated with a
                [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer] as
                symbolic initializer.
            stddev: The standard deviation parameter of shape $(K)$, where $K$ is the number of
                output units. If it is None, then a default symbolic parameter will be instantiated
                with a [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer] as
                symbolic initializer, which is then re-parameterized to be positve using a
                [ScaledSigmoidParameter][cirkit.symbolic.parameters.ScaledSigmoidParameter].
            log_partition: The log-partition parameter of the Gaussian, of shape $(K,)$.
                If the Gaussian is a normalized Gaussian, then this should be None.
            mean_factory: A factory used to construct the mean parameter, if it is not specified.
            stddev_factory: A factory used to construct the standard deviation parameter, if it is
                not specified.
        """
        if len(scope) != 1:
            raise ValueError("The Gaussian layer encodes a univariate distribution")
        super().__init__(scope, num_output_units)
        if mean is None:
            if mean_factory is None:
                mean = Parameter.from_input(
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
    def _mean_stddev_shape(self) -> tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def _log_partition_shape(self) -> tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def config(self) -> Mapping[str, Any]:
        return {"scope": self.scope, "num_output_units": self.num_output_units}

    @property
    def params(self) -> Mapping[str, Parameter]:
        params = {"mean": self.mean, "stddev": self.stddev}
        if self.log_partition is not None:
            params.update(log_partition=self.log_partition)
        return params


class PolynomialLayer(InputLayer):
    """A symbolic layer that evaluates polynomials."""

    def __init__(
        self,
        scope: Scope,
        num_output_units: int,
        *,
        label: LayerLabel | None = None,
        degree: int,
        coeff: Parameter | None = None,
        coeff_factory: ParameterFactory | None = None,
    ):
        r"""Initializes a polynomial layer,

        Args:
            scope: The variables scope the layer depends on.
            num_output_units: The number of units each encoding a polynomial in the layer.
            degree: The degree of the polynomials.
            coeff: The coefficient parameter of shape $(K, \mathsf{degree} + 1)$, where $K$ is the
                number of output units. If it is None, then either the coefficient factory
                is used (if it not None), or a default
                symbolic parameter will be instantiated with a
                [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer] as
                symbolic initializer.
            coeff_factory: A factory used to construct the coeff parameter, if it is not specified.
        """
        if len(scope) != 1:
            raise ValueError("The Polynomial layer encodes univariate functions")
        super().__init__(scope, num_output_units)
        self.degree = degree
        if coeff is None:
            if coeff_factory is None:
                coeff = Parameter.from_input(
                    TensorParameter(*self._coeff_shape, initializer=NormalInitializer())
                )
            else:
                coeff = coeff_factory(self._coeff_shape)
        if coeff.shape != self._coeff_shape:
            raise ValueError(f"Expected parameter shape {self._coeff_shape}, found {coeff.shape}")
        self.coeff = coeff

    @property
    def _coeff_shape(self) -> tuple[int, ...]:
        return self.num_output_units, self.degree + 1

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "scope": self.scope,
            "num_output_units": self.num_output_units,
            "degree": self.degree,
            "label": self.label,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"coeff": self.coeff}


class ConstantValueLayer(ConstantLayer):
    """A symbolic layer computing a constant function encoded by a parameter."""

    def __init__(
        self,
        num_output_units: int,
        *,
        log_space: bool = False,
        value: Parameter,
        label: LayerLabel | None = None,
    ):
        """Initializes a constant value layer.

        Args:
            num_output_units: The number of output log partition functions.
            log_space: Whether the given value is in the log-space, i.e., this constant
                layer should encode ```exp(value)``` rather than ```value```.
            label: The label of this input layer.
            value: The symbolic parameter representing the encoded value.
                This symbolic paramater should have shape (K,), where K is the number of
                output units.
        """
        super().__init__(num_output_units)
        if value.shape != self._value_shape:
            raise ValueError(f"Expected parameter shape {self._value_shape}, found {value.shape}")
        self.value = value
        self.log_space = log_space

    @property
    def _value_shape(self) -> tuple[int, ...]:
        return (self.num_output_units,)

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_output_units": self.num_output_units,
            "log_space": self.log_space,
            "label": self.label,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"value": self.value}


class ProductLayer(Layer, ABC):
    """The abstract base class for symbolic product layers."""

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 2,
        *,
        label: LayerLabel | None = None,
    ):
        """Initializes a product layer.

        Args:
            num_input_units: The number of units in each input layer.
            num_output_units: The number of product units in the product layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.
            label: The label of this input layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        super().__init__(num_input_units, num_output_units, arity, label=label)


class HadamardLayer(ProductLayer):
    """The symbolic element-wise product (or Hadamard) layer. This layer computes the element-wise
    product of the vectors given in output by some input layers. Therefore, the number of product
    units in the layer is equal to the number of units in each input layer."""

    def __init__(self, num_input_units: int, arity: int = 2, *, label: LayerLabel | None = None):
        """Initializes a Hadamard product layer.

        Args:
            num_input_units: The number of units in each input layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.
            label: The label of this input layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        super().__init__(num_input_units, num_input_units, arity=arity, label=label)

    @property
    def config(self) -> Mapping[str, Any]:
        return {"num_input_units": self.num_input_units, "arity": self.arity, "label": self.label}


class KroneckerLayer(ProductLayer):
    """The symbolic outer product (or Kronecker) layer. This layer computes the outer
    product of the vectors given in output by some input layers. Therefore, the number of product
    units in the layer is equal to the product of the number of units in each input layer.
    Note that the output of a Kronecker layer is a vector."""

    def __init__(self, num_input_units: int, arity: int = 2, *, label: LayerLabel | None = None):
        """Initializes a Kronecker product layer.

        Args:
            num_input_units: The number of units in each input layer.
            arity: The arity of the layer, i.e., the number of input layers to the product layer.
            label: The label of this input layer.

        Raises:
            ValueError: If the arity is less than two.
        """
        if arity < 2:
            raise ValueError("The arity should be at least 2")
        super().__init__(
            num_input_units, cast(int, num_input_units**arity), arity=arity, label=label
        )

    @property
    def config(self) -> Mapping[str, Any]:
        return {"num_input_units": self.num_input_units, "arity": self.arity, "label": self.label}


class SumLayer(Layer):
    r"""The symbolic sum layer. A sum layer computes a matrix-by-vector product
    $\mathbf{W} \mathbf{x}$, where $\mathbf{W}\in\mathbb{R}^{K_1\times HK_2}$, where $K_1$ is the number
    of output units, $K_2$ is the number of input units, and $H$ is the arity, i.e., the number of
    layers that are input to the sum layer.
    In the product $\mathbf{W} \mathbf{x}$ above, $\mathbf{x}$ is the vector obtained by
    concatenating the outputs of all layers that are input to the sum layer. Note that if the arity
    is exactly 1, then this layer computes a simple linear transformation of an input vector.

    Depending on the parameterization of the parameter matrix $\mathbf{W}$, a different semantics
    can be set for the sum layer. For instance, if the parameter weight factory is chosen to be the
    [mixing weight factory][cirkit.symbolic.parameters.mixing_weight_factory], then the sum layer
    computes a weighted linear combination of the input vectors. See the
    [mixing weight factory][cirkit.symbolic.parameters.mixing_weight_factory] for more details.
    """

    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        weight: Parameter | None = None,
        weight_factory: ParameterFactory | None = None,
        *,
        label: LayerLabel | None = None,
    ):
        r"""Initializes a dense layer.

        Args:
            num_input_units: The number of units of the input layers.
            num_output_units: The number of sum units in the sum layer.
            arity: The arity of the layer, i.e., the number of input layers to the sum layer.
            weight: The symbolic weight matrix parameter, having shape
                $(K_o, K_i \times H)$, where $K_o$ is the number of output units, $K_i$ is the
                number of input units, and $H$ is the arity. It can be None.
            weight_factory: A factory that constructs the symbolic weight matrix parameter,
                if the given weight is None. If this factory is also None, then a weight
                parameter with [NormalInitializer][cirkit.symbolic.initializers.NormalInitializer]
                as initializer will be instantiated.
            label: The label of this input layer.
        """
        super().__init__(num_input_units, num_output_units, arity=arity, label=label)
        if weight is None:
            if weight_factory is None:
                weight = Parameter.from_input(
                    TensorParameter(*self._weight_shape, initializer=NormalInitializer())
                )
            else:
                weight = weight_factory(self._weight_shape)
        if weight.shape != self._weight_shape:
            raise ValueError(f"Expected parameter shape {self._weight_shape}, found {weight.shape}")
        self.weight = weight

    @property
    def _weight_shape(self) -> tuple[int, ...]:
        return self.num_output_units, self.arity * self.num_input_units

    @property
    def config(self) -> Mapping[str, Any]:
        return {
            "num_input_units": self.num_input_units,
            "num_output_units": self.num_output_units,
            "arity": self.arity,
            "label": self.label,
        }

    @property
    def params(self) -> Mapping[str, Parameter]:
        return {"weight": self.weight}
