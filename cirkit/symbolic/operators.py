from typing import Protocol

import numpy as np

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import (
    CategoricalLayer,
    ConstantValueLayer,
    EmbeddingLayer,
    GaussianLayer,
    HadamardLayer,
    KroneckerLayer,
    Layer,
    LayerOperator,
    PolynomialLayer,
    SumLayer,
)
from cirkit.symbolic.parameters import (
    ConjugateParameter,
    ConstantParameter,
    GaussianProductLogPartition,
    GaussianProductMean,
    GaussianProductStddev,
    KroneckerParameter,
    LogParameter,
    OuterProductParameter,
    OuterSumParameter,
    Parameter,
    PolynomialDifferential,
    PolynomialProduct,
    ReduceLSEParameter,
    ReduceSumParameter,
    SumParameter,
)
from cirkit.utils.scope import Scope


def integrate_embedding_layer(sl: EmbeddingLayer, *, scope: Scope) -> CircuitBlock:
    if not len(sl.scope & scope):
        raise ValueError(
            f"The scope of the Embedding layer '{sl.scope}'"
            f" is expected to be a subset of the integration scope '{scope}'"
        )
    reduce_sum = ReduceSumParameter(sl.weight.shape, axis=1)
    value = Parameter.from_unary(reduce_sum, sl.weight.ref())
    sl = ConstantValueLayer(sl.num_output_units, log_space=False, value=value)
    return CircuitBlock.from_layer(sl)


def integrate_categorical_layer(sl: CategoricalLayer, *, scope: Scope) -> CircuitBlock:
    if not len(sl.scope & scope):
        raise ValueError(
            f"The scope of the Categorical layer '{sl.scope}'"
            f" is expected to be a subset of the integration scope '{scope}'"
        )
    if sl.logits is None:
        log_partition = Parameter.from_input(ConstantParameter(sl.num_output_units, value=0.0))
    else:
        reduce_lse = ReduceLSEParameter(sl.logits.shape, axis=1)
        log_partition = Parameter.from_unary(reduce_lse, sl.logits.ref())
    sl = ConstantValueLayer(sl.num_output_units, log_space=True, value=log_partition)
    return CircuitBlock.from_layer(sl)


def integrate_gaussian_layer(sl: GaussianLayer, *, scope: Scope) -> CircuitBlock:
    if not len(sl.scope & scope):
        raise ValueError(
            f"The scope of the Gaussian layer '{sl.scope}'"
            f" is expected to be a subset of the integration scope '{scope}'"
        )
    if sl.log_partition is None:
        log_partition = Parameter.from_input(ConstantParameter(sl.num_output_units, value=0.0))
    else:
        log_partition = sl.log_partition.ref()
    sl = ConstantValueLayer(sl.num_output_units, log_space=True, value=log_partition)
    return CircuitBlock.from_layer(sl)


def multiply_embedding_layers(sl1: EmbeddingLayer, sl2: EmbeddingLayer) -> CircuitBlock:
    if sl1.scope != sl2.scope:
        raise ValueError(
            f"Expected Embedding layers to have the same scope,"
            f" but found '{sl1.scope}' and '{sl2.scope}'"
        )
    if sl1.num_states != sl2.num_states:
        raise ValueError(
            f"Expected Embedding layers to have the number of categories,"
            f"but found '{sl1.num_states}' and '{sl2.num_states}'"
        )

    weight = Parameter.from_binary(
        OuterProductParameter(sl1.weight.shape, sl2.weight.shape, axis=0),
        sl1.weight.ref(),
        sl2.weight.ref(),
    )
    sl = EmbeddingLayer(
        sl1.scope,
        sl1.num_output_units * sl2.num_output_units,
        num_states=sl1.num_states,
        weight=weight,
    )
    return CircuitBlock.from_layer(sl)


def multiply_categorical_layers(sl1: CategoricalLayer, sl2: CategoricalLayer) -> CircuitBlock:
    if sl1.scope != sl2.scope:
        raise ValueError(
            f"Expected Categorical layers to have the same scope,"
            f" but found '{sl1.scope}' and '{sl2.scope}'"
        )
    if sl1.num_categories != sl2.num_categories:
        raise ValueError(
            f"Expected Categorical layers to have the number of categories,"
            f"but found '{sl1.num_categories}' and '{sl2.num_categories}'"
        )

    if sl1.logits is None:
        sl1_logits = Parameter.from_unary(LogParameter(sl1.probs.shape), sl1.probs.ref())
    else:
        sl1_logits = sl1.logits.ref()
    if sl2.logits is None:
        sl2_logits = Parameter.from_unary(LogParameter(sl2.probs.shape), sl2.probs.ref())
    else:
        sl2_logits = sl2.logits.ref()
    sl_logits = Parameter.from_binary(
        OuterSumParameter(sl1_logits.shape, sl2_logits.shape, axis=0),
        sl1_logits,
        sl2_logits,
    )
    sl = CategoricalLayer(
        sl1.scope,
        sl1.num_output_units * sl2.num_output_units,
        num_categories=sl1.num_categories,
        logits=sl_logits,
    )
    return CircuitBlock.from_layer(sl)


def multiply_gaussian_layers(sl1: GaussianLayer, sl2: GaussianLayer) -> CircuitBlock:
    if sl1.scope != sl2.scope:
        raise ValueError(
            f"Expected Gaussian layers to have the same scope,"
            f" but found '{sl1.scope}' and '{sl2.scope}'"
        )

    mean = Parameter.from_nary(
        GaussianProductMean(sl1.mean.shape, sl1.stddev.shape, sl2.mean.shape, sl2.stddev.shape),
        sl1.mean.ref(),
        sl1.stddev.ref(),
        sl2.mean.ref(),
        sl2.stddev.ref(),
    )
    stddev = Parameter.from_binary(
        GaussianProductStddev(sl1.stddev.shape, sl2.stddev.shape),
        sl1.stddev.ref(),
        sl2.stddev.ref(),
    )
    log_partition = Parameter.from_nary(
        GaussianProductLogPartition(
            sl1.mean.shape, sl1.stddev.shape, sl2.mean.shape, sl2.stddev.shape
        ),
        sl1.mean.ref(),
        sl1.stddev.ref(),
        sl2.mean.ref(),
        sl2.stddev.ref(),
    )

    if sl1.log_partition is not None or sl2.log_partition is not None:
        if sl1.log_partition is None:
            log_partition1 = ConstantParameter(sl1.num_output_units, value=0.0)
        else:
            log_partition1 = sl1.log_partition.ref()
        if sl2.log_partition is None:
            log_partition2 = ConstantParameter(sl2.num_output_units, value=0.0)
        else:
            log_partition2 = sl2.log_partition.ref()
        log_partition = Parameter.from_binary(
            SumParameter(log_partition.shape, log_partition.shape),
            log_partition,
            Parameter.from_binary(
                OuterSumParameter(log_partition1.shape, log_partition2.shape, axis=0),
                log_partition1,
                log_partition2,
            ),
        )

    sl = GaussianLayer(
        sl1.scope,
        sl1.num_output_units * sl2.num_output_units,
        mean=mean,
        stddev=stddev,
        log_partition=log_partition,
    )
    return CircuitBlock.from_layer(sl)


def multiply_polynomial_layers(sl1: PolynomialLayer, sl2: PolynomialLayer) -> CircuitBlock:
    if sl1.scope != sl2.scope:
        raise ValueError(
            f"Expected Polynomial layers to have the same scope,"
            f" but found '{sl1.scope}' and '{sl2.scope}'"
        )

    shape1, shape2 = sl1.coeff.shape, sl2.coeff.shape
    coeff = Parameter.from_binary(
        PolynomialProduct(shape1, shape2),
        sl1.coeff.ref(),
        sl2.coeff.ref(),
    )

    sl = PolynomialLayer(
        sl1.scope,
        sl1.num_output_units * sl2.num_output_units,
        degree=sl1.degree + sl2.degree,
        coeff=coeff,
    )
    return CircuitBlock.from_layer(sl)


def multiply_hadamard_layers(sl1: HadamardLayer, sl2: HadamardLayer) -> CircuitBlock:
    arity = max(sl1.arity, sl2.arity)
    sl = HadamardLayer(
        sl1.num_input_units * sl2.num_input_units,
        arity=arity,
    )
    return CircuitBlock.from_layer(sl)


def multiply_kronecker_layers(sl1: KroneckerLayer, sl2: KroneckerLayer) -> CircuitBlock:
    arity = max(sl1.arity, sl2.arity)
    kron_sl = KroneckerLayer(sl1.num_input_units * sl2.num_input_units, arity=arity)
    # Start with a reshaped identity matrix
    perm_matrix = np.eye(kron_sl.num_output_units, dtype=np.float32).reshape(
        kron_sl.num_output_units,
        *((sl1.num_input_units,) * sl1.arity),
        *((sl2.num_input_units,) * sl2.arity),
    )
    # Construct the permutation matrix required to represent the product of
    # Kronecker layers as yet another Kronecker layer
    perm_matrix = np.transpose(
        perm_matrix, axes=sum([(1 + a, 1 + a + arity) for a in range(arity)], start=(0,))
    ).reshape(kron_sl.num_output_units, kron_sl.num_output_units)
    # The permutation matrix is applied by using a sum layer having the permutation
    # matrix has constant parameters
    sum_sl = SumLayer(
        kron_sl.num_output_units,
        kron_sl.num_output_units,
        weight=Parameter.from_input(
            ConstantParameter(kron_sl.num_output_units, kron_sl.num_output_units, value=perm_matrix)
        ),
    )
    return CircuitBlock.from_layer_composition(kron_sl, sum_sl)


def multiply_sum_layers(sl1: SumLayer, sl2: SumLayer) -> CircuitBlock:
    weight = Parameter.from_binary(
        KroneckerParameter(sl1.weight.shape, sl2.weight.shape), sl1.weight.ref(), sl2.weight.ref()
    )
    sl = SumLayer(
        sl1.num_input_units * sl2.num_input_units,
        sl1.num_output_units * sl2.num_output_units,
        arity=sl1.arity * sl2.arity,
        weight=weight,
    )
    return CircuitBlock.from_layer(sl)


def differentiate_polynomial_layer(
    sl: PolynomialLayer, *, var_idx: int, order: int = 1
) -> CircuitBlock:
    # PolynomialLayer is constructed univariate, but we still take the 2 idx for unified interface
    assert var_idx == 0, "This should not happen"
    if order <= 0:
        raise ValueError("The order of differentiation must be positive.")
    coeff = Parameter.from_unary(
        PolynomialDifferential(sl.coeff.shape, order=order), sl.coeff.ref()
    )
    sl = PolynomialLayer(sl.scope, sl.num_output_units, degree=coeff.shape[-1] - 1, coeff=coeff)
    return CircuitBlock.from_layer(sl)


def conjugate_embedding_layer(sl: EmbeddingLayer) -> CircuitBlock:
    weight = Parameter.from_unary(ConjugateParameter(sl.weight.shape), sl.weight.ref())
    sl = EmbeddingLayer(sl.scope, sl.num_output_units, num_states=sl.num_states, weight=weight)
    return CircuitBlock.from_layer(sl)


def conjugate_categorical_layer(sl: CategoricalLayer) -> CircuitBlock:
    logits = sl.logits.ref() if sl.logits is not None else None
    probs = sl.probs.ref() if sl.probs is not None else None
    sl = CategoricalLayer(
        sl.scope,
        sl.num_output_units,
        num_categories=sl.num_categories,
        logits=logits,
        probs=probs,
    )
    return CircuitBlock.from_layer(sl)


def conjugate_gaussian_layer(sl: GaussianLayer) -> CircuitBlock:
    mean = sl.mean.ref() if sl.mean is not None else None
    stddev = sl.stddev.ref() if sl.stddev is not None else None
    sl = GaussianLayer(sl.scope, sl.num_output_units, mean=mean, stddev=stddev)
    return CircuitBlock.from_layer(sl)


def conjugate_polynomial_layer(sl: PolynomialLayer) -> CircuitBlock:
    coeff = Parameter.from_unary(ConjugateParameter(sl.coeff.shape), sl.coeff.ref())
    sl = PolynomialLayer(sl.scope, sl.num_output_units, degree=sl.degree, coeff=coeff)
    return CircuitBlock.from_layer(sl)


def conjugate_sum_layer(sl: SumLayer) -> CircuitBlock:
    weight = Parameter.from_unary(ConjugateParameter(sl.weight.shape), sl.weight.ref())
    sl = SumLayer(sl.num_input_units, sl.num_output_units, arity=sl.arity, weight=weight)
    return CircuitBlock.from_layer(sl)


class LayerOperatorFunc(Protocol):
    """The layer operator function protocol."""

    def __call__(self, *sl: Layer, **kwargs) -> CircuitBlock:
        """Apply an operator on one or more layers.

        Args:
            *sl: The sequence of layers.
            **kwargs: The hyperparameters of the operator.

        Returns:
            A circuit block representing the sub computational graph resulting from
                the application of the operator.
        """


DEFAULT_OPERATOR_RULES: dict[LayerOperator, list[LayerOperatorFunc]] = {
    LayerOperator.INTEGRATION: [
        integrate_embedding_layer,
        integrate_categorical_layer,
        integrate_gaussian_layer,
    ],
    LayerOperator.DIFFERENTIATION: [differentiate_polynomial_layer],
    LayerOperator.MULTIPLICATION: [
        multiply_embedding_layers,
        multiply_categorical_layers,
        multiply_gaussian_layers,
        multiply_polynomial_layers,
        multiply_hadamard_layers,
        multiply_kronecker_layers,
        multiply_sum_layers,
    ],
    LayerOperator.CONJUGATION: [
        conjugate_embedding_layer,
        conjugate_categorical_layer,
        conjugate_gaussian_layer,
        conjugate_polynomial_layer,
        conjugate_sum_layer,
    ],
}
LayerOperatorSign = tuple[type[Layer], ...]
LayerOperatorSpecs = dict[LayerOperatorSign, LayerOperatorFunc]
