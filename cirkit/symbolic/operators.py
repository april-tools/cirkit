from typing import Dict, Iterable, List, Optional, Protocol, Tuple, Type

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import (
    AbstractLayerOperator,
    CategoricalLayer,
    DenseLayer,
    GaussianLayer,
    HadamardLayer,
    IndexLayer,
    KroneckerLayer,
    Layer,
    LayerOperation,
    LogPartitionLayer,
    MixingLayer,
)
from cirkit.symbolic.parameters import (
    ConjugateParameter,
    ConstantParameter,
    GaussianProductLogPartition,
    GaussianProductMean,
    GaussianProductStddev,
    KroneckerParameter,
    LogParameter,
    OuterSumParameter,
    Parameter,
    ReduceLSEParameter,
    ReduceSumParameter,
    SumParameter,
)
from cirkit.utils.scope import Scope


def integrate_categorical_layer(
    sl: CategoricalLayer, scope: Optional[Iterable[int]] = None
) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    if sl.scope != scope:
        raise NotImplementedError()
    if sl.logits is None:
        log_partition = Parameter.from_leaf(ConstantParameter(sl.num_output_units, value=0.0))
    else:
        reduce_lse = ReduceLSEParameter(sl.logits.shape, axis=3)
        reduce_sum1 = ReduceSumParameter(reduce_lse.shape, axis=2)
        reduce_sum2 = ReduceSumParameter(reduce_sum1.shape, axis=0)
        log_partition = Parameter.from_sequence(
            sl.logits.ref(), reduce_lse, reduce_sum1, reduce_sum2
        )
    sl = LogPartitionLayer(sl.scope, sl.num_output_units, sl.num_channels, value=log_partition)
    return CircuitBlock.from_layer(sl)


def integrate_gaussian_layer(
    sl: GaussianLayer, scope: Optional[Iterable[int]] = None
) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    if sl.scope != scope:
        raise NotImplementedError()
    if sl.log_partition is None:
        log_partition = Parameter.from_leaf(ConstantParameter(sl.num_output_units, value=0.0))
    else:
        reduce_sum1 = ReduceSumParameter(sl.log_partition.shape, axis=2)
        reduce_sum2 = ReduceSumParameter(reduce_sum1.shape, axis=0)
        log_partition = Parameter.from_sequence(sl.log_partition.ref(), reduce_sum1, reduce_sum2)
    sl = LogPartitionLayer(sl.scope, sl.num_output_units, sl.num_channels, value=log_partition)
    return CircuitBlock.from_layer(sl)


def multiply_categorical_layers(sl1: CategoricalLayer, sl2: CategoricalLayer) -> CircuitBlock:
    assert sl1.num_variables == sl2.num_variables
    assert sl1.num_channels == sl2.num_channels
    assert sl1.num_categories == sl2.num_categories
    if sl1.logits is None:
        sl1_logits = Parameter.from_unary(LogParameter(sl1.probs.shape), sl1.probs)
    else:
        sl1_logits = sl1.logits
    if sl2.logits is None:
        sl2_logits = Parameter.from_unary(LogParameter(sl2.probs.shape), sl2.probs)
    else:
        sl2_logits = sl2.logits
    sl_logits = Parameter.from_binary(
        OuterSumParameter(sl1_logits.shape, sl2_logits.shape, axis=1),
        sl1_logits.ref(),
        sl2_logits.ref(),
    )
    sl = CategoricalLayer(
        sl1.scope | sl2.scope,
        sl1.num_output_units * sl2.num_output_units,
        num_channels=sl1.num_channels,
        num_categories=sl1.num_categories,
        logits=sl_logits,
    )
    return CircuitBlock.from_layer(sl)


def multiply_gaussian_layers(sl1: GaussianLayer, sl2: GaussianLayer) -> CircuitBlock:
    assert sl1.num_variables == sl2.num_variables
    assert sl1.num_channels == sl2.num_channels

    gaussian1_shape, gaussian2_shape = sl1.mean.shape, sl2.mean.shape
    mean = Parameter.from_nary(
        GaussianProductMean(gaussian1_shape, gaussian2_shape),
        sl1.mean.ref(),
        sl2.mean.ref(),
        sl1.stddev.ref(),
        sl2.stddev.ref(),
    )
    stddev = Parameter.from_binary(
        GaussianProductStddev(gaussian1_shape, gaussian2_shape),
        sl1.stddev.ref(),
        sl2.stddev.ref(),
    )
    log_partition = Parameter.from_nary(
        GaussianProductLogPartition(gaussian1_shape, gaussian2_shape),
        sl1.mean.ref(),
        sl2.mean.ref(),
        sl1.stddev.ref(),
        sl2.stddev.ref(),
    )

    if sl1.log_partition is not None or sl2.log_partition is not None:
        if sl1.log_partition is None:
            log_partition1 = ConstantParameter(
                sl1.num_variables, sl1.num_output_units, sl1.num_channels, value=0.0
            )
        else:
            log_partition1 = sl1.log_partition.ref()
        if sl2.log_partition is None:
            log_partition2 = ConstantParameter(
                sl2.num_variables, sl2.num_output_units, sl2.num_channels, value=0.0
            )
        else:
            log_partition2 = sl2.log_partition.ref()
        log_partition = Parameter.from_binary(
            SumParameter(log_partition.shape, log_partition.shape),
            log_partition,
            Parameter.from_binary(
                OuterSumParameter(log_partition1.shape, log_partition2.shape, axis=1),
                log_partition1,
                log_partition2,
            ),
        )

    sl = GaussianLayer(
        sl1.scope | sl2.scope,
        sl1.num_output_units * sl2.num_output_units,
        num_channels=sl1.num_channels,
        mean=mean,
        stddev=stddev,
        log_partition=log_partition,
    )
    return CircuitBlock.from_layer(sl)


def conjugate_categorical_layer(sl: CategoricalLayer) -> CircuitBlock:
    logits = sl.logits.ref() if sl.logits is not None else None
    probs = sl.probs.ref() if sl.probs is not None else None
    sl = CategoricalLayer(
        sl.scope,
        sl.num_output_units,
        sl.num_channels,
        num_categories=sl.num_categories,
        logits=logits,
        probs=probs,
    )
    return CircuitBlock.from_layer(sl)


def conjugate_gaussian_layer(sl: GaussianLayer) -> CircuitBlock:
    mean = sl.mean.ref() if sl.mean is not None else None
    stddev = sl.stddev.ref() if sl.stddev is not None else None
    sl = GaussianLayer(sl.scope, sl.num_output_units, sl.num_channels, mean=mean, stddev=stddev)
    return CircuitBlock.from_layer(sl)


def multiply_hadamard_layers(sl1: HadamardLayer, sl2: HadamardLayer) -> CircuitBlock:
    sl = HadamardLayer(
        sl1.scope | sl2.scope,
        sl1.num_input_units * sl2.num_input_units,
        arity=max(sl1.arity, sl2.arity),
    )
    return CircuitBlock.from_layer(sl)


def multiply_kronecker_layers(sl1: KroneckerLayer, sl2: KroneckerLayer) -> CircuitBlock:
    sl = KroneckerLayer(
        sl1.scope | sl2.scope,
        sl1.num_input_units * sl2.num_input_units,
        arity=max(sl1.arity, sl2.arity),
    )
    # The product of kronecker layers is a kronecker layer followed by a permutation
    idx: List[int] = []  # TODO
    sil = IndexLayer(sl1.scope | sl2.scope, sl.num_output_units, sl.num_output_units, indices=idx)
    return CircuitBlock.from_layer_composition(sl, sil)


def multiply_dense_layers(sl1: DenseLayer, sl2: DenseLayer) -> CircuitBlock:
    weight = Parameter.from_binary(
        KroneckerParameter(sl1.weight.shape, sl2.weight.shape), sl1.weight.ref(), sl2.weight.ref()
    )
    sl = DenseLayer(
        sl1.scope | sl2.scope,
        sl1.num_input_units * sl2.num_input_units,
        sl1.num_output_units * sl2.num_output_units,
        weight=weight,
    )
    return CircuitBlock.from_layer(sl)


def multiply_mixing_layers(sl1: MixingLayer, sl2: MixingLayer) -> CircuitBlock:
    weight = Parameter.from_binary(
        KroneckerParameter(sl1.weight.shape, sl2.weight.shape), sl1.weight.ref(), sl2.weight.ref()
    )
    sl = MixingLayer(
        sl1.scope | sl2.scope,
        sl1.num_input_units * sl2.num_input_units,
        sl1.arity * sl2.arity,
        weight=weight,
    )
    return CircuitBlock.from_layer(sl)


def conjugate_dense_layer(sl: DenseLayer) -> CircuitBlock:
    weight = Parameter.from_unary(ConjugateParameter(sl.weight.shape), sl.weight.ref())
    sl = DenseLayer(sl.scope, sl.num_input_units, sl.num_output_units, weight=weight)
    return CircuitBlock.from_layer(sl)


def conjugate_mixing_layer(sl: MixingLayer) -> CircuitBlock:
    weight = Parameter.from_unary(ConjugateParameter(sl.weight.shape), sl.weight.ref())
    sl = MixingLayer(sl.scope, sl.num_input_units, sl.arity, weight=weight)
    return CircuitBlock.from_layer(sl)


class LayerOperatorFunc(Protocol):
    def __call__(self, *sl: Layer, **kwargs) -> CircuitBlock:
        ...


DEFAULT_OPERATOR_RULES: Dict[AbstractLayerOperator, List[LayerOperatorFunc]] = {
    LayerOperation.INTEGRATION: [integrate_categorical_layer, integrate_gaussian_layer],
    LayerOperation.DIFFERENTIATION: [],
    LayerOperation.MULTIPLICATION: [
        multiply_categorical_layers,
        multiply_gaussian_layers,
        multiply_hadamard_layers,
        multiply_kronecker_layers,
        multiply_dense_layers,
        multiply_mixing_layers,
    ],
    LayerOperation.CONJUGATION: [
        conjugate_categorical_layer,
        conjugate_gaussian_layer,
        conjugate_dense_layer,
        conjugate_mixing_layer,
    ],
}
LayerOperatorSign = Tuple[Type[Layer], ...]
LayerOperatorSpecs = Dict[LayerOperatorSign, LayerOperatorFunc]
