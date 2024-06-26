from typing import Dict, Iterable, List, Optional, Protocol, Tuple, Type

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import (
    AbstractLayerOperator,
    CategoricalLayer,
    DenseLayer,
    HadamardLayer,
    IndexLayer,
    KroneckerLayer,
    Layer,
    LayerOperation,
    LogPartitionLayer,
    MixingLayer,
)
from cirkit.symbolic.parameters import (
    ConstantParameter,
    KroneckerParameter,
    LogParameter,
    OuterSumParameter,
    Parameter,
    ReduceLSEParameter,
    ReduceSumParameter,
)
from cirkit.utils.scope import Scope


def integrate_categorical_layer(
    sl: CategoricalLayer, scope: Optional[Iterable[int]] = None
) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    assert sl.scope == scope
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


def multiply_categorical_layers(sl1: CategoricalLayer, sl2: CategoricalLayer) -> CircuitBlock:
    assert sl1.num_variables == sl2.num_variables
    assert sl1.num_channels == sl2.num_channels
    if sl1.logits is None:
        sl1_logits = Parameter.from_unary(sl1.probs, LogParameter(sl1.probs.shape))
    else:
        sl1_logits = sl1.logits
    if sl2.logits is None:
        sl2_logits = Parameter.from_unary(sl2.probs, LogParameter(sl2.probs.shape))
    else:
        sl2_logits = sl2.logits
    sl_logits = Parameter.from_binary(
        sl1_logits.ref(),
        sl2_logits.ref(),
        OuterSumParameter(sl1_logits.shape, sl2_logits.shape, axis=1),
    )
    sl = CategoricalLayer(
        sl1.scope | sl2.scope,
        sl1.num_output_units * sl2.num_output_units,
        num_channels=sl1.num_channels,
        logits=sl_logits,
    )
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
        sl1.weight.ref(), sl2.weight.ref(), KroneckerParameter(sl1.weight.shape, sl2.weight.shape)
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
        sl1.weight.ref(), sl2.weight.ref(), KroneckerParameter(sl1.weight.shape, sl2.weight.shape)
    )
    sl = MixingLayer(
        sl1.scope | sl2.scope,
        sl1.num_input_units * sl2.num_input_units,
        sl1.arity * sl2.arity,
        weight=weight,
    )
    return CircuitBlock.from_layer(sl)


class LayerOperatorFunc(Protocol):
    def __call__(self, *sl: Layer, **kwargs) -> CircuitBlock:
        ...


DEFAULT_OPERATOR_RULES: Dict[AbstractLayerOperator, List[LayerOperatorFunc]] = {
    LayerOperation.INTEGRATION: [
        integrate_categorical_layer,
    ],
    LayerOperation.DIFFERENTIATION: [],
    LayerOperation.MULTIPLICATION: [
        multiply_categorical_layers,
        multiply_hadamard_layers,
        multiply_kronecker_layers,
        multiply_dense_layers,
        multiply_mixing_layers,
    ],
}
LayerOperatorSign = Tuple[Type[Layer], ...]
LayerOperatorSpecs = Dict[LayerOperatorSign, LayerOperatorFunc]
