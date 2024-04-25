from typing import Dict, Iterable, List, Optional, Union

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import (
    AbstractLayerOperator,
    CategoricalLayer,
    ConstantLayer,
    DenseLayer,
    ExpFamilyLayer,
    HadamardLayer,
    IndexLayer,
    KroneckerLayer,
    LayerOperation,
    MixingLayer,
    NormalLayer,
    PlaceholderParameter,
)
from cirkit.symbolic.params import ConstantParameter, KroneckerParameter, ReduceSumParameter
from cirkit.symbolic.registry import LayerOperatorFunc
from cirkit.utils.scope import Scope


def integrate_ef_layer(sl: ExpFamilyLayer, scope: Optional[Iterable[int]] = None) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    assert sl.scope == scope
    if sl.part is None:
        sv = ConstantParameter(1.0, shape=(sl.num_output_units,))
    else:
        sv = PlaceholderParameter(sl, name="part")
    sl = ConstantLayer(sl.scope, sl.num_output_units, sl.num_channels, value=sv)
    return CircuitBlock.from_layer(sl)


def multiply_categorical_layers(lhs: CategoricalLayer, rhs: CategoricalLayer) -> CircuitBlock:
    assert lhs.num_channels == rhs.num_channels
    sl_probs = KroneckerParameter(
        PlaceholderParameter(lhs, name="probs"), PlaceholderParameter(rhs, name="probs"), axis=2
    )
    sl = CategoricalLayer(
        lhs.scope | rhs.scope,
        lhs.num_output_units * rhs.num_input_units,
        num_channels=lhs.num_channels,
        probs=sl_probs,
        part=ReduceSumParameter(sl_probs, axis=3),
    )
    return CircuitBlock.from_layer(sl)


def multiply_normal_layers(lhs: NormalLayer, rhs: NormalLayer) -> CircuitBlock:
    assert lhs.num_channels == rhs.num_channels
    sl = NormalLayer(
        lhs.scope | rhs.scope,
        lhs.num_output_units * rhs.num_input_units,
        num_channels=lhs.num_channels,
        mean=None,
        variance=None,
        part=None,  # What to put here?
    )
    return CircuitBlock.from_layer(sl)


def multiply_hadamard_layers(lhs: HadamardLayer, rhs: HadamardLayer) -> CircuitBlock:
    sl = HadamardLayer(
        lhs.scope | rhs.scope,
        lhs.num_input_units * rhs.num_input_units,
        arity=max(lhs.arity, rhs.arity),
    )
    return CircuitBlock.from_layer(sl)


def multiply_kronecker_layers(lhs: KroneckerLayer, rhs: KroneckerLayer) -> CircuitBlock:
    sl = KroneckerLayer(
        lhs.scope | rhs.scope,
        lhs.num_input_units * rhs.num_input_units,
        arity=max(lhs.arity, rhs.arity),
    )
    # The product of kronecker layers is a kronecker layer followed by a permutation
    idx: List[int] = []  # TODO
    sil = IndexLayer(lhs.scope | rhs.scope, sl.num_output_units, sl.num_output_units, indices=idx)
    return CircuitBlock.from_layer_composition(sl, sil)


def multiply_dense_layers(lhs: DenseLayer, rhs: DenseLayer) -> CircuitBlock:
    sl = DenseLayer(
        lhs.scope | rhs.scope,
        lhs.num_input_units * rhs.num_input_units,
        lhs.num_output_units * rhs.num_output_units,
        weight=KroneckerParameter(
            PlaceholderParameter(lhs, name="weight"), PlaceholderParameter(rhs, name="weight")
        ),
    )
    return CircuitBlock.from_layer(sl)


def multiply_mixing_layers(lhs: MixingLayer, rhs: MixingLayer) -> CircuitBlock:
    sl = MixingLayer(
        lhs.scope | rhs.scope,
        lhs.num_input_units * rhs.num_input_units,
        lhs.arity * rhs.arity,
        weight=KroneckerParameter(
            PlaceholderParameter(lhs, name="weight"), PlaceholderParameter(rhs, name="weight")
        ),
    )
    return CircuitBlock.from_layer(sl)


DEFAULT_COMMUTATIVE_OPERATORS = [LayerOperation.MULTIPLICATION]

DEFAULT_OPERATOR_RULES: Dict[AbstractLayerOperator, List[LayerOperatorFunc]] = {
    LayerOperation.INTEGRATION: [
        integrate_ef_layer,
    ],
    LayerOperation.DIFFERENTIATION: [],
    LayerOperation.MULTIPLICATION: [
        multiply_categorical_layers,
        multiply_normal_layers,
        multiply_hadamard_layers,
        multiply_kronecker_layers,
        multiply_dense_layers,
        multiply_mixing_layers,
    ],
}
