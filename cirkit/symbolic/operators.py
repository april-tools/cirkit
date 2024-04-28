from typing import Iterable, List, Optional

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import (
    CategoricalLayer,
    ConstantLayer,
    DenseLayer,
    ExpFamilyLayer,
    GaussianLayer,
    HadamardLayer,
    IndexLayer,
    KroneckerLayer,
    MixingLayer,
    PlaceholderParameter,
)
from cirkit.symbolic.params import (
    ConstantParameter,
    ExpParameter,
    KroneckerParameter,
    LogParameter,
    MeanNormalProduct,
    OuterProductParameter,
    OuterSumParameter,
    PartitionGaussianProduct,
    ReduceSumParameter,
    VarianceNormalProduct,
)
from cirkit.utils.scope import Scope


def integrate_ef_layer(sl: ExpFamilyLayer, scope: Optional[Iterable[int]] = None) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    assert sl.scope == scope
    if sl.log_partition is None:
        sv = ConstantParameter(1.0, shape=(sl.num_output_units,))
    else:
        sv = PlaceholderParameter(sl, name="log_partition")
    sl = ConstantLayer(sl.scope, sl.num_output_units, sl.num_channels, value=sv)
    return CircuitBlock.from_layer(sl)


def multiply_categorical_layers(lhs: CategoricalLayer, rhs: CategoricalLayer) -> CircuitBlock:
    assert lhs.num_variables == rhs.num_variables
    assert lhs.num_channels == rhs.num_channels
    sl_logits = OuterSumParameter(
        PlaceholderParameter(lhs, name="logits"), PlaceholderParameter(rhs, name="logits"), axis=2
    )
    sl = CategoricalLayer(
        lhs.scope | rhs.scope,
        lhs.num_output_units * rhs.num_input_units,
        num_channels=lhs.num_channels,
        logits=sl_logits,
        log_partition=LogParameter(ReduceSumParameter(ExpParameter(sl_logits), axis=3)),
    )
    return CircuitBlock.from_layer(sl)


def multiply_gaussian_layers(lhs: GaussianLayer, rhs: GaussianLayer) -> CircuitBlock:
    assert lhs.num_variables == rhs.num_variables
    assert lhs.num_channels == rhs.num_channels
    mean_lhs = PlaceholderParameter(lhs, name="mean")
    mean_rhs = PlaceholderParameter(rhs, name="mean")
    variance_lhs = PlaceholderParameter(lhs, name="variance")
    variance_rhs = PlaceholderParameter(rhs, name="variance")
    sl = GaussianLayer(
        lhs.scope | rhs.scope,
        lhs.num_output_units * rhs.num_input_units,
        num_channels=lhs.num_channels,
        mean=MeanNormalProduct(mean_lhs, mean_rhs, variance_lhs, variance_rhs),
        stddev=VarianceNormalProduct(variance_lhs, variance_rhs),
        log_partition=PartitionGaussianProduct(mean_lhs, mean_rhs, variance_lhs, variance_rhs),
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
        weight=OuterProductParameter(
            PlaceholderParameter(lhs, name="weight"), PlaceholderParameter(rhs, name="weight")
        ),
    )
    return CircuitBlock.from_layer(sl)
