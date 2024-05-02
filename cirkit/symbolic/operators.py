from typing import Dict, Iterable, List, Optional, Protocol, Tuple, Type, Union

from cirkit.symbolic.circuit import CircuitBlock
from cirkit.symbolic.layers import (
    AbstractLayerOperator,
    CategoricalLayer,
    DenseLayer,
    GaussianLayer,
    GaussianProductLayer,
    HadamardLayer,
    IndexLayer,
    KroneckerLayer,
    Layer,
    LayerOperation,
    LogPartitionLayer,
    MixingLayer,
    PlaceholderParameter,
)
from cirkit.symbolic.params import (
    ConstantParameter,
    KroneckerParameter,
    LogPartitionGaussianProduct,
    MeanGaussianProduct,
    OuterSumParameter,
    ReduceLSEParameter,
    ReduceSumParameter,
    StddevGaussianProduct,
)
from cirkit.utils.scope import Scope


def integrate_categorical_layer(
    sl: CategoricalLayer, scope: Optional[Iterable[int]] = None
) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    assert sl.scope == scope
    lp = ReduceSumParameter(
        ReduceSumParameter(ReduceLSEParameter(PlaceholderParameter(sl, "logits"), axis=-1), axis=2),
        axis=0,
    )
    sl = LogPartitionLayer(sl.scope, sl.num_output_units, sl.num_channels, value=lp)
    return CircuitBlock.from_layer(sl)


def integrate_gaussian_layer(
    sl: GaussianLayer, scope: Optional[Iterable[int]] = None
) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    assert sl.scope == scope
    lp = ConstantParameter(shape=(sl.num_output_units,), value=0.0)
    sl = LogPartitionLayer(sl.scope, sl.num_output_units, sl.num_channels, value=lp)
    return CircuitBlock.from_layer(sl)


def integrate_gproduct_layer(
    sl: GaussianProductLayer, scope: Optional[Iterable[int]] = None
) -> CircuitBlock:
    scope = Scope(scope) if scope is not None else sl.scope
    assert sl.scope == scope
    lp = PlaceholderParameter(sl, name="log_partition")
    sl = LogPartitionLayer(sl.scope, sl.num_output_units, sl.num_channels, value=lp)
    return CircuitBlock.from_layer(sl)


def multiply_categorical_layers(lhs: CategoricalLayer, rhs: CategoricalLayer) -> CircuitBlock:
    assert lhs.num_variables == rhs.num_variables
    assert lhs.num_channels == rhs.num_channels
    sl_logits = OuterSumParameter(
        PlaceholderParameter(lhs, name="logits"), PlaceholderParameter(rhs, name="logits"), axis=1
    )
    sl = CategoricalLayer(
        lhs.scope | rhs.scope,
        lhs.num_output_units * rhs.num_output_units,
        num_channels=lhs.num_channels,
        logits=sl_logits,
    )
    return CircuitBlock.from_layer(sl)


# TODO: the operators registry should register the cross product of signatures,
#  when Union types are specified in the args, e.g., see below
#
def _multiply_unnormalized_gaussian_layers(
    lhs: Union[GaussianLayer, GaussianProductLayer], rhs: Union[GaussianLayer, GaussianProductLayer]
) -> CircuitBlock:
    assert lhs.num_variables == rhs.num_variables
    assert lhs.num_channels == rhs.num_channels

    # Retrieve placeholders
    if isinstance(lhs, GaussianLayer):
        m_lhs = [PlaceholderParameter(lhs, name="mean")]
        s_lhs = [PlaceholderParameter(lhs, name="stddev")]
    else:
        m_lhs = lhs.mean.mean_ls
        s_lhs = lhs.stddev.stddev_ls

    if isinstance(rhs, GaussianLayer):
        m_rhs = [PlaceholderParameter(rhs, name="mean")]
        s_rhs = [PlaceholderParameter(rhs, name="stddev")]
    else:
        m_rhs = rhs.mean.mean_ls
        s_rhs = rhs.stddev.stddev_ls

    # Build a "product of gaussian layers" layer
    mean_ls = m_lhs + m_rhs
    stddev_ls = s_lhs + s_rhs
    sl = GaussianProductLayer(
        lhs.scope | rhs.scope,
        lhs.num_output_units * rhs.num_output_units,
        num_channels=lhs.num_channels,
        mean=MeanGaussianProduct(mean_ls, stddev_ls),
        stddev=StddevGaussianProduct(stddev_ls),
        log_partition=LogPartitionGaussianProduct(mean_ls, stddev_ls),
    )
    return CircuitBlock.from_layer(sl)


def multiply_gaussian_layers(lhs: GaussianLayer, rhs: GaussianLayer) -> CircuitBlock:
    return _multiply_unnormalized_gaussian_layers(lhs, rhs)


def multiply_gproduct_gaussian_layer(lhs: GaussianProductLayer, rhs: GaussianLayer) -> CircuitBlock:
    return _multiply_unnormalized_gaussian_layers(lhs, rhs)


def multiply_gaussian_gproduct_layer(lhs: GaussianLayer, rhs: GaussianProductLayer) -> CircuitBlock:
    return _multiply_unnormalized_gaussian_layers(lhs, rhs)


def multiply_gproduct_gproduct_layer(
    lhs: GaussianProductLayer, rhs: GaussianProductLayer
) -> CircuitBlock:
    return _multiply_unnormalized_gaussian_layers(lhs, rhs)


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


class LayerOperatorFunc(Protocol):
    def __call__(self, *sl: Layer, **kwargs) -> CircuitBlock:
        ...


DEFAULT_OPERATOR_RULES: Dict[AbstractLayerOperator, List[LayerOperatorFunc]] = {
    LayerOperation.INTEGRATION: [
        integrate_categorical_layer,
        integrate_gaussian_layer,
        integrate_gproduct_layer,
    ],
    LayerOperation.DIFFERENTIATION: [],
    LayerOperation.MULTIPLICATION: [
        multiply_categorical_layers,
        multiply_gaussian_layers,
        multiply_hadamard_layers,
        multiply_kronecker_layers,
        multiply_dense_layers,
        multiply_mixing_layers,
        multiply_gaussian_layers,
        multiply_gproduct_gaussian_layer,
        multiply_gaussian_gproduct_layer,
        multiply_gproduct_gproduct_layer,
    ],
}
LayerOperatorSign = Tuple[Type[Layer], ...]
LayerOperatorSpecs = Dict[LayerOperatorSign, LayerOperatorFunc]
