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
    PlaceholderParameter,
)
from cirkit.symbolic.params import (
    ConstantParameter,
    EntrywiseSumParameter,
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
    if sl.log_partition is None:
        lp = ConstantParameter(shape=(sl.num_output_units,), value=0.0)
    else:
        lp = PlaceholderParameter(sl, "log_partition")
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


def multiply_gaussian_layers(lhs: GaussianLayer, rhs: GaussianLayer) -> CircuitBlock:
    assert lhs.num_variables == rhs.num_variables
    assert lhs.num_channels == rhs.num_channels

    # Retrieve placeholders and/or aggregators of means and standard deviations
    if isinstance(lhs.mean, MeanGaussianProduct):
        assert isinstance(lhs.stddev, StddevGaussianProduct)
        mean_ls1 = lhs.mean.mean_ls
        stddev_ls1 = lhs.stddev.stddev_ls
        log_partition1 = None
    else:
        mean_ls1 = [PlaceholderParameter(lhs, name="mean")]
        stddev_ls1 = [PlaceholderParameter(lhs, name="stddev")]
        log_partition1 = PlaceholderParameter(lhs, name="log_partition")
    if isinstance(rhs.mean, MeanGaussianProduct):
        assert isinstance(rhs.stddev, StddevGaussianProduct)
        mean_ls2 = rhs.mean.mean_ls
        stddev_ls2 = rhs.stddev.stddev_ls
        log_partition2 = None
    else:
        mean_ls2 = [PlaceholderParameter(rhs, name="mean")]
        stddev_ls2 = [PlaceholderParameter(rhs, name="stddev")]
        log_partition2 = PlaceholderParameter(rhs, name="log_partition")

    # Build a new gaussian layer
    mean_ls = mean_ls1 + mean_ls2
    stddev_ls = stddev_ls1 + stddev_ls2
    log_partition = LogPartitionGaussianProduct(mean_ls, stddev_ls)
    oth_log_partitions = [lp for lp in [log_partition1, log_partition2] if lp is not None]
    if oth_log_partitions:
        log_partition = EntrywiseSumParameter(log_partition, *oth_log_partitions)
    sl = GaussianLayer(
        lhs.scope | rhs.scope,
        lhs.num_output_units * rhs.num_input_units,
        num_channels=lhs.num_channels,
        mean=MeanGaussianProduct(mean_ls, stddev_ls),
        stddev=StddevGaussianProduct(stddev_ls),
        log_partition=log_partition,
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
}
LayerOperatorSign = Tuple[Type[Layer], ...]
LayerOperatorSpecs = Dict[LayerOperatorSign, LayerOperatorFunc]
