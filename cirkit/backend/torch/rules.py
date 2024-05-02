from typing import TYPE_CHECKING, Dict, Optional

from cirkit.backend.base import (
    LayerCompilationFunc,
    LayerCompilationSign,
    ParameterCompilationFunc,
    ParameterCompilationSign,
)
from cirkit.backend.torch.layers import (
    TorchCategoricalLayer,
    TorchGaussianLayer,
    TorchLogPartitionLayer,
)
from cirkit.backend.torch.layers.inner import (
    TorchDenseLayer,
    TorchHadamardLayer,
    TorchKroneckerLayer,
    TorchMixingLayer,
)
from cirkit.backend.torch.params import TorchParameter
from cirkit.backend.torch.params.base import AbstractTorchParameter
from cirkit.backend.torch.params.composed import (
    TorchExpParameter,
    TorchKroneckerParameter,
    TorchLogSoftmaxParameter,
    TorchOuterSumParameter,
    TorchReduceLSEParameter,
    TorchReduceSumParameter,
    TorchScaledSigmoidParameter,
    TorchSoftmaxParameter,
)
from cirkit.backend.torch.params.ef import (
    TorchLogPartitionGaussianProductParameter,
    TorchMeanGaussianProductParameter,
    TorchStddevGaussianProductParameter,
)
from cirkit.backend.torch.params.parameter import TorchConstantParameter
from cirkit.backend.torch.utils import InitializerFunc
from cirkit.symbolic.layers import (
    CategoricalLayer,
    DenseLayer,
    GaussianLayer,
    GaussianProductLayer,
    HadamardLayer,
    KroneckerLayer,
    LogPartitionLayer,
    MixingLayer,
    PlaceholderParameter,
)
from cirkit.symbolic.params import (
    ConstantParameter,
    ExpParameter,
    KroneckerParameter,
    LogPartitionGaussianProduct,
    LogSoftmaxParameter,
    MeanGaussianProduct,
    OuterSumParameter,
    Parameter,
    ReduceLSEParameter,
    ReduceSumParameter,
    ScaledSigmoidParameter,
    SoftmaxParameter,
    StddevGaussianProduct,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_log_partition_layer(
    compiler: "TorchCompiler", sl: LogPartitionLayer
) -> TorchLogPartitionLayer:
    value = compiler.compile_parameter(sl.value)
    return TorchLogPartitionLayer(
        sl.num_variables,
        sl.num_output_units,
        num_channels=sl.num_channels,
        value=value,
        semiring=compiler.semiring,
    )


def compile_categorical_layer(
    compiler: "TorchCompiler", sl: CategoricalLayer
) -> TorchCategoricalLayer:
    logits = compiler.compile_parameter(
        sl.logits, init_func=TorchCategoricalLayer.get_default_initializer("logits")
    )
    return TorchCategoricalLayer(
        sl.num_variables,
        sl.num_output_units,
        num_channels=sl.num_channels,
        num_categories=sl.num_categories,
        logits=logits,
        semiring=compiler.semiring,
    )


def compile_gaussian_layer(compiler: "TorchCompiler", sl: GaussianLayer) -> TorchGaussianLayer:
    mean = compiler.compile_parameter(
        sl.mean, init_func=TorchGaussianLayer.get_default_initializer("mean")
    )
    stddev = compiler.compile_parameter(
        sl.stddev, init_func=TorchGaussianLayer.get_default_initializer("stddev")
    )
    return TorchGaussianLayer(
        sl.num_variables,
        sl.num_output_units,
        num_channels=sl.num_channels,
        mean=mean,
        stddev=stddev,
        semiring=compiler.semiring,
    )


def compile_hadamard_layer(compiler: "TorchCompiler", sl: KroneckerLayer) -> TorchHadamardLayer:
    return TorchHadamardLayer(
        sl.num_input_units, sl.num_output_units, arity=sl.arity, semiring=compiler.semiring
    )


def compile_kronecker_layer(compiler: "TorchCompiler", sl: KroneckerLayer) -> TorchKroneckerLayer:
    return TorchKroneckerLayer(
        sl.num_input_units, sl.num_output_units, arity=sl.arity, semiring=compiler.semiring
    )


def compile_dense_layer(compiler: "TorchCompiler", sl: DenseLayer) -> TorchDenseLayer:
    weight = compiler.compile_parameter(
        sl.weight, init_func=TorchDenseLayer.get_default_initializer("weight")
    )
    return TorchDenseLayer(
        sl.num_input_units, sl.num_output_units, weight=weight, semiring=compiler.semiring
    )


def compile_mixing_layer(compiler: "TorchCompiler", sl: MixingLayer) -> TorchMixingLayer:
    weight = compiler.compile_parameter(
        sl.weight, init_func=TorchMixingLayer.get_default_initializer("weight")
    )
    return TorchMixingLayer(
        sl.num_input_units,
        sl.num_output_units,
        arity=sl.arity,
        weight=weight,
        semiring=compiler.semiring,
    )


def compile_gproduct_layer(
    compiler: "TorchCompiler", sl: GaussianProductLayer
) -> TorchGaussianLayer:
    return TorchGaussianLayer(
        sl.num_variables,
        sl.num_output_units,
        num_channels=sl.num_channels,
        mean=compiler.compile_parameter(sl.mean),
        stddev=compiler.compile_parameter(sl.stddev),
        log_partition=compiler.compile_parameter(sl.log_partition),
        semiring=compiler.semiring,
    )


def compile_parameter(
    compiler: "TorchCompiler", p: Parameter, init_func: Optional[InitializerFunc] = None
) -> TorchParameter:
    compiled_p = TorchParameter(*p.shape)
    if init_func is not None:
        compiled_p.initialize(init_func)
    return compiled_p


def compile_placeholder_parameter(
    compiler: "TorchCompiler", p: PlaceholderParameter, init_func: Optional[InitializerFunc] = None
) -> AbstractTorchParameter:
    return compiler.retrieve_parameter(p.layer, p.name)


def compile_constant_parameter(
    compiler: "TorchCompiler", p: ConstantParameter, init_func: Optional[InitializerFunc] = None
) -> TorchParameter:
    return TorchConstantParameter(p.shape, p.value)


def compile_exp_parameter(
    compiler: "TorchCompiler", p: ExpParameter, init_func: Optional[InitializerFunc] = None
) -> TorchExpParameter:
    new_init_func = (lambda t: init_func(t).log_()) if init_func is not None else None
    opd = compiler.compile_parameter(p.opd, init_func=new_init_func)
    return TorchExpParameter(opd)


def compile_softmax_parameter(
    compiler: "TorchCompiler", p: SoftmaxParameter, init_func: Optional[InitializerFunc] = None
) -> TorchSoftmaxParameter:
    new_init_func = (lambda t: init_func(t).log_()) if init_func is not None else None
    opd = compiler.compile_parameter(p.opd, init_func=new_init_func)
    return TorchSoftmaxParameter(opd, dim=p.axis)


def compile_log_softmax_parameter(
    compiler: "TorchCompiler", p: SoftmaxParameter, init_func: Optional[InitializerFunc] = None
) -> TorchLogSoftmaxParameter:
    new_init_func = (lambda t: init_func(t).exp_()) if init_func is not None else None
    opd = compiler.compile_parameter(p.opd, init_func=new_init_func)
    return TorchLogSoftmaxParameter(opd, dim=p.axis)


def compile_reduce_sum_parameter(
    compiler: "TorchCompiler", p: ReduceSumParameter, init_func: Optional[InitializerFunc] = None
) -> TorchReduceSumParameter:
    opd = compiler.compile_parameter(p.opd, init_func=init_func)
    return TorchReduceSumParameter(opd, dim=p.axis)


def compile_reduce_lse_parameter(
    compiler: "TorchCompiler", p: ReduceSumParameter, init_func: Optional[InitializerFunc] = None
) -> TorchReduceLSEParameter:
    opd = compiler.compile_parameter(p.opd, init_func=init_func)
    return TorchReduceLSEParameter(opd, dim=p.axis)


def compile_outer_sum_parameter(
    compiler: "TorchCompiler", p: OuterSumParameter, init_func: Optional[InitializerFunc] = None
) -> TorchOuterSumParameter:
    opd1 = compiler.compile_parameter(p.opd1)
    opd2 = compiler.compile_parameter(p.opd2)
    return TorchOuterSumParameter(opd1, opd2, dim=p.axis)


def compile_kronecker_parameter(
    compiler: "TorchCompiler",
    p: KroneckerParameter,
    init_func: Optional[InitializerFunc] = None,
) -> TorchKroneckerParameter:
    opd1 = compiler.compile_parameter(p.opd1)
    opd2 = compiler.compile_parameter(p.opd2)
    return TorchKroneckerParameter(opd1, opd2)


def compile_scaled_sigmoid_paramater(
    compiler: "TorchCompiler",
    p: ScaledSigmoidParameter,
    init_func: Optional[InitializerFunc] = None,
) -> TorchScaledSigmoidParameter:
    # TODO: invert scaled sigmoid
    opd = compiler.compile_parameter(p.opd, init_func=init_func)
    return TorchScaledSigmoidParameter(opd, vmin=p.vmin, vmax=p.vmax)


# def compile_gproduct_parameters(
#     compiler: "TorchCompiler", p: LogPartitionGaussianProduct
# ) -> TorchGaussianProductParameters:
#     compiled_m_ls = [
#         compiler.compile_parameter(m, init_func=TorchGaussianLayer.get_default_initializer("mean"))
#         for m in p.mean_ls
#     ]
#     compiled_s_ls = [
#         compiler.compile_parameter(
#             s, init_func=TorchGaussianLayer.get_default_initializer("stddev")
#         )
#         for s in p.stddev_ls
#     ]
#     return TorchGaussianProductParameters(compiled_m_ls, compiled_s_ls)


def compile_mean_gaussian_product_parameter(
    compiler: "TorchCompiler",
    p: MeanGaussianProduct,
    init_func: Optional[InitializerFunc] = None,
) -> TorchMeanGaussianProductParameter:
    mean_ls = [compiler.compile_parameter(m) for m in p.mean_ls]
    stddev_ls = [compiler.compile_parameter(m) for m in p.stddev_ls]
    return TorchMeanGaussianProductParameter(mean_ls, stddev_ls)


def compile_stddev_gaussian_product_parameter(
    compiler: "TorchCompiler",
    p: StddevGaussianProduct,
    init_func: Optional[InitializerFunc] = None,
) -> TorchStddevGaussianProductParameter:
    stddev_ls = [compiler.compile_parameter(m) for m in p.stddev_ls]
    return TorchStddevGaussianProductParameter(stddev_ls)


def compile_log_partition_gaussian_product_parameter(
    compiler: "TorchCompiler",
    p: LogPartitionGaussianProduct,
    init_func: Optional[InitializerFunc] = None,
) -> TorchLogPartitionGaussianProductParameter:
    mean_ls = [compiler.compile_parameter(m) for m in p.mean_ls]
    stddev_ls = [compiler.compile_parameter(m) for m in p.stddev_ls]
    return TorchLogPartitionGaussianProductParameter(mean_ls, stddev_ls)


DEFAULT_LAYER_COMPILATION_RULES: Dict[LayerCompilationSign, LayerCompilationFunc] = {  # type: ignore[misc]
    LogPartitionLayer: compile_log_partition_layer,
    CategoricalLayer: compile_categorical_layer,
    GaussianLayer: compile_gaussian_layer,
    HadamardLayer: compile_hadamard_layer,
    KroneckerLayer: compile_kronecker_layer,
    DenseLayer: compile_dense_layer,
    MixingLayer: compile_mixing_layer,
    GaussianProductLayer: compile_gproduct_layer,
}
DEFAULT_PARAMETER_COMPILATION_RULES: Dict[ParameterCompilationSign, ParameterCompilationFunc] = {  # type: ignore[misc]
    Parameter: compile_parameter,
    PlaceholderParameter: compile_placeholder_parameter,
    ConstantParameter: compile_constant_parameter,
    ExpParameter: compile_exp_parameter,
    SoftmaxParameter: compile_softmax_parameter,
    LogSoftmaxParameter: compile_log_softmax_parameter,
    ReduceSumParameter: compile_reduce_sum_parameter,
    ReduceLSEParameter: compile_reduce_lse_parameter,
    OuterSumParameter: compile_outer_sum_parameter,
    KroneckerParameter: compile_kronecker_parameter,
    ScaledSigmoidParameter: compile_scaled_sigmoid_paramater,
    MeanGaussianProduct: compile_mean_gaussian_product_parameter,
    StddevGaussianProduct: compile_stddev_gaussian_product_parameter,
    LogPartitionGaussianProduct: compile_log_partition_gaussian_product_parameter,
}
