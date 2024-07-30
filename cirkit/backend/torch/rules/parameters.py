from typing import TYPE_CHECKING, Dict

import torch

from cirkit.backend.compiler import ParameterCompilationFunc, ParameterCompilationSign
from cirkit.backend.torch.parameters.nodes import (
    TorchClampParameter,
    TorchConjugateParameter,
    TorchExpParameter,
    TorchGaussianProductLogPartition,
    TorchGaussianProductMean,
    TorchGaussianProductStddev,
    TorchHadamardParameter,
    TorchKroneckerParameter,
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchOuterProductParameter,
    TorchOuterSumParameter,
    TorchPointerParameter,
    TorchReduceLSEParameter,
    TorchReduceProductParameter,
    TorchReduceSumParameter,
    TorchScaledSigmoidParameter,
    TorchSigmoidParameter,
    TorchSoftmaxParameter,
    TorchSquareParameter,
    TorchSumParameter,
    TorchTensorParameter,
)
from cirkit.symbolic.dtypes import DataType
from cirkit.symbolic.parameters import (
    ClampParameter,
    ConjugateParameter,
    ConstantParameter,
    ExpParameter,
    GaussianProductLogPartition,
    GaussianProductMean,
    GaussianProductStddev,
    HadamardParameter,
    KroneckerParameter,
    LogParameter,
    LogSoftmaxParameter,
    OuterProductParameter,
    OuterSumParameter,
    ReduceLSEParameter,
    ReduceProductParameter,
    ReduceSumParameter,
    ReferenceParameter,
    ScaledSigmoidParameter,
    SigmoidParameter,
    SoftmaxParameter,
    SquareParameter,
    SumParameter,
    TensorParameter,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def _retrieve_dtype(dtype: DataType) -> torch.dtype:
    if dtype == DataType.INTEGER:
        return torch.int64
    default_float_dtype = torch.get_default_dtype()
    if dtype == DataType.REAL:
        return default_float_dtype
    if dtype == DataType.COMPLEX:
        if default_float_dtype == torch.float32:
            return torch.complex64
        if default_float_dtype == torch.float64:
            return torch.complex128
    raise ValueError(
        f"Cannot determine the torch.dtype to use, current default: {default_float_dtype}, given dtype: {dtype}"
    )


def compile_tensor_parameter(compiler: "TorchCompiler", p: TensorParameter) -> TorchTensorParameter:
    initializer_ = compiler.compile_initializer(p.initializer)
    dtype = _retrieve_dtype(p.dtype)
    compiled_p = TorchTensorParameter(
        *p.shape, requires_grad=p.learnable, initializer_=initializer_, dtype=dtype
    )
    compiler.state.register_compiled_parameter(p, compiled_p)
    return compiled_p


def compile_constant_parameter(
    compiler: "TorchCompiler", p: ConstantParameter
) -> TorchTensorParameter:
    initializer_ = compiler.compile_initializer(p.initializer)
    compiled_p = TorchTensorParameter(*p.shape, requires_grad=False, initializer_=initializer_)
    compiler.state.register_compiled_parameter(p, compiled_p)
    return compiled_p


def compile_reference_parameter(
    compiler: "TorchCompiler", p: ReferenceParameter
) -> TorchPointerParameter:
    # Obtain the other parameter's graph (and its fold index),
    # and wrap it in a pointer parameter node.
    compiled_p, fold_idx = compiler.state.retrieve_compiled_parameter(p.deref())
    return TorchPointerParameter(compiled_p, fold_idx=fold_idx)


def compile_sum_parameter(compiler: "TorchCompiler", p: SumParameter) -> TorchSumParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchSumParameter(in_shape1, in_shape2)


def compile_hadamard_parameter(
    compiler: "TorchCompiler", p: HadamardParameter
) -> TorchHadamardParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchHadamardParameter(in_shape1, in_shape2)


def compile_kronecker_parameter(
    compiler: "TorchCompiler", p: KroneckerParameter
) -> TorchKroneckerParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchKroneckerParameter(in_shape1, in_shape2)


def compile_outer_product_parameter(
    compiler: "TorchCompiler",
    p: OuterProductParameter,
) -> TorchOuterProductParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchOuterProductParameter(in_shape1, in_shape2, dim=p.axis)


def compile_outer_sum_parameter(
    compiler: "TorchCompiler", p: OuterSumParameter
) -> TorchOuterSumParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchOuterSumParameter(in_shape1, in_shape2, dim=p.axis)


def compile_exp_parameter(compiler: "TorchCompiler", p: ExpParameter) -> TorchExpParameter:
    (in_shape,) = p.in_shapes
    return TorchExpParameter(in_shape)


def compile_log_parameter(compiler: "TorchCompiler", p: LogParameter) -> TorchLogParameter:
    (in_shape,) = p.in_shapes
    return TorchLogParameter(in_shape)


def compile_square_parameter(compiler: "TorchCompiler", p: SquareParameter) -> TorchSquareParameter:
    (in_shape,) = p.in_shapes
    return TorchSquareParameter(in_shape)


def compile_sigmoid_parameter(
    compiler: "TorchCompiler", p: SigmoidParameter
) -> TorchSigmoidParameter:
    (in_shape,) = p.in_shapes
    return TorchSigmoidParameter(in_shape)


def compile_scaled_sigmoid_parameter(
    compiler: "TorchCompiler", p: ScaledSigmoidParameter
) -> TorchScaledSigmoidParameter:
    (in_shape,) = p.in_shapes
    return TorchScaledSigmoidParameter(in_shape, vmin=p.vmin, vmax=p.vmax)


def compile_clamp_parameter(compiler: "TorchCompiler", p: ClampParameter) -> TorchClampParameter:
    (in_shape,) = p.in_shapes
    return TorchClampParameter(in_shape, vmin=p.vmin, vmax=p.vmax)


def compile_conjugate_parameter(
    compiler: "TorchCompiler", p: ClampParameter
) -> TorchConjugateParameter:
    (in_shape,) = p.in_shapes
    return TorchConjugateParameter(in_shape)


def compile_reduce_sum_parameter(
    compiler: "TorchCompiler", p: ReduceSumParameter
) -> TorchReduceSumParameter:
    (in_shape,) = p.in_shapes
    return TorchReduceSumParameter(in_shape, dim=p.axis)


def compile_reduce_product_parameter(
    compiler: "TorchCompiler", p: ReduceProductParameter
) -> TorchReduceProductParameter:
    (in_shape,) = p.in_shapes
    return TorchReduceProductParameter(in_shape, dim=p.axis)


def compile_reduce_lse_parameter(
    compiler: "TorchCompiler", p: ReduceSumParameter
) -> TorchReduceLSEParameter:
    (in_shape,) = p.in_shapes
    return TorchReduceLSEParameter(in_shape, dim=p.axis)


def compile_softmax_parameter(
    compiler: "TorchCompiler", p: SoftmaxParameter
) -> TorchSoftmaxParameter:
    (in_shape,) = p.in_shapes
    return TorchSoftmaxParameter(in_shape, dim=p.axis)


def compile_log_softmax_parameter(
    compiler: "TorchCompiler", p: SoftmaxParameter
) -> TorchLogSoftmaxParameter:
    (in_shape,) = p.in_shapes
    return TorchLogSoftmaxParameter(in_shape, dim=p.axis)


def compile_gaussian_product_mean(
    compiler: "TorchCompiler", p: GaussianProductMean
) -> TorchGaussianProductMean:
    return TorchGaussianProductMean(*p.in_shapes)


def compile_gaussian_product_stddev(
    compiler: "TorchCompiler", p: GaussianProductStddev
) -> TorchGaussianProductStddev:
    return TorchGaussianProductStddev(*p.in_shapes)


def compile_gaussian_product_log_partition(
    compiler: "TorchCompiler", p: GaussianProductLogPartition
) -> TorchGaussianProductLogPartition:
    return TorchGaussianProductLogPartition(*p.in_shapes)


DEFAULT_PARAMETER_COMPILATION_RULES: Dict[ParameterCompilationSign, ParameterCompilationFunc] = {  # type: ignore[misc]
    TensorParameter: compile_tensor_parameter,
    ConstantParameter: compile_constant_parameter,
    ReferenceParameter: compile_reference_parameter,
    SumParameter: compile_sum_parameter,
    HadamardParameter: compile_hadamard_parameter,
    KroneckerParameter: compile_kronecker_parameter,
    OuterProductParameter: compile_outer_product_parameter,
    OuterSumParameter: compile_outer_sum_parameter,
    ExpParameter: compile_exp_parameter,
    LogParameter: compile_log_parameter,
    SquareParameter: compile_square_parameter,
    SigmoidParameter: compile_sigmoid_parameter,
    ScaledSigmoidParameter: compile_scaled_sigmoid_parameter,
    ClampParameter: compile_clamp_parameter,
    ConjugateParameter: compile_conjugate_parameter,
    ReduceSumParameter: compile_reduce_sum_parameter,
    ReduceProductParameter: compile_reduce_product_parameter,
    ReduceLSEParameter: compile_reduce_lse_parameter,
    SoftmaxParameter: compile_softmax_parameter,
    LogSoftmaxParameter: compile_log_softmax_parameter,
    GaussianProductMean: compile_gaussian_product_mean,
    GaussianProductStddev: compile_gaussian_product_stddev,
    GaussianProductLogPartition: compile_gaussian_product_log_partition,
}
