from typing import TYPE_CHECKING

import torch

from cirkit.backend.compiler import ParameterCompilationFunc, ParameterCompilationSign
from cirkit.backend.torch.parameters.nodes import (
    TorchClampParameter,
    TorchConjugateParameter,
    TorchExpParameter,
    TorchGateFunctionParameter,
    TorchGaussianProductLogPartition,
    TorchGaussianProductMean,
    TorchGaussianProductStddev,
    TorchHadamardParameter,
    TorchIndexParameter,
    TorchKroneckerParameter,
    TorchLogParameter,
    TorchLogSoftmaxParameter,
    TorchMixingWeightParameter,
    TorchOuterProductParameter,
    TorchOuterSumParameter,
    TorchPointerParameter,
    TorchPolynomialDifferential,
    TorchPolynomialProduct,
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
from cirkit.backend.torch.utils import CachedGateFunctionEval
from cirkit.symbolic.dtypes import DataType
from cirkit.symbolic.parameters import (
    ClampParameter,
    ConjugateParameter,
    ConstantParameter,
    ExpParameter,
    GateFunctionParameter,
    GaussianProductLogPartition,
    GaussianProductMean,
    GaussianProductStddev,
    HadamardParameter,
    IndexParameter,
    KroneckerParameter,
    LogParameter,
    LogSoftmaxParameter,
    MixingWeightParameter,
    OuterProductParameter,
    OuterSumParameter,
    PolynomialDifferential,
    PolynomialProduct,
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
        return default_float_dtype.to_complex()
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


def compile_gate_function_parameter(
    compiler: "TorchCompiler", p: GateFunctionParameter
) -> TorchGateFunctionParameter:
    # Register the external model to the running state of the compiler, if needed
    if compiler.state.has_gate_function(p.name):
        gate_function_eval = compiler.state.retrieve_gate_function(p.name)
    else:
        # Retrieve the external model, based on the model id
        gate_function = compiler.get_gate_function(p.name)
        # Build the external model evaluator, and register it
        gate_function_eval = CachedGateFunctionEval(p.name, gate_function)
        compiler.state.register_gate_function(p.name, gate_function_eval)
    # Build the torch model parameter computational node
    return TorchGateFunctionParameter(
        *p.shape, gate_function_eval=gate_function_eval, name=p.name, fold_idx=p.index
    )


def compile_index_parameter(compiler: "TorchCompiler", p: IndexParameter) -> TorchIndexParameter:
    (in_shape,) = p.in_shapes
    return TorchIndexParameter(in_shape, indices=p.indices, dim=p.axis)


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


def compile_mixing_weight_parameter(
    compiler: "TorchCompiler", p: MixingWeightParameter
) -> TorchMixingWeightParameter:
    (in_shape,) = p.in_shapes
    return TorchMixingWeightParameter(in_shape)


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


def compile_polynomial_product(
    compiler: "TorchCompiler", p: PolynomialProduct
) -> TorchPolynomialProduct:
    return TorchPolynomialProduct(*p.in_shapes)


def compile_polynomial_differential(
    compiler: "TorchCompiler", p: PolynomialDifferential
) -> TorchPolynomialDifferential:
    return TorchPolynomialDifferential(*p.in_shapes, order=p.order)


DEFAULT_PARAMETER_COMPILATION_RULES: dict[ParameterCompilationSign, ParameterCompilationFunc] = {  # type: ignore[misc]
    TensorParameter: compile_tensor_parameter,
    ConstantParameter: compile_constant_parameter,
    ReferenceParameter: compile_reference_parameter,
    GateFunctionParameter: compile_gate_function_parameter,
    IndexParameter: compile_index_parameter,
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
    MixingWeightParameter: compile_mixing_weight_parameter,
    GaussianProductMean: compile_gaussian_product_mean,
    GaussianProductStddev: compile_gaussian_product_stddev,
    GaussianProductLogPartition: compile_gaussian_product_log_partition,
    PolynomialProduct: compile_polynomial_product,
    PolynomialDifferential: compile_polynomial_differential,
}
