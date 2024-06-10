import functools
from typing import Optional, Dict, TYPE_CHECKING

import torch

from cirkit.backend.base import ParameterCompilationSign, ParameterCompilationFunc
from cirkit.backend.torch.parameters.leaves import TorchTensorParameter, TorchPointerParameter
from cirkit.backend.torch.parameters.ops import (
    TorchHadamardParameter,
    TorchKroneckerParameter,
    TorchOuterProductParameter,
    TorchOuterSumParameter,
    TorchLogParameter,
    TorchSigmoidParameter,
    TorchScaledSigmoidParameter,
    TorchReduceSumParameter,
    TorchReduceProductParameter,
    TorchReduceLSEParameter,
    TorchExpParameter,
    TorchSquareParameter,
    TorchSoftmaxParameter,
    TorchLogSoftmaxParameter
)
from cirkit.backend.torch.utils import InitializerFunc
from cirkit.symbolic.parameters import (
    TensorParameter,
    ConstantParameter,
    ReferenceParameter,
    HadamardParameter,
    KroneckerParameter,
    OuterProductParameter,
    OuterSumParameter,
    ExpParameter,
    LogParameter,
    SquareParameter,
    ScaledSigmoidParameter,
    ReduceSumParameter,
    ReduceProductParameter,
    SoftmaxParameter,
    SigmoidParameter,
    ReduceLSEParameter,
    LogSoftmaxParameter
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_parameter(
    compiler: "TorchCompiler",
    p: TensorParameter
) -> TorchTensorParameter:
    compiled_p = TorchTensorParameter(*p.shape, requires_grad=p.learnable)
    compiler.state.register_compiled_parameter(p, compiled_p)
    return compiled_p


def compile_constant_parameter(compiler: "TorchCompiler", p: ConstantParameter) -> TorchTensorParameter:
    compiled_p = TorchTensorParameter(
        *p.shape, init_func=functools.partial(torch.fill_, value=p.value), requires_grad=False)
    compiler.state.register_compiled_parameter(p, compiled_p)
    return compiled_p


def compile_reference_parameter(compiler: "TorchCompiler", p: ReferenceParameter) -> TorchPointerParameter:
    # Obtain the other parameter's graph (and its fold index),
    # and wrap it in a pointer parameter node.
    compiled_p, fold_idx = compiler.state.retrieve_compiled_parameter(p.deref())
    fold_idx = None if fold_idx == 0 and compiled_p.num_folds == 1 else fold_idx
    return TorchPointerParameter(compiled_p, fold_idx=fold_idx)


def compile_hadamard_parameter(compiler: "TorchCompiler", p: HadamardParameter) -> TorchHadamardParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchHadamardParameter(in_shape1, in_shape2)


def compile_kronecker_parameter(compiler: "TorchCompiler", p: KroneckerParameter) -> TorchKroneckerParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchKroneckerParameter(in_shape1, in_shape2)


def compile_outer_product_parameter(
    compiler: "TorchCompiler",
    p: OuterProductParameter,
) -> TorchOuterProductParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchOuterProductParameter(in_shape1, in_shape2, dim=p.axis)


def compile_outer_sum_parameter(
    compiler: "TorchCompiler",
    p: OuterSumParameter
) -> TorchOuterSumParameter:
    in_shape1, in_shape2 = p.in_shapes
    return TorchOuterSumParameter(in_shape1, in_shape2, dim=p.axis)


def compile_exp_parameter(
    compiler: "TorchCompiler", p: ExpParameter, init_func: Optional[InitializerFunc] = None
) -> TorchExpParameter:
    in_shape, = p.in_shapes
    return TorchExpParameter(in_shape)


def compile_log_parameter(
    compiler: "TorchCompiler", p: LogParameter, init_func: Optional[InitializerFunc] = None
) -> TorchLogParameter:
    in_shape, = p.in_shapes
    return TorchLogParameter(in_shape)


def compile_square_parameter(
    compiler: "TorchCompiler", p: SquareParameter, init_func: Optional[InitializerFunc] = None
) -> TorchSquareParameter:
    in_shape, = p.in_shapes
    return TorchSquareParameter(in_shape)


def compile_sigmoid_parameter(
    compiler: "TorchCompiler",
    p: ScaledSigmoidParameter
) -> TorchSigmoidParameter:
    in_shape, = p.in_shapes
    return TorchSigmoidParameter(in_shape)


def compile_scaled_sigmoid_parameter(
    compiler: "TorchCompiler",
    p: ScaledSigmoidParameter
) -> TorchScaledSigmoidParameter:
    in_shape, = p.in_shapes
    return TorchScaledSigmoidParameter(in_shape, vmin=p.vmin, vmax=p.vmax)


def compile_reduce_sum_parameter(
    compiler: "TorchCompiler", p: ReduceSumParameter
) -> TorchReduceSumParameter:
    in_shape, = p.in_shapes
    return TorchReduceSumParameter(in_shape, dim=p.axis)


def compile_reduce_product_parameter(
    compiler: "TorchCompiler", p: ReduceProductParameter
) -> TorchReduceProductParameter:
    in_shape, = p.in_shapes
    return TorchReduceProductParameter(in_shape, dim=p.axis)


def compile_reduce_lse_parameter(
    compiler: "TorchCompiler", p: ReduceSumParameter
) -> TorchReduceLSEParameter:
    in_shape, = p.in_shapes
    return TorchReduceLSEParameter(in_shape, dim=p.axis)


def compile_softmax_parameter(
    compiler: "TorchCompiler", p: SoftmaxParameter
) -> TorchSoftmaxParameter:
    in_shape, = p.in_shapes
    return TorchSoftmaxParameter(in_shape, dim=p.axis)


def compile_log_softmax_parameter(
    compiler: "TorchCompiler", p: SoftmaxParameter
) -> TorchLogSoftmaxParameter:
    in_shape, = p.in_shapes
    return TorchLogSoftmaxParameter(in_shape, dim=p.axis)


DEFAULT_PARAMETER_COMPILATION_RULES: Dict[ParameterCompilationSign, ParameterCompilationFunc] = {  # type: ignore[misc]
    TensorParameter: compile_parameter,
    ConstantParameter: compile_constant_parameter,
    ReferenceParameter: compile_reference_parameter,
    HadamardParameter: compile_hadamard_parameter,
    KroneckerParameter: compile_kronecker_parameter,
    OuterProductParameter: compile_outer_product_parameter,
    OuterSumParameter: compile_outer_sum_parameter,
    ExpParameter: compile_exp_parameter,
    LogParameter: compile_log_parameter,
    SquareParameter: compile_square_parameter,
    SigmoidParameter: compile_sigmoid_parameter,
    ScaledSigmoidParameter: compile_scaled_sigmoid_parameter,
    ReduceSumParameter: compile_reduce_sum_parameter,
    ReduceProductParameter: compile_reduce_product_parameter,
    ReduceLSEParameter: compile_reduce_lse_parameter,
    SoftmaxParameter: compile_softmax_parameter,
    LogSoftmaxParameter: compile_log_softmax_parameter
}
