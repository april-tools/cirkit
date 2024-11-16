import functools
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from cirkit.backend.compiler import InitializerCompilationFunc, InitializerCompilationSign
from cirkit.backend.torch.initializers import (
    InitializerFunc,
    copy_from_ndarray_,
    dirichlet_,
    mixing_weights_,
)
from cirkit.symbolic.initializers import (
    ConstantTensorInitializer,
    DirichletInitializer,
    MixingWeightInitializer,
    NormalInitializer,
    UniformInitializer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_constant_tensor_initializer(
    compiler: "TorchCompiler", init: ConstantTensorInitializer
) -> InitializerFunc:
    if isinstance(init.value, np.ndarray):
        return functools.partial(copy_from_ndarray_, array=init.value)
    return functools.partial(torch.fill_, value=init.value)


def compile_uniform_initializer(
    compiler: "TorchCompiler", init: UniformInitializer
) -> InitializerFunc:
    return functools.partial(nn.init.uniform_, a=init.a, b=init.b)


def compile_normal_initializer(
    compiler: "TorchCompiler", init: NormalInitializer
) -> InitializerFunc:
    return functools.partial(nn.init.normal_, mean=init.mean, std=init.stddev)


def compile_dirichlet_initializer(
    compiler: "TorchCompiler", init: DirichletInitializer
) -> InitializerFunc:
    axis = init.axis if init.axis < 0 else init.axis + 1
    return functools.partial(dirichlet_, alpha=init.alpha, dim=axis)


def compile_mixing_weights_initializer(
    compiler: "TorchCompiler", init: MixingWeightInitializer
) -> InitializerFunc:
    mixing_weights_fn_ = compiler.compile_initializer(init.initializer)
    return functools.partial(
        mixing_weights_, weights_init_=mixing_weights_fn_, fill_value=init.fill_value
    )


DEFAULT_INITIALIZER_COMPILATION_RULES: dict[InitializerCompilationSign, InitializerCompilationFunc] = {  # type: ignore[misc]
    ConstantTensorInitializer: compile_constant_tensor_initializer,
    UniformInitializer: compile_uniform_initializer,
    NormalInitializer: compile_normal_initializer,
    DirichletInitializer: compile_dirichlet_initializer,
    MixingWeightInitializer: compile_mixing_weights_initializer,
}
