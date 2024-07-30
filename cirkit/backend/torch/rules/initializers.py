import functools
from typing import TYPE_CHECKING, Dict

import torch
from torch import nn

from cirkit.backend.compiler import InitializerCompilationFunc, InitializerCompilationSign
from cirkit.backend.torch.initializers import InitializerFunc, copy_from_ndarray_, dirichlet_
from cirkit.symbolic.initializers import (
    ConstantInitializer,
    ConstantTensorInitializer,
    DirichletInitializer,
    NormalInitializer,
    UniformInitializer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_constant_initializer(
    compiler: "TorchCompiler", init: ConstantInitializer
) -> InitializerFunc:
    return functools.partial(torch.fill_, value=init.value)


def compile_constant_tensor_initializer(
    compiler: "TorchCompiler", init: ConstantTensorInitializer
) -> InitializerFunc:
    return functools.partial(copy_from_ndarray_, array=init.value)


def compile_uniform_initializer(
    compiler: "TorchCompiler", init: UniformInitializer
) -> InitializerFunc:
    return functools.partial(nn.init.uniform_, a=init.a, b=init.b)


def compile_normal_initializer(
    compiler: "TorchCompiler", init: NormalInitializer
) -> InitializerFunc:
    return functools.partial(nn.init.normal_, mean=init.mean, std=init.stddev)


def compiler_dirichlet_initializer(
    compiler: "TorchCompiler", init: DirichletInitializer
) -> InitializerFunc:
    axis = init.axis if init.axis < 0 else init.axis + 1
    return functools.partial(dirichlet_, alpha=init.alpha, dim=axis)


DEFAULT_INITIALIZER_COMPILATION_RULES: Dict[InitializerCompilationSign, InitializerCompilationFunc] = {  # type: ignore[misc]
    ConstantInitializer: compile_constant_initializer,
    ConstantTensorInitializer: compile_constant_tensor_initializer,
    UniformInitializer: compile_uniform_initializer,
    NormalInitializer: compile_normal_initializer,
    DirichletInitializer: compiler_dirichlet_initializer,
}
