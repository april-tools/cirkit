# pylint: disable=unused-argument

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from cirkit.backend.compiler import InitializerCompilationSign
from cirkit.backend.torch.initializers import (
    InitializerFunc,
    copy_from_ndarray_,
    dirichlet_,
)
from cirkit.symbolic.initializers import (
    ConstantTensorInitializer,
    DirichletInitializer,
    NormalInitializer,
    UniformInitializer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def normalize_initializer(init: Callable[[torch.Tensor], torch.Tensor]):
    """Modify an initializer to normalize the parameter to a convex sum.

    Args:
        init: initializer function (can be partial).

    Returns:
        Normalized initializer function
    """

    def norm_init(tensor: torch.Tensor):
        init(tensor)
        tensor.copy_(tensor.softmax(dim=-1))
        return tensor

    return norm_init


def compile_constant_tensor_initializer(
    compiler: "TorchCompiler", init: ConstantTensorInitializer
) -> InitializerFunc:
    if isinstance(init.value, np.ndarray):
        return functools.partial(copy_from_ndarray_, array=init.value)
    return functools.partial(torch.fill_, value=init.value)


def compile_uniform_initializer(
    compiler: "TorchCompiler", init: UniformInitializer
) -> InitializerFunc:
    if init.convex:
        return normalize_initializer(
            functools.partial(nn.init.uniform_, a=init.a, b=init.b)
        )
    else:
        return functools.partial(nn.init.uniform_, a=init.a, b=init.b)


def compile_normal_initializer(
    compiler: "TorchCompiler", init: NormalInitializer
) -> InitializerFunc:
    if init.convex:
        return normalize_initializer(
            functools.partial(nn.init.normal_, mean=init.mean, std=init.stddev)
        )
    else:
        return functools.partial(nn.init.normal_, mean=init.mean, std=init.stddev)


def compile_dirichlet_initializer(
    compiler: "TorchCompiler", init: DirichletInitializer
) -> InitializerFunc:
    axis = init.axis if init.axis < 0 else init.axis + 1
    return functools.partial(dirichlet_, alpha=init.alpha, dim=axis)


DEFAULT_INITIALIZER_COMPILATION_RULES: dict[
    InitializerCompilationSign, Callable[..., InitializerFunc]
] = {
    ConstantTensorInitializer: compile_constant_tensor_initializer,
    UniformInitializer: compile_uniform_initializer,
    NormalInitializer: compile_normal_initializer,
    DirichletInitializer: compile_dirichlet_initializer,
}
