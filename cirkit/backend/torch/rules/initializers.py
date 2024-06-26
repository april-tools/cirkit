import functools
from typing import TYPE_CHECKING, Callable, Dict, List, Union

import torch
from torch import Tensor, nn

from cirkit.backend.base import InitializerCompilationFunc, InitializerCompilationSign
from cirkit.symbolic.initializers import (
    ConstantInitializer,
    DirichletInitializer,
    NormalInitializer,
    UniformInitializer,
)

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_constant_initializer(
    compiler: "TorchCompiler", init: ConstantInitializer
) -> Callable[[Tensor], Tensor]:
    return functools.partial(torch.fill_, value=init.value)


def compile_uniform_initializer(
    compiler: "TorchCompiler", init: UniformInitializer
) -> Callable[[Tensor], Tensor]:
    return functools.partial(nn.init.uniform_, a=init.a, b=init.b)


def compile_normal_initializer(
    compiler: "TorchCompiler", init: NormalInitializer
) -> Callable[[Tensor], Tensor]:
    return functools.partial(nn.init.normal_, mean=init.mean, std=init.stddev)


def compiler_dirichlet_initializer(
    compiler: "TorchCompiler", init: DirichletInitializer
) -> Callable[[Tensor], Tensor]:
    axis = init.axis if init.axis < 0 else init.axis + 1
    return functools.partial(dirichlet_, alpha=init.alpha, dim=axis)


def dirichlet_(tensor: torch.Tensor, alpha: Union[float, List[float]], *, dim: int = -1) -> Tensor:
    shape = tensor.shape
    if len(shape) == 0:
        raise ValueError(
            "Cannot initialize a tensor with no dimensions by sampling from a Dirichlet"
        )
    dim = dim if dim >= 0 else dim + len(shape)
    if isinstance(alpha, float):
        concentration = torch.full([shape[dim]], fill_value=alpha)
    else:
        if shape[dim] != len(alpha):
            raise ValueError(
                "The selected dim of the tensor and the size of concentration parameters do not match"
            )
        concentration = torch.tensor(alpha)
    dirichlet = torch.distributions.Dirichlet(concentration)
    samples = dirichlet.sample(torch.Size([d for i, d in enumerate(shape) if i != dim]))
    tensor.copy_(torch.transpose(samples, dim, -1))
    return tensor


DEFAULT_INITIALIZER_COMPILATION_RULES: Dict[InitializerCompilationSign, InitializerCompilationFunc] = {  # type: ignore[misc]
    ConstantInitializer: compile_constant_initializer,
    UniformInitializer: compile_uniform_initializer,
    NormalInitializer: compile_normal_initializer,
    DirichletInitializer: compiler_dirichlet_initializer,
}
