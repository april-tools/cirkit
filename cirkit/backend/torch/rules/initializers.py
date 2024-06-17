import functools
from typing import TYPE_CHECKING, Callable, Dict

import torch
from torch import Tensor

from cirkit.backend.base import InitializerCompilationFunc, InitializerCompilationSign
from cirkit.symbolic.initializers import ConstantInitializer

if TYPE_CHECKING:
    from cirkit.backend.torch.compiler import TorchCompiler


def compile_constant_initializer(
    compiler: "TorchCompiler", init: ConstantInitializer
) -> Callable[[Tensor], Tensor]:
    return functools.partial(torch.fill_, value=init.value)


DEFAULT_INITIALIZER_COMPILATION_RULES: Dict[
    InitializerCompilationSign, InitializerCompilationFunc
] = {}  # type: ignore[misc]
