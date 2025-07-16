import numpy as np
import torch

from cirkit.backend.torch.compiler import TorchCompiler
from cirkit.symbolic.initializers import ConstantTensorInitializer


def test_compile_initializer_constant_tensor() -> None:
    compiler = TorchCompiler()
    array = np.arange(10)
    symbolic_initializer = ConstantTensorInitializer(array)
    initializer_ = compiler.compile_initializer(symbolic_initializer)
    x = torch.randint(10000, size=(10,))
    initializer_(x)
    assert torch.all(x == torch.arange(10))
