from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

InitializerFunc = Callable[[Tensor], Tensor]


def foldwise_initializer_(t: Tensor, *, initializers: list[InitializerFunc | None]) -> Tensor:
    for i, initializer_ in enumerate(initializers):
        if initializer_ is not None:
            initializer_(t[i])
    return t


def copy_from_ndarray_(tensor: Tensor, *, array: np.ndarray) -> Tensor:
    t = torch.from_numpy(array)
    default_float_dtype = torch.get_default_dtype()
    if t.is_floating_point():
        if t.dtype != default_float_dtype:
            t = t.to(default_float_dtype)
    elif t.is_complex():
        if t.dtype != torch.complex64 and default_float_dtype == torch.float32:
            t = t.to(torch.complex64)
        elif t.dtype != torch.complex128 and default_float_dtype == torch.float64:
            t = t.to(torch.complex128)
    return tensor.copy_(t)


def dirichlet_(tensor: Tensor, alpha: float | list[float], *, dim: int = -1) -> Tensor:
    shape = tensor.shape
    if not shape:
        raise ValueError(
            "Cannot initialize a tensor with no dimensions by sampling from a Dirichlet"
        )
    dim = dim if dim >= 0 else dim + len(shape)
    if isinstance(alpha, float):
        concentration = torch.full([shape[dim]], fill_value=alpha)
    else:
        if shape[dim] != len(alpha):
            raise ValueError(
                "The selected dim of the tensor and the size of concentration parameters "
                "do not match"
            )
        concentration = Tensor(alpha)
    dirichlet = torch.distributions.Dirichlet(concentration)
    samples = dirichlet.sample(torch.Size([d for i, d in enumerate(shape) if i != dim]))
    tensor.copy_(torch.transpose(samples, dim, -1))
    return tensor
