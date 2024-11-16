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
        concentration = Tensor(alpha)
    dirichlet = torch.distributions.Dirichlet(concentration)
    samples = dirichlet.sample(torch.Size([d for i, d in enumerate(shape) if i != dim]))
    tensor.copy_(torch.transpose(samples, dim, -1))
    return tensor


def mixing_weights_(
    tensor: Tensor, weights_init_: InitializerFunc, fill_value: float = 0.0
) -> Tensor:
    if len(tensor.shape) < 2 or tensor.shape[-1] % tensor.shape[-2] != 0:
        raise ValueError(
            f"Expected destination shape (..., num_units, arity * num_units), "
            f"but found {tensor.shape}"
        )
    num_units = tensor.shape[-2]
    arity = tensor.shape[-1] // tensor.shape[-2]
    # Initialize the mixing weights, then reshape them to be the sum layer weights
    weights = torch.empty(num_units, arity, dtype=tensor.dtype)
    weights_init_(weights)
    # (arity, num_units, num_units)
    diag_weights = torch.vmap(torch.diag)(weights.T)
    diag_weights[:, ~torch.eye(num_units, dtype=torch.bool)] = fill_value
    # (num_units, arity, num_units) -> (num_units, arity * num_units)
    return tensor.copy_(diag_weights.permute(1, 0, 2).flatten(start_dim=1))
