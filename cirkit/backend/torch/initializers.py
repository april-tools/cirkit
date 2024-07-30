from typing import Callable, List, Optional, Union

import torch
from torch import Tensor

InitializerFunc = Callable[[Tensor], Tensor]


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


def stacked_initializer_(t: Tensor, *, initializers: List[Optional[InitializerFunc]]) -> Tensor:
    for i, initializer_ in enumerate(initializers):
        if initializer_ is not None:
            initializer_(t[i])
    return t
