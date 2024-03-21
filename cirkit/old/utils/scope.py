from typing import List, Union

import torch
from torch import Tensor


def one_hot_variables(
    num_vars: int, ivars: Union[List[int], List[List[int]]], device: torch.device
) -> Tensor:
    """Return a one-hot encoding mask of a batch of list of variables.

    Args:
        num_vars: The maximum number of variables.
        ivars: A list of variables or a batch of list of variables.
        device: Device where to create the mask

    Returns:
        Tensor: A floating-point mask M of shape (batch_size, num_vars)
         such that M[i,j] = 1 iff j is in ivars[i]. If ivars is a list then batch_size is 1.
    """
    assert len(ivars) > 0
    if isinstance(ivars[0], int):
        batch_ivars: List[List[int]] = [ivars]  # type: ignore[list-item]
    else:
        batch_ivars: List[List[int]] = ivars  # type: ignore[no-redef]
    all_vars = list(range(num_vars))
    for i, vs in enumerate(batch_ivars):
        assert all(
            v in all_vars for v in vs
        ), f"The id of variables in the {i}-th batch should be in [0, {num_vars})"
    mask = torch.zeros(len(batch_ivars), num_vars, requires_grad=False, device=device)
    for i, vs in enumerate(batch_ivars):
        mask[i, vs] = 1
    return mask
