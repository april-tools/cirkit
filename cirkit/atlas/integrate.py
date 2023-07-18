from functools import cached_property
from typing import List, Union

import torch


class IntegrationContext:  # pylint: disable=too-few-public-methods
    """The integration context."""

    def __init__(self, num_vars: int, ivars: Union[List[int], List[List[int]]]):
        """Initialize an integration context.

        Args:
            num_vars: The total number of variables.
            ivars: A list of variables to integrate, or a batch of lists of variables to integrate.
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
        self.num_vars = num_vars
        self.batch_ivars: List[List[int]] = batch_ivars

    @cached_property
    def as_mask(self) -> torch.Tensor:
        """Return a mask representation of the integration context.

        Return a tensor of shape (batch_size, num_vars) whose entries are ones if the
        corresponding variables is marginalized, and zero otherwise.
        """
        mask = torch.zeros(
            len(self.batch_ivars), self.num_vars, dtype=torch.long, requires_grad=False
        )
        for i, vs in enumerate(self.batch_ivars):
            mask[i, vs] = 1
        return mask
