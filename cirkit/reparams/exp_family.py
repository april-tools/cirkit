from typing import Any, Sequence

import torch
from torch import Tensor

from .leaf import ReparamLeaf

# TODO: these are too specific, reconsider how to implement


# This is just Indentity, optionally we can add a scaling factor but currently not implemented.
##
# class ReparamEFBinomial(ReparamLeaf):
#     """Reparameterization for ExpFamily -- Binomial."""


class ReparamEFCategorical(ReparamLeaf):
    """Reparameterization for ExpFamily -- Categorical."""

    def __init__(  # type: ignore[misc]
        self,
        size: Sequence[int],
        /,
        *,
        num_categories: int = 0,
        **kwargs: Any,  # hold dim/mask/log_mask, but irrelevant here.
    ) -> None:
        """Init class.

        Args:
            size (Sequence[int]): The size of the parameter.
            num_categories (int, optional): The number of categories, must override default with a \
                non-zero value. Defaults to 0.
        """
        assert num_categories > 0
        super().__init__(size, **kwargs)  # type: ignore[misc]
        self.num_categories = num_categories

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        # TODO: x.unflatten is not typed
        param = torch.unflatten(
            self.param, dim=-1, sizes=(-1, self.num_categories)
        )  # shape (..., C, cat)
        param = torch.log_softmax(param, dim=-1)
        return param.flatten(start_dim=-2)  # shape (..., C*cat)


class ReparamEFNormal(ReparamLeaf):
    """Reparameterization for ExpFamily -- Normal."""

    def __init__(  # type: ignore[misc]
        self,
        size: Sequence[int],
        /,
        *,
        min_var: float = 0.0001,
        max_var: float = 10.0,
        **kwargs: Any,  # hold dim/mask/log_mask, but irrelevant here.
    ) -> None:
        """Init class.

        Args:
            size (Sequence[int]): The size of the parameter.
            min_var (float, optional): The min variance. Defaults to 0.0001.
            max_var (float, optional): The max variance. Defaults to 10.0.
        """
        assert not size[-1] % 2
        assert 0 <= min_var < max_var
        super().__init__(size, **kwargs)  # type: ignore[misc]
        self.num_channels = size[-1] // 2
        self.min_var = min_var
        self.max_var = max_var

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        mu = self.param[..., : self.num_channels]  # shape (..., C)
        var = (
            torch.sigmoid(self.param[..., self.num_channels :]) * (self.max_var - self.min_var)
            + self.min_var
        )  # shape (..., C)
        param = torch.stack(
            (mu, torch.tensor(-0.5).to(mu).expand_as(mu)), dim=-2
        )  # shape (..., 2, C)
        return (param / var.unsqueeze(dim=-2)).flatten(start_dim=-2)  # shape (..., 2*C)
