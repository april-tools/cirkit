from numbers import Number
from typing import Any, Callable, Dict, Optional, Tuple, final

import torch
from torch import Tensor, nn

from cirkit.backend.torch.params.base import AbstractTorchParameter


@final
class TorchParameter(AbstractTorchParameter):
    """The leaf in reparameterizations that holds the parameter Tensor."""

    def __init__(
        self,
        *shape: int,
        init_func: Callable[[Tensor], Tensor],
        num_folds: int = 1,
        requires_grad: bool = True,
    ) -> None:
        """Init class."""
        super().__init__(num_folds=num_folds)
        self._shape = shape
        self._ptensor = None
        self.init_func = init_func
        self.requires_grad = requires_grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return dict(shape=self._shape, num_folds=self.num_folds, learnable=self.requires_grad)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        assert self._ptensor is not None
        return self._ptensor.dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        assert self._ptensor is not None
        return self._ptensor.device

    @property
    def is_initialized(self) -> bool:
        return self._ptensor is not None

    @torch.no_grad()
    def reset(self) -> None:
        """Initialize the internal parameter tensor with the given initializer."""
        if self._ptensor is None:
            ptensor = torch.empty(self.num_folds, *self._shape)
            self._ptensor = nn.Parameter(ptensor, requires_grad=self.requires_grad)
        self.init_func(self._ptensor.data)

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        assert self._ptensor is not None
        return self._ptensor
