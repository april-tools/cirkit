from numbers import Number
from typing import Callable, Tuple, Dict, Any

import torch
from torch import Tensor, nn

from cirkit.backend.torch.params.base import AbstractTorchParameter


class TorchParameter(AbstractTorchParameter):
    """The leaf in reparameterizations that holds the parameter Tensor."""

    def __init__(self, *shape: int, num_folds: int = 1, requires_grad: bool = True) -> None:
        """Init class."""
        super().__init__(num_folds=num_folds)
        pshape = (num_folds, *shape)
        self._ptensor = nn.Parameter(torch.empty(*pshape), requires_grad=requires_grad)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._ptensor.shape[1:])

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return dict(shape=self.shape, num_folds=self.num_folds, requires_grad=self.requires_grad)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""
        return self._ptensor.dtype

    @property
    def device(self) -> torch.device:
        """The device of the output parameter."""
        return self._ptensor.device

    @property
    def requires_grad(self) -> bool:
        return self._ptensor.requires_grad

    @torch.no_grad()
    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter buffer with the given initializer.

        Args:
            initializer_ (Callable[[Tensor], Tensor]): The function that initialize a Tensor inplace.
        """
        initializer_(self._ptensor)

    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        return self._ptensor


class TorchConstantParameter(TorchParameter):
    def __init__(self, shape: Tuple[int, ...], value: Number, *, num_folds: int = 1) -> None:
        """Init class."""
        super().__init__(*shape, num_folds=num_folds, requires_grad=False)
        self.value = value
        self.initialize(self._initializer)

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        config = super().config
        config.update(value=self.value)
        return config

    def _initializer(self, t: Tensor) -> Tensor:
        return t.copy_(torch.full(size=t.shape, fill_value=self.value))
