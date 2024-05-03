from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


class AbstractTorchParameter(nn.Module, ABC):
    """The abstract base class for all reparameterizations."""

    def __init__(self, *, num_folds: int = 1) -> None:
        """Init class."""
        super().__init__()
        self.num_folds = num_folds

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """The shape of the output parameter."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device of the output parameter."""

    @property
    def config(self) -> Dict[str, Any]:
        """Configuration flags for the parameter."""
        return {}

    @property
    def params(self) -> Dict[str, 'AbstractTorchParameter']:
        """The other parameters this parameter might depend on."""
        return {}

    def __call__(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
