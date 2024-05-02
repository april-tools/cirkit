from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn


class AbstractTorchParameter(nn.Module, ABC):
    """The abstract base class for all reparameterizations.

    NOTE: An instance of this class can be materialized only once, and following materializations \
          are all no-op. If we do want to a true re-materialize, another instance should be \
          constructed.
    """

    def __init__(self) -> None:
        """Init class."""
        super().__init__()

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        ...

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
