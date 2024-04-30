from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, Tuple

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

    # NOTE: Subclasses should include @torch.no_grad() to disable grad for initialization.
    @abstractmethod
    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensor(s) with the given initializer.

        This can only be called after materialization and will always overwrite whatever is \
        already in the internal param. To safely provide an initial value to a possibly reused \
        reparam, initialize through materialize() instead.

        The provided initializer_ is expected to provide an initial value for the output \
        parameter, and implementations may define how the value is transformated to initialize the \
        internal tensor(s).

        Args:
            initializer_ (Callable[[Tensor], Tensor]): The function that initialize a Tensor \
                inplace while also returning the value.
        """

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
