from abc import ABC, abstractmethod
from typing import Any, TypedDict

import torch
from torch import Tensor, nn

# TODO: rework docstrings


class _ClampValue(TypedDict, total=False):
    """Wraps the kwargs passed to `torch.clamp()`."""

    min: float
    max: float


# TODO: name it layer?
# TODO: what interface do we need in this very generic class?
class Layer(nn.Module, ABC):
    """Abstract layer class. Specifies functionality every layer in an EiNet should implement."""

    def __init__(self) -> None:
        """Init class."""
        super().__init__()  # TODO: do we need multi-inherit init?
        self.param_clamp_value: _ClampValue = {}
        self.fold_count = 0

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset parameters to default initialization."""

    @property
    def num_params(self) -> int:
        """Get the number of params.

        Returns:
            int: the number of params
        """
        return sum(param.numel() for param in self.parameters())

    @torch.no_grad()
    def clamp_params(self, clamp_all: bool = False) -> None:
        """Clamp parameters such that they are non-negative and is impossible to \
            get zero probabilities.

        This involves using a constant that is specific on the computation.

        Args:
            clamp_all (bool, optional): Whether to clamp all. Defaults to False.
        """
        for param in self.parameters():
            if clamp_all or param.requires_grad:
                param.clamp_(**self.param_clamp_value)

    def __call__(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Invoke forward.

        Returns:
            Tensor: Return of forward.
        """
        return super().__call__(*args, **kwargs)  # type: ignore[no-any-return,misc]

    @abstractmethod
    # pylint: disable-next=missing-param-doc
    def forward(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Implement forward.

        Returns:
            Tensor: Return of forward.
        """

    # TODO: need to implement relevant things
    # TODO: should be abstract but for now NO to prevent blocking downstream
    def backtrack(self, *args: Any, **kwargs: Any) -> Tensor:  # type: ignore[misc]
        """Define routines for backtracking in EiNets, for sampling and MPE approximation.

        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError
