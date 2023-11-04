from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor, nn

from cirkit.utils.type_aliases import ClampBounds

# TODO: rework docstrings


# TODO: what interface do we need in this very generic class?
class Layer(nn.Module, ABC):
    """Abstract layer class. Specifies functionality every layer in an EiNet should implement."""

    _fold_mask: Optional[Tensor]

    def __init__(
        self,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
    ) -> None:
        """Initialize a layer.

        Args:
            num_folds: The number of folds that the layer computes.
            fold_mask (Optional[Tensor]): The mask to apply to the folded parameter tensors.
        """
        super().__init__()
        assert num_folds > 0
        self.num_folds = num_folds
        if fold_mask is not None:
            if fold_mask.dtype == torch.bool:
                fold_mask = fold_mask.to(torch.get_default_dtype())
        self.register_buffer("_fold_mask", fold_mask)
        self.param_clamp_value: ClampBounds = {}

    @property
    def num_params(self) -> int:
        """Get the number of params.

        Returns:
            int: The number of params
        """
        return sum(param.numel() for param in self.parameters())

    @property
    def fold_mask(self) -> Optional[Tensor]:
        """Get the fold mask.

        Returns:
            Tensor: The fold mask.
        """
        return self._fold_mask

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset parameters to default initialization."""

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
