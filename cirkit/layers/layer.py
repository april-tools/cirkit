from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor, nn

# TODO: rework docstrings


# TODO: what interface do we need in this very generic class?
class Layer(nn.Module, ABC):
    """Abstract layer class. Specifies functionality every layer in an EiNet should implement."""

    _fold_mask: Optional[Tensor]

    # TODO: design kwarg-only. We mostly call by Layer(**kwargs)
    # TODO: design what info is saved in which class (e.g. K_in and K_out can be here?)
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

    @property
    def num_params(self) -> int:
        """Get the number of params.

        Returns:
            int: The number of params.
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

    # TODO: temp solution to accomodate IntegralInputLayer
    def __call__(self, x: Tensor, *_: Any) -> Tensor:  # type: ignore[misc]
        """Invoke forward function.

        Args:
            x (Tensor): The input to this layer.

        Returns:
            Tensor: The output of this layer.
        """
        return super().__call__(x, *_)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer.

        Returns:
            Tensor: The output of this layer.
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
