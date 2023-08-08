from abc import ABC, abstractmethod
from typing import Any, List, Optional, TypedDict, Union

import torch
from torch import Tensor, nn

from cirkit.region_graph import PartitionNode, RegionNode

# TODO: rework docstrings


class _ClampValue(TypedDict, total=False):
    """Wraps the kwargs passed to `torch.clamp()`."""

    min: float
    max: float


# TODO: what interface do we need in this very generic class?
class Layer(nn.Module, ABC):
    """Abstract layer class. Specifies functionality every layer in an EiNet should implement."""

    def __init__(
        self,
        rg_nodes: Union[List[RegionNode], List[PartitionNode]],
        fold_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize a layer.

        Args:
            rg_nodes (List[RegionNode]): The region nodes on which the layer is defined on.
            fold_mask (Optional[torch.Tensor]): The mask to apply to the folded parameter tensors.
        """
        super().__init__()  # TODO: do we need multi-inherit init?
        self.rg_nodes = rg_nodes
        if fold_mask is not None:
            if fold_mask.dtype == torch.bool:
                fold_mask = fold_mask.to(torch.get_default_dtype())
            fold_mask = fold_mask.unsqueeze(dim=-1)
        self.register_buffer("fold_mask", fold_mask)
        self.param_clamp_value: _ClampValue = {}

    def reset_parameters(self) -> None:
        """Reset parameters to default initialization."""

    @property
    def fold_size(self) -> int:
        """Get the number of folds computed by the layer.

        Returns:
            int: The number of folds.
        """
        return len(self.rg_nodes)

    @property
    def num_params(self) -> int:
        """Get the number of params.

        Returns:
            int: The number of params
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
