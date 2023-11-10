from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional

from torch import Tensor, nn

from cirkit.reparams.leaf import ReparamIdentity
from cirkit.utils.type_aliases import ReparamFactory


class Layer(nn.Module, ABC):
    """The abstract base class for all layers.

    Here saves the basic properties for layers and provide interface for the functionalities that \
    each layer should implement.
    """

    fold_mask: Optional[Tensor]

    def __init__(  # type: ignore[misc]  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int,
        num_folds: int,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,  # pylint: disable=unused-argument
        **_: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int): The arity of the layer.
            num_folds (int): The number of folds.
            fold_mask (Optional[Tensor], optional): The mask of valid folds. Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        super().__init__()
        assert num_input_units > 0
        assert num_output_units > 0
        assert arity > 0
        assert num_folds > 0
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity
        self.num_folds = num_folds
        self.register_buffer("fold_mask", fold_mask)
        # reparam is not used here, but is commonly used and therefore added to Layer.__init__

    # expected to be fixed, so use cached to avoid recalc
    @cached_property
    def num_params(self) -> int:
        """The number of params."""
        return sum(param.numel() for param in self.parameters())

    @cached_property
    def num_buffers(self) -> int:
        """The number of buffers."""
        return sum(buffer.numel() for buffer in self.buffers())

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset parameters with default initialization."""

    # TODO: temp solution to accomodate IntegralInputLayer
    def __call__(self, x: Tensor, *_: Any) -> Tensor:  # type: ignore[misc]
        """Invoke the forward function.

        Args:
            x (Tensor): The input to this layer, shape (F, H, K, *B).

        Returns:
            Tensor: The output of this layer, shape (F, K, *B).
        """
        return super().__call__(x, *_)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (F, H, K, *B).

        Returns:
            Tensor: The output of this layer, shape (F, K, *B).
        """
