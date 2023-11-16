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
    """The mask of valid folds, if not None, shape (F, H). None means all folds are valid."""

    def __init__(  # type: ignore[misc]  # pylint: disable=too-many-arguments
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 1,
        num_folds: int = 1,
        fold_mask: Optional[Tensor] = None,
        reparam: ReparamFactory = ReparamIdentity,  # pylint: disable=unused-argument
        **_: Any,
    ) -> None:
        """Initialize the layer.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 1.
            num_folds (int, optional): The number of folds. Defaults to 1.
            fold_mask (Optional[Tensor], optional): The mask of valid folds, shape (F, H). \
                Defaults to None.
            reparam (ReparamFactory, optional): The reparameterization. Defaults to ReparamIdentity.
        """
        super().__init__()
        assert num_input_units > 0, "The number of input units must be positive."
        assert num_output_units > 0, "The number of output units must be positive."
        assert arity > 0, "The arity must be positive."
        assert num_folds > 0, "The number of folds must be positive."
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity
        self.num_folds = num_folds
        self.register_buffer("fold_mask", fold_mask)
        # Kwarg reparam is not used here but only in layers with params. Yet it is commonly used
        # and therefore added to Layer.__init__ interface.

    # Expected to be fixed, so use cached property to avoid recalculation.
    @cached_property
    def num_params(self) -> int:
        """The number of params."""
        return sum(param.numel() for param in self.parameters())

    # Expected to be fixed, so use cached property to avoid recalculation.
    @cached_property
    def num_buffers(self) -> int:
        """The number of buffers."""
        return sum(buffer.numel() for buffer in self.buffers())

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset parameters with default initialization."""

    # NOTE We should run forward with layer(x) instead of layer.forward(x).
    #      In torch.nn.Module, the typing and docstring for forward is not auto copied to __call__.
    #      Therefore we override __call__ here to provide typing and docstring for layer(x) calling.
    # TODO: if pytorch finds a way to sync forward and __call__, we can remove this
    # TODO: `*_: Any` is the temp solution to accomodate IntegralInputLayer, we should refactor that
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
