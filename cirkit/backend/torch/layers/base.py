from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar, Dict, Optional, Type

from torch import Tensor, nn

from cirkit.backend.torch.semiring import SemiringCls, SumProductSemiring
from cirkit.backend.torch.utils import InitializerFunc


class TorchLayer(nn.Module, ABC):
    """The abstract base class for all layers."""

    # DISABLE: reparam is not used in the base class. It's only here for the interface.
    def __init__(
        self,
        num_input_units: int,
        num_output_units: int,
        *,
        arity: int = 1,
        semiring: Optional[SemiringCls] = None,
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 1.
        """
        super().__init__()
        assert num_input_units > 0, "The number of input units must be positive."
        assert num_output_units > 0, "The number of output units must be positive."
        assert arity > 0, "The arity must be positive."
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity
        self.semiring = semiring if semiring is not None else SumProductSemiring

    @classmethod
    def default_initializers(cls) -> Dict[str, InitializerFunc]:
        return {}

    @classmethod
    def get_default_initializer(cls, name: str) -> InitializerFunc:
        return cls.default_initializers()[name]

    # Expected to be fixed, so use cached property to avoid recalculation.
    @cached_property
    def num_parameters(self) -> int:
        """The number of params."""
        return sum(p.numel() for p in self.parameters())

    # Expected to be fixed, so use cached property to avoid recalculation.
    @cached_property
    def num_buffers(self) -> int:
        """The number of buffers."""
        return sum(buffer.numel() for buffer in self.buffers())

    # We should run forward with layer(x) instead of layer.forward(x). However, in nn.Module, the
    # typing and docstring for forward is not auto-copied to __call__. Therefore, we override
    # __call__ here to provide a complete interface and documentation for layer(x).
    # NOTE: Should we want to change the interface or docstring of forward in subclasses, __call__
    #       also needs to be overriden to sync the change.
    # TODO: if pytorch finds a way to sync forward and __call__, we can remove this __call__
    def __call__(self, x: Tensor) -> Tensor:
        """Invoke the forward function.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
        # IGNORE: Idiom for nn.Module.__call__.
        return super().__call__(x)  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass.

        Args:
            x (Tensor): The input to this layer, shape (H, *B, Ki).

        Returns:
            Tensor: The output of this layer, shape (*B, Ko).
        """
