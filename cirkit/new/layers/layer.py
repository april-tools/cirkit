from abc import ABC, abstractmethod
from functools import cached_property
from typing import Callable, ClassVar, Optional, Sequence, Type, Union

import torch
from torch import Tensor, nn

from cirkit.new.reparams import Reparameterization
from cirkit.new.utils import ComputationSapce
from cirkit.new.utils.type_aliases import SymbCfgFactory, SymbLayerCfg


class Layer(nn.Module, ABC):
    """The abstract base class for all layers."""

    # NOTE: This is uninitialized for the class, but to be set later.
    comp_space: ClassVar[Type[ComputationSapce]]
    """The computational space for all Layers."""

    # DISABLE: reparam is not used in the base class. It's only here for the interface.
    def __init__(
        self,
        *,
        num_input_units: int,
        num_output_units: int,
        arity: int = 0,
        reparam: Optional[Reparameterization] = None,  # pylint: disable=unused-argument
    ) -> None:
        """Init class.

        Args:
            num_input_units (int): The number of input units.
            num_output_units (int): The number of output units.
            arity (int, optional): The arity of the layer. Defaults to 0.
            reparam (Optional[Reparameterization], optional): The reparameterization for layer \
                parameters, can be None if the layer has no params. Defaults to None.
        """
        super().__init__()
        assert num_input_units > 0, "The number of input units must be positive."
        assert num_output_units > 0, "The number of output units must be positive."
        assert arity > 0, "The arity must be positive."
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.arity = arity

    # We enforce subclasses to implement this property so that we don't forget it. For layers that
    # don't have parameters, just return None.
    @property
    @abstractmethod
    def _default_initializer_(self) -> Optional[Callable[[Tensor], Tensor]]:
        """The default inplace initializer for the parameters of this layer.

        The function should modify a Tensor inplace and also return its value. Or use None for no \
        initialization.
        """

    def materialize_params(
        self, shape: Sequence[int], /, *, dim: Union[int, Sequence[int]]
    ) -> None:
        """Wrap self.params.materialize() with initializer_=self._default_initializer_, if \
        self.params is a valid Reparameterization.

        For details of the args and kwargs, see Reparameterization.materialize().

        Args:
            shape (Sequence[int]): The shape for self.params.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied.
        """
        # IGNORE: getattr gives Any.
        if not isinstance(
            self_params := getattr(self, "params", None), Reparameterization  # type: ignore[misc]
        ):
            return

        self_params.materialize(shape, dim=dim, initializer_=self._default_initializer_)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset parameters with default initialization."""
        initializer_ = self._default_initializer_
        for child in self.children():  # Only immediate children, not all modules().
            if isinstance(child, Reparameterization) and initializer_ is not None:
                child.initialize(initializer_)
            if isinstance(child, Layer):  # Recursively reset with their own initializer_.
                child.reset_parameters()

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

    @classmethod
    @abstractmethod
    def get_product(
        cls, left_symb_cfg: SymbLayerCfg["Layer"], right_symb_cfg: SymbLayerCfg["Layer"]
    ) -> SymbCfgFactory["Layer"]:
        """Get the symbolic config to construct the product of this layer and the other layer.

        Because layer product (Kronecker of units) is not commutative, here we allow any one of \
        the two operands to be of self class to enable reflected implementation, but at least one \
        of them should be. The detail is slightly different for different Layer subclasses:
            - For InputLayer, we allow any two (or same) InputLayer to be multiplied;
            - For InnerLayer, we require both left and right to be the same class.

        NOTE: To avoid excessive repetition of overloads, the signature typing is not narrowed \
              down, and potential errors will not be captured by static checkers but only during \
              runtime. Implementation should check and may impose further constraints.

        Args:
            left_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the left operand.
            right_symb_cfg (SymbLayerCfg[Layer]): The symbolic config for the right operand.

        Returns:
            SymbCfgFactory[Layer]: The symbolic config for the product. NOTE: Implicit to typing, \
                NotImplemented may also be returned, which indicates the reflection should be tried.
        """
