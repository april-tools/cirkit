from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


class Reparameterization(nn.Module, ABC):
    """The abstract base class for all reparameterizations.

    NOTE: This can be materialized only once. Another Reparameterization instance should be \
          constructed if we want to re-materialize.
    """

    # We can also save mask as buffer, but now we only use log_mask.
    log_mask: Optional[Tensor]
    """The log of normalization mask, shape same as the parameter itself."""

    def __init__(self) -> None:
        """Init class."""
        super().__init__()
        # All attributes available but empty before materialization.
        self.shape = ()
        self.dims: Tuple[int, ...] = ()  # The sum weight normalization dims; see materialize(dim=).
        # Once registered as buffer, it can be reassigned using self.xxx=..., while still a buffer.
        self.register_buffer("log_mask", None)

    # TODO: should this be a property?
    shape: Tuple[int, ...]
    """The shape of the output parameter."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype of the output parameter."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device of the output parameter."""

    # We forbid re-materialization by checking this flag.
    @property
    def is_materialized(self) -> bool:
        """Whether this reparameterization is already materialized."""
        # self.shape is set during materialization, so it indicates whether is materialized.
        return bool(self.shape)

    # No default value for dim, so we don't forget to provide one. We require dim because it's an
    # essential property of sum weights. For input params that may be unnormalized, just explicitly
    # provide dim=().
    # NOTE: We provide a default materialization, but subclasses are still expected to override, and
    #       therefore this is marked @abstractmethod. Yet subclasses are still expected to call
    #       super().materialize(...).
    @abstractmethod
    def materialize(
        self,
        shape: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        mask: Optional[Tensor] = None,
        log_mask: Optional[Tensor] = None,
    ) -> bool:
        """Materialize the internal parameter tensors with given shape.

        If it is already materialized, False will be returned to indicate no materialization. \
        However, a second call to materialize must give the same config, so that the underlying \
        params can indeed be reused.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        The three kwargs, dim, mask/log_mask, are used to hint the normalization of sum weights. \
        The dim kwarg must be supplied to hint the sum-to-1 dimension, but mask/log_mask can be \
        optional and at most one can be provided. Subclasses may choose whether and how to use \
        them (e.g. unnormalized reparams just accept them and ignore).

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied. However a subclass impl may choose to ignore this.
            mask (Optional[Tensor], optional): The 0/1 mask for normalization positions. None for \
                no masking. The shape must be broadcastable to shape if not None. Defaults to None.
            log_mask (Optional[Tensor], optional): The -inf/0 mask for normalization positions. \
                None for no masking. The shape must be broadcastable to shape if not None. \
                Defaults to None.

        Returns:
            bool: Whether the materialization is done.
        """
        shape = tuple(shape)

        dims = (
            tuple(sorted(d if d >= 0 else d + len(shape) for d in dim))
            if isinstance(dim, Sequence)
            else (dim if dim >= 0 else dim + len(shape),)
        )
        assert all(0 <= d < len(shape) for d in dims), f"dim={dim} out of range for {len(shape)}-d."

        if self.is_materialized:
            # TODO: check for mask?
            assert (
                self.shape == shape and self.dims == dims
            ), "Reparameterization cannot be re-materialized into a different configuration."
            return False

        self.shape = shape
        self.dims = dims

        assert mask is None or log_mask is None, "mask and log_mask may not be supplied together."

        if mask is not None:
            # An easy way to check if broadcastable: broadcast_to raises RuntimeError when not.
            mask.broadcast_to(shape)
            self.log_mask = torch.log(mask)  # Already registered as buffer in __init__
        elif log_mask is not None:
            log_mask.broadcast_to(shape)
            self.log_mask = log_mask
        # else: both is None, self.log_mask = None, which is the default.

        return True

    @abstractmethod
    def initialize(self, initializer_: Callable[[Tensor], Tensor]) -> None:
        """Initialize the internal parameter tensors with the given initializer.

        Initialization will cause error if not materialized first.

        Subclasses may choose how to use the value given by the initializer, and transformations \
        may be applied depending on the implementation.

        Args:
            initializer_ (Callable[[Tensor], Tensor]): A function that can initialize a tensor \
                inplace while also returning the value.
        """

    def __call__(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
        # Ignore: Idiom for nn.Module.__call__.
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        """Get the reparameterized parameters.

        Returns:
            Tensor: The parameters after reparameterization.
        """
