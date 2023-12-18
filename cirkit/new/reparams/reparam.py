from abc import ABC, abstractmethod
from typing import Callable, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


class Reparameterization(nn.Module, ABC):
    """The abstract base class for all reparameterizations.

    NOTE: This can be materialized only once. Another Reparameterization instance should be \
          constructed if we want to re-materialize.
    """

    def __init__(self) -> None:
        """Init class."""
        super().__init__()
        # All attributes available but empty before materialization.
        self.shape = ()
        self.dims: Tuple[int, ...] = ()  # The sum weight normalization dims; see materialize(dim=).

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
    # essential property of sum weights. If a reparam at some place is expected to be always
    # unnormalized, just explicitly pass dim=().
    # NOTE: We provide a default materialization, but subclasses are still expected to override, and
    #       therefore this is marked @abstractmethod. Yet subclasses are still expected to call
    #       super().materialize(...).
    @abstractmethod
    def materialize(self, shape: Sequence[int], /, *, dim: Union[int, Sequence[int]]) -> bool:
        """Materialize the internal parameter tensors with given shape.

        If it is already materialized, False will be returned to indicate no materialization. \
        However, a second call to materialize must give the same config, so that the underlying \
        params can indeed be reused.

        The initial value of the parameter after materialization is not guaranteed, and explicit \
        initialization is expected.

        The kwarg, dim, is used to hint the normalization of sum weights. It's not always used but \
        must be supplied with the sum-to-1 dimension(s) so that it's guaranteed to be available \
        when a normalized reparam is passed as self.

        Args:
            shape (Sequence[int]): The shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied. However a subclass impl may choose to ignore this.

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
            assert (
                self.shape == shape and self.dims == dims
            ), "Reparameterization cannot be re-materialized into a different configuration."
            return False

        self.shape = shape
        self.dims = dims
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
