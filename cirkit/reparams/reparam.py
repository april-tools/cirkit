from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, nn


class Reparameterization(nn.Module, ABC):
    """The base class for all reparameterizaions."""

    log_mask: Optional[Tensor]  # to be registered as buffer

    # TODO: for now **_: Any is added for unified interface. investigate if really need
    # TODO: accept *size like tensor constructors?

    def __init__(  # type: ignore[misc]
        self,
        size: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        mask: Optional[Tensor] = None,
        log_mask: Optional[Tensor] = None,
        **_: Any,
    ) -> None:
        """Init class.

        The three kwargs, dim, mask/log_mask, are used to hint the normalization of sum weights. \
        The dim kwarg must be supplied to hint the sum-to-1 dimension, but mask/log_mask can be \
        optional. Subclasses choose whether and how to use them (e.g. unnormalized reparams don't).

        Args:
            size (Sequence[int]): The size/shape of the output parameter.
            dim (Union[int, Sequence[int]]): The dimension(s) along which the normalization will \
                be applied. However a subclass impl may choose not to use this.
            mask (Optional[Tensor], optional): The 0/1 mask for normalization positions. None for \
                no masks. Must not be used together with log_mask. The shape must be broadcastable \
                to size, if not None. Defaults to None.
            log_mask (Optional[Tensor], optional): The -inf/0 mask for normalization positions. \
                None for no masks. Must not be used together with mask. The shape must be \
                broadcastable to size, if not None. Defaults to None. Defaults to None.
        """
        super().__init__()
        self.shape = tuple(size)

        if isinstance(dim, Sequence):
            assert all(
                -len(size) <= d < len(size) for d in dim
            ), f"dim={dim} out of range for {len(size)}-d."
            self.dims = tuple(sorted(d if d >= 0 else d + len(size) for d in dim))
        else:
            assert -len(size) <= dim < len(size), f"dim={dim} out of range for {len(size)}-d."
            self.dims = (dim if dim >= 0 else dim + len(size),)

        assert mask is None or log_mask is None, "mask and log_mask may not be supplied together."

        # Currently only saves log_mask. We can add mask if useful
        if mask is not None:
            # broadcast_to raises RuntimeError when not broadcastable
            mask.broadcast_to(size)
            self.register_buffer("log_mask", torch.log(mask))
        elif log_mask is not None:
            log_mask.broadcast_to(size)
            self.register_buffer("log_mask", log_mask)
        else:
            self.register_buffer("log_mask", None)

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype of the output param."""

    def __call__(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        return super().__call__()  # type: ignore[no-any-return,misc]

    @abstractmethod
    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """

    def _flatten_dims(self, x: Tensor) -> Tensor:
        """Flatten the dims of self.dims in the input with the shape of self.shape.

        Args:
            x (Tensor): The tensor to be flattened, expected to have shape of self.shape.

        Returns:
            Tensor: The flattened tensor.
        """
        if not self.dims:
            return x

        perm = (
            tuple(range(0, self.dims[0]))
            + self.dims
            + tuple(d for d in range(self.dims[0], len(self.shape)) if d not in self.dims)
        )
        # TODO: consider torch.movedim?
        # flatten end_dim is inclusive
        return x.permute(perm).flatten(
            start_dim=self.dims[0], end_dim=self.dims[0] + len(self.dims) - 1
        )

    def _unflatten_dims(self, x: Tensor) -> Tensor:
        """Unflatten the dims of self.dims to get the output with the shape of self.shape.

        Args:
            x (Tensor): The tensor to be unflattened.

        Returns:
            Tensor: The unflattened tensor.
        """
        if not self.dims:
            return x

        perm = (
            tuple(range(0, self.dims[0]))
            + self.dims
            + tuple(d for d in range(self.dims[0], len(self.shape)) if d not in self.dims)
        )
        inv_perm = tuple(perm.index(d) for d in range(len(self.shape)))
        # TODO: x.unflatten is not typed
        return torch.unflatten(
            x, dim=self.dims[0], sizes=[self.shape[d] for d in self.dims]
        ).permute(inv_perm)
