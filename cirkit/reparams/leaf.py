from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from cirkit.utils.type_aliases import ClampBounds

from .reparam import Reparameterization


class ReparamLeaf(Reparameterization):
    """A leaf in reparameterizaion that holds the parameter instance and does simple transforms.

    There's no param initialization here. That's the responsibility of Layers.
    """

    def __init__(
        self,
        size: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        mask: Optional[Tensor] = None,
        log_mask: Optional[Tensor] = None,
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
        super().__init__(size, dim=dim, mask=mask, log_mask=log_mask)
        self.param = nn.Parameter(torch.empty(size))

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the output param."""
        return self.param.dtype


class ReparamIdentity(ReparamLeaf):
    """Identity reparametrization.

    Range: (-inf, +inf).
    """

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        # TODO: is there any way to inherit the docstring?
        return self.param


class ReparamExp(ReparamLeaf):
    """Exp reparametrization.

    Range: (0, +inf).
    """

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        return torch.exp(self.param)


class ReparamSquare(ReparamLeaf):
    """Square reparametrization.

    Range: [0, +inf).
    """

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        return torch.square(self.param)


class ReparamClamp(ReparamLeaf):
    """Clamp reparametrization.

    Range: [min, max], as provided.

    Note that clamped values stop gradient.
    """

    def __init__(  # type: ignore[misc]
        self,
        size: Sequence[int],
        /,
        *,
        min: Optional[float] = None,  # pylint: disable=redefined-builtin
        max: Optional[float] = None,  # pylint: disable=redefined-builtin
        **kwargs: Any,  # hold dim/mask/log_mask, but irrelevant here.
    ) -> None:
        """Init class.

        Args:
            size (Sequence[int]): The size of the parameter.
            min (Optional[float], optional): The lower-bound for clamping, None to disable this \
                direction. Defaults to None.
            max (Optional[float], optional): The upper-bound for clamping, None to disable this \
                direction. Defaults to None.
        """
        super().__init__(size, **kwargs)  # type: ignore[misc]
        self.clamp_bounds: ClampBounds = {"min": min, "max": max}

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        return torch.clamp(self.param, **self.clamp_bounds)


class ReparamSigmoid(ReparamLeaf):
    """Sigmoid (with temperature) reparametrization.

    Range: (offset, offset+scale), as provided.

    Calculates scale*sigmoid(x/temp)+offset.
    """

    def __init__(  # type: ignore[misc]
        self,
        size: Sequence[int],
        /,
        *,
        temperature: float = 1,
        scale: float = 1,
        offset: float = 0,
        **kwargs: Any,  # hold dim/mask/log_mask, but irrelevant here.
    ) -> None:
        """Init class.

        Args:
            size (Sequence[int]): The size of the parameter.
            temperature (float, optional): The temperature for sigmoid. Defaults to 1.
            scale (float, optional): The scale for sigmoid. Defaults to 1.
            offset (float, optional): The offset for sigmoid. Defaults to 0.
        """
        super().__init__(size, **kwargs)  # type: ignore[misc]
        self.temperature = temperature
        self.scale = scale
        self.offset = offset

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        # TODO: split out a linear reparam?
        return self.scale * torch.sigmoid(self.param / self.temperature) + self.offset


class ReparamSoftmax(ReparamLeaf):
    """Softmax reparametrization.

    Range: (0, 1), 0 available through mask, 1 available when only one element.
    Constraints: sum to 1.
    """

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        param = self.param if self.log_mask is None else self.param + self.log_mask
        # torch.softmax can only accept one dim
        param = self._unflatten_dims(torch.softmax(self._flatten_dims(param), dim=self.dims[0]))
        # nan will appear when the only 1 element is masked. fill nan as 1 (0 in log-sapce)
        # +inf and -inf will not appear
        return torch.nan_to_num(param, nan=1)


class ReparamLogSoftmax(ReparamLeaf):
    """LogSoftmax reparametrization that is more nemarically-stable than log(ReparamSoftmax).

    Range: (-inf, 0), and -inf available through mask, 0 available when only one element.
    Constraints: logsumexp to 0.
    """

    def forward(self) -> Tensor:
        """Get the reparameterized params.

        Returns:
            Tensor: The params after reparameterizaion.
        """
        param = self.param if self.log_mask is None else self.param + self.log_mask
        # torch.softmax can only accept one dim
        param = self._unflatten_dims(torch.log_softmax(self._flatten_dims(param), dim=self.dims[0]))
        # redundant projection of -inf to stop gradients
        # although by default +inf is projected to max_finite, we don't have positive values here
        return torch.nan_to_num(param, neginf=float("-inf"))
