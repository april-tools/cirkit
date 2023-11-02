from typing import Optional, Protocol, Sequence, TypedDict, Union

from torch import Tensor

from cirkit.reparams.reparam import Reparameterizaion

# Here're all the type defs and aliases shared across the lib.
# For private types that is only used in one file, can be defined there.

# TODO: move other commonly used here


class ClampBounds(TypedDict, total=False):
    """Wrapper of the kwargs for `torch.clamp()`.

    Items can be either missing or None to disable clamping in corresponding direction.
    """

    min: Optional[float]
    max: Optional[float]


class ReparamFactory(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for Callable that mimics Reparameterizaion constructor."""

    def __call__(
        self,
        size: Sequence[int],
        /,
        *,
        dim: Union[int, Sequence[int]],
        mask: Optional[Tensor] = None,
        log_mask: Optional[Tensor] = None,
    ) -> Reparameterizaion:
        """Construct a Reparameterizaion object."""
        # TODO: pylance issue, ellipsis is required here
        ...  # pylint:disable=unnecessary-ellipsis
