from typing import Optional, TypedDict

# Here're all the type defs and aliases shared across the lib.
# For private types that is only used in one file, can be defined there.

# TODO: move other commonly used here


class ClampBounds(TypedDict, total=False):
    """Wrapper of the kwargs for `torch.clamp()`.

    Items can be either missing or None to disable clamping in corresponding direction.
    """

    min: Optional[float]
    max: Optional[float]
