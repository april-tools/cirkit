from typing import Dict, List, Optional, Sequence, TypedDict, Union
from typing_extensions import Required  # TODO: in typing from 3.11

from torch import Tensor

# Here're all the type defs and aliases shared across the library.
# If a type is private and only used in one file, it can also be defined there.


class ClampBounds(TypedDict, total=False):
    """Wrapper of the kwargs for torch.clamp().

    Items can be either missing or None to disable clamping in corresponding direction.
    """

    min: Optional[float]
    max: Optional[float]


class PartitionDict(TypedDict):
    """The struction of a partition in the json file."""

    # TODO: more description?

    p: int
    l: int
    r: int

    # TODO: for more than 2 inputs
    # i: List[int]
    # o: int


class RegionGraphJson(TypedDict):
    """The structure of region graph json file."""

    # TODO: more description?

    regions: Dict[str, List[int]]
    graph: List[PartitionDict]


class MaterializeKwargs(TypedDict, total=False):
    """Wrapper of the kwargs for Reparameterization.materialize().

    See Reparameterization.materialize() for details, including Required or default value.
    """

    dim: Required[Union[int, Sequence[int]]]
    mask: Optional[Tensor]
    log_mask: Optional[Tensor]
