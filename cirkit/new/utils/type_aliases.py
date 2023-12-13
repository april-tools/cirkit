from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import Required  # TODO: in typing from 3.11
from typing_extensions import TypedDict  # TODO: in typing from 3.11 for generic TypedDict

from torch import Tensor

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    # NOTE: The following must be quoted in type annotations.
    from cirkit.new.layers import Layer
    from cirkit.new.reparams import Reparameterization

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


ReparamFactory = Callable[[], "Reparameterization"]
OptReparamFactory = Callable[[], Optional["Reparameterization"]]


LayerT_co = TypeVar("LayerT_co", bound="Layer", covariant=True)


# We need a covariant LayerT_co for subclass compatibility. Also, we don't expect this dict to
# behave as mutable, so covariance can be safe to adopt.
class SymbLayerCfg(TypedDict, Generic[LayerT_co], total=False):
    """The config of a layer in symbolic form, which is part of SymbolicLayer constructor.

    We make it generic so that it can be specific on a Layer subclass.
    """

    layer_cls: Required[Type[LayerT_co]]
    layer_kwargs: Dict[str, Any]
    reparam: Optional["Reparameterization"]
