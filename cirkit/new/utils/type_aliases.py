from typing import TYPE_CHECKING, Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
from typing_extensions import TypeAlias  # FUTURE: in typing from 3.10, deprecated in 3.12
from typing_extensions import TypedDict  # FUTURE: in typing from 3.11 for generic TypedDict
from typing_extensions import NotRequired, Required  # FUTURE: in typing from 3.11

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    # NOTE: The following must be quoted in type annotations.
    from cirkit.new.layers import Layer
    from cirkit.new.reparams import Reparameterization

# Here're all the type defs and aliases shared across the library.
# If a type is private and only used in one file, it can also be defined there.


class ClampBounds(TypedDict, total=False):
    """Wrapper of the kwargs for torch.clamp().

    Items can be either missing or None for no clamping in the corresponding direction.
    """

    min: Optional[float]
    max: Optional[float]


# The allowed value types are what can be saved in json.
RGNodeMetadata: TypeAlias = Dict[str, Union[int, float, str, bool]]
"""The type of RGNode.metadata."""


class RegionDict(TypedDict):
    """The structure of a region node in the json file."""

    scope: List[int]  # The scope of this region node, specified by id of variable.
    metadata: NotRequired[RGNodeMetadata]  # The metadata of this region node, if any.


class PartitionDict(TypedDict):
    """The structure of a partition node in the json file."""

    inputs: List[int]  # The inputs of this partition node, specified by id of region node.
    output: int  # The output of this partition node, specified by id of region node.
    metadata: NotRequired[RGNodeMetadata]  # The metadata of this partition node, if any.


class RegionGraphJson(TypedDict):
    """The structure of the region graph json file."""

    # The regions of RG represented by a mapping from id in str to either a dict or only the scope.
    regions: Dict[str, Union[RegionDict, List[int]]]
    # The graph of RG represented by a list of partitions.
    graph: List[PartitionDict]


ReparamFactory: TypeAlias = Callable[[], "Reparameterization"]
OptReparamFactory: TypeAlias = Callable[[], Optional["Reparameterization"]]


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
