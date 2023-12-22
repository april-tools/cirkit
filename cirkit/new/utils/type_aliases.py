from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
)
from typing_extensions import NotRequired  # FUTURE: in typing from 3.11
from typing_extensions import TypeAlias  # FUTURE: in typing from 3.10, deprecated in 3.12

if TYPE_CHECKING:  # Only imported for static type checking but not runtime, to avoid cyclic import.
    # NOTE: For safety, anything from cirkit should be imported here.
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


# We allow None here because all abstract bases of Layer accepts it.
ReparamFactory: TypeAlias = Callable[[], Optional["Reparameterization"]]


# For subclass compatibility, covariance is needed.
LayerT_co = TypeVar("LayerT_co", bound="Layer", covariant=True)


# NOTE: We add frozen=True because this should be immutable. Also covariant TypeVar should only
#       appear in immutable generic classes.
# FUTURE: kw_only=True in 3.10
# IGNORE: SymbLayerCfg contains Any (due to kwargs).
@dataclass(frozen=True)  # type: ignore[misc]
class SymbLayerCfg(Generic[LayerT_co]):  # type: ignore[misc]
    """The config of a layer in symbolic form, which is part of SymbolicLayer constructor that \
    saves the info of the corresponding Layer.

    The reparam can be provided as a factory function, for use as a configuration on how to \
    construct a reparam instance when we want a different instance each time. But in a SymbLayer, \
    an instantiated reparam should be used, so that the same instance can be reused if needed.

    We make this class generic so that it can be specific on a Layer subclass.
    """

    layer_cls: Type[LayerT_co]
    # IGNORE: Unavoidable for kwargs.
    layer_kwargs: Dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]
    reparam: Optional["Reparameterization"] = None
    # reparam_factory: Optional[ReparamFactory] = None  # TODO: to be enabled
