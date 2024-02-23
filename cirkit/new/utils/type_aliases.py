from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
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
    from cirkit.new.symbolic import GenericSymbolicLayer

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


# NOTE: The following related to symbolic config. Although there are definitions on actual behaviour
#       aside from type definitions, we put them here because they are mainly for type def, and are
#       too widely used that may cause cyclic import if put elsewhere.


# For subclass compatibility, covariance is needed.
LayerT_co = TypeVar("LayerT_co", bound="Layer", covariant=True)


# NOTE: We add frozen=True because this should be immutable. Also covariant TypeVar should only
#       appear in immutable generic classes.
# FUTURE: kw_only=True in 3.10
# IGNORE: SymbCfgFactory contains Any.
@dataclass(frozen=True)  # type: ignore[misc]
class SymbCfgFactory(Generic[LayerT_co]):  # type: ignore[misc]
    """The config to construct a symbolic layer, possibly including a factory.

    This is part of the SymbolicLayer constructor and provides the basic specification of how a \
    SymbL is tensorized, by referencing the corresponding Layer class. It also includes a \
    Reparameterization object which can be useful to parameter sharing. Optionally, a \
    ReparamFactory can be provided, which will take precedence, to be used for constructing new \
    reparams instead of reusing.

    We make this class generic so that it can be specific on a Layer subclass.
    """

    layer_cls: Type[LayerT_co]
    # IGNORE: Unavoidable for kwargs.
    layer_kwargs: Mapping[str, Any] = field(default_factory=dict)  # type: ignore[misc]
    reparam: Optional["Reparameterization"] = None
    reparam_factory: Optional[ReparamFactory] = None

    def instantiate(
        self, symb_layer: "GenericSymbolicLayer[LayerT_co]"
    ) -> "SymbLayerCfg[LayerT_co]":
        """Instantiate a SymbLayerCfg for a SymbolicLayer, optionally also instantiate the reparam \
        from factory.

        Args:
            symb_layer (GenericSymbolicLayer[LayerT_co]): The SymbL for the instantiated config.

        Returns:
            SymbLayerCfg[LayerT_co]: The config for the SymbL.
        """
        # IGNORE: Unavoidable for kwargs.
        return SymbLayerCfg(
            layer_cls=self.layer_cls,
            layer_kwargs=self.layer_kwargs,  # type: ignore[misc]
            reparam=self.reparam_factory() if self.reparam_factory is not None else self.reparam,
            symb_layer=symb_layer,
        )


# FUTURE: kw_only=True in 3.10
# IGNORE: SymbLayerCfg contains Any.
@dataclass(frozen=True)  # type: ignore[misc]
class SymbLayerCfg(SymbCfgFactory[LayerT_co]):  # type: ignore[misc]
    """The config for a symbolic layer.

    When a SymbolicLayer is constructed based on a config (factory), it saves an instantiated \
    version of the config, which keeps a reference to self (the SymbL). This layer config should \
    be held by only this one SymbL, so that we can track the SymbL corresponding to the config.
    """

    reparam_factory: None = None
    # IGNORE: This is expected to be kw_only (with future 3.10).
    symb_layer: "GenericSymbolicLayer[LayerT_co]" = None  # type: ignore[assignment]

    # NOTE: Inherited instantiate() can provide "copy" with symb_layer replaced.
