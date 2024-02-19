from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable, cast

from cirkit.new.utils import OrderedSet, Scope
from cirkit.new.utils.type_aliases import RGNodeMetadata


class RGNode(ABC):
    """The abstract base class for nodes in region graphs."""

    def __init__(self, scope: Iterable[int]) -> None:
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__()
        self.scope = Scope(scope)
        assert self.scope, "The scope of a node in RG must not be empty."

        # NOTE: metadata is used for anything that is not part of the RG but is useful for node.
        # ANNOTATE: Specify content for empty container.
        self.metadata: RGNodeMetadata = {}

    # NOTE: These two edge tables are initiated empty because a node may be contructed without
    #       knowing the whole RG structure.
    #       We use cached_property and construct an empty container inside, to achieve:
    #       - No _inputs/_outputs is explicitly needed;
    #       - The same mutable object is returned every time the property is accessed.
    @cached_property
    @abstractmethod
    def inputs(self) -> OrderedSet["RGNode"]:
        """The input nodes of this node."""
        return OrderedSet()

    @cached_property
    @abstractmethod
    def outputs(self) -> OrderedSet["RGNode"]:
        """The output nodes of this node."""
        return OrderedSet()

    def __repr__(self) -> str:
        """Generate the repr string of the node.

        Returns:
            str: The str representation of the node.
        """
        return f"{type(self).__name__}@0x{id(self):x}({self.scope})"

    # __hash__ and __eq__ are defined by default to compare on object identity, i.e.,
    # (a is b) <=> (a == b) <=> (hash(a) == hash(b)).

    # `other: Self` is wrong as it can be RGNode instead of just same as self.
    def __lt__(self, other: "RGNode") -> bool:
        """Compare the node with another node, for < operator implicitly used in sorting.

        The comparison between two RGNode is:
            - If they have different scopes, the one with a smaller scope is smaller;
            - With the same scope, PartitionNode is smaller than RegionNode;
            - For same type of node and same scope, an extra sort_key can be provided in \
                self.metadata to define the order;
            - If the above cannot compare, __lt__ is always False, indicating "equality for the \
                purpose of sorting".

        With the extra sorting key provided, it is guaranteed to have total ordering, i.e., \
        exactly one of a == b, a < b, a > b is True, and will lead to a deterministic sorted order.

        This comparison also guarantees the topological order in a (smooth and decomposable) RG:
            - For a RegionNode->PartitionNode edge, Region.scope < Partition.scope;
            - For a PartitionNode->RegionNode edge, they have the same scope and Partition < Region.

        Args:
            other (RGNode): The other node to compare with.

        Returns:
            bool: Whether self < other.
        """
        # A trick to compare classes: if the class name is equal, then the class is the same;
        # otherwise "P" < "R" and PartitionNode < RegionNode, so comparison of class names works.
        # And amazingly, all possible types in metadata are comparable.
        return (self.scope, type(self).__name__, self.metadata.get("sort_key", -1)) < (
            other.scope,
            type(other).__name__,
            other.metadata.get("sort_key", -1),
        )


# In the following RGNode subclasses we simply force a narrowed type on inputs/outputs, while
# concretizing the abstract methods. All the actual implementation is already in RGNode.
# CAST: We know a narrower type so we enforce the narrowing.
# IGNORE: Mutable container is invariant, and is typically incompatible with another, but we know
#         it's correct here.


class RegionNode(RGNode):
    """The region node in the region graph."""

    @cached_property
    def inputs(self) -> OrderedSet["PartitionNode"]:  # type: ignore[override]
        """The input nodes of this node."""
        return cast(OrderedSet[PartitionNode], super().inputs)

    @cached_property
    def outputs(self) -> OrderedSet["PartitionNode"]:  # type: ignore[override]
        """The output nodes of this node."""
        return cast(OrderedSet[PartitionNode], super().outputs)


class PartitionNode(RGNode):
    """The partition node in the region graph."""

    @cached_property
    def inputs(self) -> OrderedSet["RegionNode"]:  # type: ignore[override]
        """The input nodes of this node."""
        return cast(OrderedSet[RegionNode], super().inputs)

    @cached_property
    def outputs(self) -> OrderedSet["RegionNode"]:  # type: ignore[override]
        """The output nodes of this node."""
        return cast(OrderedSet[RegionNode], super().outputs)
