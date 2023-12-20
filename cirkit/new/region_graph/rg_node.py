from abc import ABC, abstractmethod
from typing import Iterable

from cirkit.new.utils import OrderedSet, Scope
from cirkit.new.utils.type_aliases import RGNodeMetadata


class RGNode(ABC):
    """The abstract base class for nodes in region graphs."""

    # We enforce __init__ to be abstract so that RGNode cannot be instantiated.
    @abstractmethod
    def __init__(self, scope: Iterable[int]) -> None:
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__()
        self.scope = Scope(scope)
        assert self.scope, "The scope of a node in RG must not be empty."

        # The edge tables are initiated empty because a node may be contructed without the whole RG.
        # ANNOTATE: Specify content for empty container.
        self.inputs: OrderedSet[RGNode] = OrderedSet()
        self.outputs: OrderedSet[RGNode] = OrderedSet()

        # NOTE: metadata is used for anything that is not part of the RG but is useful for node.
        self.metadata: RGNodeMetadata = {}

    def __repr__(self) -> str:
        """Generate the repr string of the node.

        Returns:
            str: The str representation of the node.
        """
        return f"{self.__class__.__name__}@0x{id(self):x}({self.scope})"

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
        return (self.scope, self.__class__.__name__, self.metadata.get("sort_key", -1)) < (
            other.scope,
            other.__class__.__name__,
            other.metadata.get("sort_key", -1),
        )


# DISABLE: It's designed to have these methods.
class RegionNode(RGNode):  # pylint: disable=too-few-public-methods
    """The region node in the region graph."""

    # IGNORE: Mutable containers are invariant, so there's no other choice.
    inputs: OrderedSet["PartitionNode"]  # type: ignore[assignment]
    outputs: OrderedSet["PartitionNode"]  # type: ignore[assignment]

    # TODO: better way to impl this? we must have an abstract method in RGNode
    def __init__(self, scope: Iterable[int]) -> None:  # pylint: disable=useless-parent-delegation
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__(scope)


# DISABLE: It's designed to have these methods.
class PartitionNode(RGNode):  # pylint: disable=too-few-public-methods
    """The partition node in the region graph."""

    # IGNORE: Mutable containers are invariant, so there's no other choice.
    inputs: OrderedSet["RegionNode"]  # type: ignore[assignment]
    outputs: OrderedSet["RegionNode"]  # type: ignore[assignment]

    # TODO: better way to impl this? we must have an abstract method in RGNode
    def __init__(self, scope: Iterable[int]) -> None:  # pylint: disable=useless-parent-delegation
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__(scope)
