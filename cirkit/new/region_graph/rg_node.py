from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

from cirkit.new.utils import OrderedSet, Scope


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
        self.inputs: OrderedSet[RGNode] = OrderedSet()
        self.outputs: OrderedSet[RGNode] = OrderedSet()

        # TODO: we might want to save something, but this is not used yet.
        self._metadata: Dict[str, Any] = {}  # type: ignore[misc]

    def __repr__(self) -> str:
        """Generate the repr string of the node.

        Returns:
            str: The str representation of the node.
        """
        return f"{self.__class__.__name__}({self.scope})"

    # __hash__ and __eq__ are defined by default to compare on object identity, i.e.,
    # (a is b) <=> (a == b) <=> (hash(a) == hash(b)).

    # `other: Self` is wrong as it can be RGNode instead of just same as self.
    def __lt__(self, other: "RGNode") -> bool:
        """Compare the node with another node, for < operator implicitly used in sorting.

        TODO: the following is currently NOT correct because the sorting rule is not complete.
        It is guaranteed that exactly one of a == b, a < b, a > b is True. Can be used for \
        sorting and order is guaranteed to be always stable.
        TODO: alternative if we ignore the above todo:
        Note that a != b does not imply a < b or b < a, as the order within the the same type of \
        node with the same scope is not defined, in which case a == b, a < b, b < a are all false. \
        Yet although there's no total ordering, sorting can still be performed.

        The comparison between two RGNode is:
        - If they have different scopes, the one with a smaller scope is smaller;
        - With the same scope, PartitionNode is smaller than RegionNode;
        - For same type of node and same scope, __lt__ is always False, indicating "equality for \
            the purpose of sorting".

        This comparison guarantees the topological order in a (smooth and decomposable) RG:
        - For a RegionNode->PartitionNode edge, Region.scope < Partition.scope;
        - For a PartitionNode->RegionNode edge, they have the same scope and Partition < Region.

        Args:
            other (RGNode): The other node to compare with.

        Returns:
            bool: Whether self < other.
        """
        # A trick to compare classes: if the class name is equal, then the class is the same;
        # otherwise "P" < "R" and PartitionNode < RegionNode, so comparison of class names works.
        return (self.scope, self.__class__.__name__) < (other.scope, other.__class__.__name__)


# Disable: It's intended for RegionNode to only have few methods.
class RegionNode(RGNode):  # pylint: disable=too-few-public-methods
    """The region node in the region graph."""

    # Ignore: Mutable containers are invariant, so there's no other choice.
    inputs: OrderedSet["PartitionNode"]  # type: ignore[assignment]
    outputs: OrderedSet["PartitionNode"]  # type: ignore[assignment]

    # TODO: better way to impl this? we must have an abstract method in RGNode
    def __init__(self, scope: Iterable[int]) -> None:  # pylint: disable=useless-parent-delegation
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__(scope)


# Disable: It's intended for PartitionNode to only have few methods.
class PartitionNode(RGNode):  # pylint: disable=too-few-public-methods
    """The partition node in the region graph."""

    # Ignore: Mutable containers are invariant, so there's no other choice.
    inputs: OrderedSet["RegionNode"]  # type: ignore[assignment]
    outputs: OrderedSet["RegionNode"]  # type: ignore[assignment]

    # TODO: better way to impl this? we must have an abstract method in RGNode
    def __init__(self, scope: Iterable[int]) -> None:  # pylint: disable=useless-parent-delegation
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__(scope)
