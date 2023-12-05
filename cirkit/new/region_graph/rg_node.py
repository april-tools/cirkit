from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


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
        self.scope = frozenset(scope)
        assert self.scope, "The scope of a node in RG must not be empty."

        # The edge lists are initiated empty because a node may be contructed without the whole RG.
        self.inputs: List[RGNode] = []
        self.outputs: List[RGNode] = []

        # TODO: we might want to save something, but this is not used yet.
        self._metadata: Dict[str, Any] = {}  # type: ignore[misc]

    def __repr__(self) -> str:
        """Generate the repr string of the node.

        Returns:
            str: The str representation of the node.
        """
        # Here we convert scope to set so that we don't get "fronzenset(...)" in output.
        return f"{self.__class__.__name__}({set(self.scope)})"

    # `other: Self` is wrong as it can be RGNode instead of just same as self.
    def __lt__(self, other: "RGNode") -> bool:
        """Compare the node with another node, can be used for sorting.

        The default comparison is:
        - First, RegionNode is smaller than PartitionNode;
        - Then, the node with smaller scope (by frozenset.__lt__) is smaller;
        - Finally, same type of nodes with same scope are ordered by hash (mem addr by default).

        This guarantees two nodes compare equal only when they're the same object.

        Args:
            other (RGNode): The other node to compare.

        Returns:
            bool: Whether self < other.
        """
        # A trick to compare classes: if the class name is equal, then the class is the same;
        # otherwise "P" < "R" but RegionNode < PartitionNode, so class names are reversed below.
        return (other.__class__.__name__, self.scope, hash(self)) < (
            self.__class__.__name__,
            other.scope,
            hash(other),
        )


# Disable: It's intended for RegionNode. It's only used to provide a concrete RGNode with nothing.
class RegionNode(RGNode):  # pylint: disable=too-few-public-methods
    """The region node in the region graph."""

    # Ignore: Mutable types are typically invariant, so there's no other choice, and we can only
    #         enforce the typing with ignore.
    inputs: List["PartitionNode"]  # type: ignore[assignment]
    outputs: List["PartitionNode"]  # type: ignore[assignment]

    # TODO: better way to impl this? we must have an abstract method in RGNode
    def __init__(self, scope: Iterable[int]) -> None:  # pylint: disable=useless-parent-delegation
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__(scope)


# Disable: It's intended for RegionNode. It's only used to provide a concrete RGNode with nothing.
class PartitionNode(RGNode):  # pylint: disable=too-few-public-methods
    """The partition node in the region graph."""

    # Ignore: Mutable types are typically invariant, so there's no other choice, and we can only
    #         enforce the typing with ignore.
    inputs: List["RegionNode"]  # type: ignore[assignment]
    outputs: List["RegionNode"]  # type: ignore[assignment]

    # TODO: better way to impl this? we must have an abstract method in RGNode
    def __init__(self, scope: Iterable[int]) -> None:  # pylint: disable=useless-parent-delegation
        """Init class.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__(scope)
