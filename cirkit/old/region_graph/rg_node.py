import itertools
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

# TODO: rework docstrings


class RGNode(ABC):
    """The base class for nodes in region graphs."""

    # TODO: id for base class or children? only used for sort? really need?
    _id_counter = itertools.count(0)
    """we assign each object a unique id."""

    @abstractmethod
    def __init__(self, scope: Iterable[int]) -> None:
        """Construct the node.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        self.scope = frozenset(scope)
        assert self.scope, "The scope of a node must be non-empty"

        # cannot mark as List[RGNode] because of variance issue
        # they're empty at construction, will be populated when adding edges
        self.inputs: List[Any] = []  # type: ignore[misc]
        self.outputs: List[Any] = []  # type: ignore[misc]

        self.node_id = next(RGNode._id_counter)
        self._sort_key = (tuple(sorted(self.scope)), self.node_id)

        self._metadata: Dict[str, Any] = {}  # type: ignore[misc]

    def __repr__(self) -> str:
        """Generate the `repr` string of the node."""
        return self.__class__.__name__ + ": " + repr(set(self.scope))

    def __lt__(self, other: "RGNode") -> bool:  # we don't use Self as it can compare to RGNode
        """Compare the node with the other, for sorting."""
        # TODO: it's better to be abstract. but do we really need this?
        return self._sort_key < other._sort_key


# TODO: do we need to enforce num public methods?
class RegionNode(RGNode):  # pylint: disable=too-few-public-methods
    """Represents either a vectorized leaf or a vectorized sum node in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    inputs: List["PartitionNode"]
    outputs: List["PartitionNode"]

    def __init__(self, scope: Iterable[int], replica_idx: int = 0) -> None:
        """Construct the node.

        Args:
            scope (Iterable[int]): The scope of this node.
            replica_idx: (int): The replica index associated to this node.
             By default, every region node will be of the same replica, i.e., with index 0.
        """
        super().__init__(scope)
        self.replica_idx = replica_idx

    def __lt__(self, other: RGNode) -> bool:
        """Compare the node with the other. A region is always smaller."""
        return isinstance(other, PartitionNode) or super().__lt__(other)


class PartitionNode(RGNode):  # pylint: disable=too-few-public-methods
    """Represents a (cross-)product in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    inputs: List[RegionNode]
    outputs: List[RegionNode]

    def __init__(self, scope: Iterable[int]) -> None:  # pylint: disable=useless-parent-delegation
        """Construct the node.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        super().__init__(scope)

    def __lt__(self, other: RGNode) -> bool:
        """Compare the node with the other. A partition is always larger."""
        return not isinstance(other, RegionNode) and super().__lt__(other)
