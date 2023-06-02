import itertools
from dataclasses import dataclass
from typing import Iterable

# TODO: rework docstrings


@dataclass
class _EiNetAddress:
    """Address of a PC node to its EiNet implementation.

    In EiNets, each layer implements a tensor of log-densities of shape
        (batch_size, vector_length, num_nodes)
    All DistributionVector's, which are either vectors of leaf distributions (exponential families)
    or vectors of sum nodes, uniquely correspond to some slice of the log-density tensor of some
    layer, where we slice the last axis.

    EiNetAddress stores the "address" of the implementation in the EinsumNetwork.

    :param layer: which layer implements this node?
    :param idx: which index does the node have in the layers log-density tensor?
    :param replica_idx: this is solely for the input layer -- see ExponentialFamilyArray and
    FactorizedLeafLayer.
    These two layers implement all leaves in parallel. To this end we need "enough leaves",
    which is achieved to make a sufficiently large "block" of input distributions.
    The replica_idx indicates in which slice of the ExponentialFamilyArray a leaf is
    represented.
    """

    # TODO: this is for einet. why in RG?
    layer: object = 0  # TODO: this is of course not good
    idx: int = 0
    replica_idx: int = 0


class RGNode:
    """The base class for nodes in region graphs."""

    # TODO: id for base class or children? only used for sort? really need?
    _id_counter = itertools.count(0)
    """we assign each object a unique id."""

    def __init__(self, scope: Iterable[int]) -> None:
        """Construct the node.

        Args:
            scope (Iterable[int]): The scope of this node.
        """
        self.scope = set(scope)

        node_id = next(RGNode._id_counter)
        self._sort_key = (tuple(self.scope), node_id)

    def __repr__(self) -> str:
        """Generate the `repr` string of the node."""
        return self.__class__.__name__ + ": " + repr(self.scope)

    def __lt__(self, other: "RGNode") -> bool:  # we don't use Self as it can compare to RGNode
        """Compare the node with the other, for sorting."""
        # TODO: it's better to be abstract. but do we really need this?
        return self._sort_key < other._sort_key


# TODO: do we need to enforce num public methods?
class RegionNode(RGNode):  # pylint: disable=too-few-public-methods
    """Represents either a vectorized leaf or a vectorized sum node in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    def __init__(self, scope: Iterable[int]) -> None:
        """Is a docstring."""  # TODO: how to avoid rewrite docstring?
        super().__init__(scope)
        self.num_dist: int = 0  # TODO: number of distributions???
        self.einet_address = _EiNetAddress()

    def __lt__(self, other: RGNode) -> bool:
        """Compare the node with the other. A region is always smaller."""
        return isinstance(other, PartitionNode) or super().__lt__(other)


class PartitionNode(RGNode):  # pylint: disable=too-few-public-methods
    """Represents a (cross-)product in the PC.

    To construct a PC, we simply use the DiGraph (directed graph) class of networkx.
    """

    def __lt__(self, other: RGNode) -> bool:
        """Compare the node with the other. A partition is always larger."""
        return not isinstance(other, RegionNode) and super().__lt__(other)
