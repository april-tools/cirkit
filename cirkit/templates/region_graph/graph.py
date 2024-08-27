import itertools
import json
from abc import ABC
from collections import defaultdict
from functools import cached_property
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypedDict, Union, cast, final
from typing_extensions import TypeAlias

import numpy as np
from numpy.typing import NDArray

from cirkit.utils.algorithms import DiAcyclicGraph
from cirkit.utils.scope import Scope

RGNodeMetadata: TypeAlias = Dict[str, Union[int, float, str, bool]]


class RegionDict(TypedDict):
    """The structure of a region node in the json file."""

    scope: List[int]  # The scope of this region node, specified by id of variable.


class PartitionDict(TypedDict):
    """The structure of a partition node in the json file."""

    inputs: List[int]  # The inputs of this partition node, specified by id of region node.
    output: int  # The output of this partition node, specified by id of region node.


class RegionGraphJson(TypedDict):
    """The structure of the region graph json file."""

    # The regions of RG represented by a mapping from id in str to either a dict or only the scope.
    regions: Dict[str, Union[RegionDict, List[int]]]

    # The list of region node roots str ids in the RG
    roots: List[str]

    # The graph of RG represented by a list of partitions.
    graph: List[PartitionDict]


class RegionGraphNode(ABC):
    """The abstract base class for nodes in region graphs."""

    def __init__(self, scope: Union[Iterable[int], Scope]) -> None:
        """Init class.

        Args:
            scope (Scope): The scope of this node.
        """
        if not scope:
            raise ValueError("The scope of a region graph node must not be empty.")
        super().__init__()
        self.scope = Scope(scope)

    def __repr__(self) -> str:
        """Generate the repr string of the node.

        Returns:
            str: The str representation of the node.
        """
        return f"{type(self).__name__}@0x{id(self):x}({self.scope})"


class RegionNode(RegionGraphNode):
    """The region node in the region graph."""

    ...


class PartitionNode(RegionGraphNode):
    """The partition node in the region graph."""

    ...


# We mark RG as final to hint that RG algorithms should not be its subclasses but factories, so that
# constructed RGs and loaded RGs are all of type RegionGraph.
@final
class RegionGraph(DiAcyclicGraph[RegionGraphNode]):
    def __init__(
        self,
        nodes: List[RegionGraphNode],
        in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]],
        outputs: List[RegionGraphNode],
    ) -> None:
        super().__init__(nodes, in_nodes, outputs)
        self._check_structure()

    def _check_structure(self):
        for node, node_children in self.nodes_inputs.items():
            if isinstance(node, RegionNode):
                for ptn in node_children:
                    if not isinstance(ptn, PartitionNode):
                        raise ValueError(
                            f"Expected partition node as children of '{node}', but found '{ptn}'"
                        )
                    if ptn.scope != node.scope:
                        raise ValueError(
                            f"Expectet partition node with scope '{node.scope}', but found '{ptn.scope}"
                        )
                continue
            if not isinstance(node, PartitionNode):
                raise ValueError(
                    f"Region graph nodes must be either partition nodes or region nodes, found '{type(node)}'"
                )
            scopes = []
            for rgn in node_children:
                if not isinstance(rgn, RegionNode):
                    raise ValueError(
                        f"Expected region node as children of '{node}', but found '{rgn}'"
                    )
                scopes.append(rgn.scope)
            scope = Scope.union(*scopes)
            if scope != node.scope or sum(len(sc) for sc in scopes) != len(scope):
                raise ValueError(
                    f"Expecte partitioning of scope '{node.scope}', but found '{scopes}'"
                )
        for ptn in self.partition_nodes:
            rgn_outs = self.node_outputs(ptn)
            if len(rgn_outs) != 1:
                raise ValueError(
                    f"Expected each partition node to have exactly one parent region node,"
                    f" but found {len(rgn_outs)} parent nodes"
                )

    def region_inputs(self, rgn: RegionNode) -> List[PartitionNode]:
        return [cast(PartitionNode, node) for node in self.node_inputs(rgn)]

    def partition_inputs(self, ptn: PartitionNode) -> List[RegionNode]:
        return [cast(RegionNode, node) for node in self.node_inputs(ptn)]

    def region_outputs(self, rgn: RegionNode) -> List[PartitionNode]:
        return [cast(PartitionNode, node) for node in self.node_outputs(rgn)]

    def partition_outputs(self, ptn: PartitionNode) -> List[RegionNode]:
        return [cast(RegionNode, node) for node in self.node_outputs(ptn)]

    @property
    def inputs(self) -> Iterator[RegionNode]:
        return (cast(RegionNode, node) for node in super().inputs)

    @property
    def outputs(self) -> Iterator[RegionNode]:
        return (cast(RegionNode, node) for node in super().outputs)

    @property
    def region_nodes(self) -> Iterator[RegionNode]:
        """Region nodes in the graph."""
        return (node for node in self.nodes if isinstance(node, RegionNode))

    @property
    def partition_nodes(self) -> Iterator[PartitionNode]:
        """Partition nodes in the graph, which are always inner nodes."""
        return (node for node in self.nodes if isinstance(node, PartitionNode))

    @property
    def inner_nodes(self) -> Iterator[RegionGraphNode]:
        """Inner (non-input) nodes in the graph."""
        return (node for node in self.nodes if self.node_inputs(node))

    @property
    def inner_region_nodes(self) -> Iterator[RegionNode]:
        """Inner region nodes in the graph."""
        return (
            node for node in self.region_nodes if self.node_inputs(node) and self.node_outputs(node)
        )

    @cached_property
    def scope(self) -> Scope:
        return Scope.union(*(node.scope for node in self.outputs))

    @cached_property
    def num_variables(self) -> int:
        return len(self.scope)

    @cached_property
    def is_structured_decomposable(self) -> bool:
        is_structured_decomposable = True
        decompositions: Dict[Scope, Tuple[Scope, ...]] = {}
        for partition in self.partition_nodes:
            # The scopes are sorted by _sort_nodes(), so the tuple has a deterministic order.
            decomp = tuple(region.scope for region in self.node_inputs(partition))
            if partition.scope not in decompositions:
                decompositions[partition.scope] = decomp
            is_structured_decomposable &= decomp == decompositions[partition.scope]
        return is_structured_decomposable

    @cached_property
    def is_omni_compatible(self) -> bool:
        return all(
            len(region.scope) == 1
            for partition in self.partition_nodes
            for region in self.node_inputs(partition)
        )

    def is_compatible(
        self, other: "RegionGraph", /, *, scope: Optional[Iterable[int]] = None
    ) -> bool:
        """Test compatibility with another region graph over the given scope.

        Args:
            other (RegionGraph): The other region graph to compare with.
            scope (Optional[Iterable[int]], optional): The scope over which to check. If None, \
                will use the intersection of the scopes of the two RG. Defaults to None.

        Returns:
            bool: Whether self is compatible to other.
        """
        # _is_frozen is implicitly tested because is_smooth is set in freeze().
        scope = Scope(scope) if scope is not None else self.scope & other.scope

        # TODO: is this correct for more-than-2 partition?
        for partition1, partition2 in itertools.product(
            self.partition_nodes, other.partition_nodes
        ):
            if partition1.scope & scope != partition2.scope & scope:
                continue  # Only check partitions with the same scope.

            if any(partition1.scope <= input.scope for input in partition2.inputs) or any(
                partition2.scope <= input.scope for input in partition1.inputs
            ):
                continue  # Only check partitions not within another partition.

            adj_mat = np.zeros((len(partition1.inputs), len(partition2.inputs)), dtype=np.bool_)
            for (i, region1), (j, region2) in itertools.product(
                enumerate(partition1.inputs), enumerate(partition2.inputs)
            ):
                # I.e., if scopes intersect over the scope to test.
                adj_mat[i, j] = bool(region1.scope & region2.scope & scope)
            adj_mat = adj_mat @ adj_mat.T
            # Now we have adjencency from inputs1 (of self) to inputs1. An edge means the two
            # regions must be partitioned together.

            # The number of zero eigen values for the laplacian matrix is the number of connected
            # components in a graph.
            # ANNOTATE: Numpy has typing issues.
            # CAST: Numpy has typing issues.
            deg_mat = np.diag(cast(NDArray[np.int64], adj_mat.sum(axis=1)))
            laplacian: NDArray[np.int64] = deg_mat - adj_mat
            eigen_values = np.linalg.eigvals(laplacian)
            num_connected: int = cast(NDArray[np.int64], np.isclose(eigen_values, 0).sum()).item()
            if num_connected == 1:  # All connected, meaning there's no valid partitioning.
                return False

        return True

    ####################################    (De)Serialization    ###################################
    # The RG can be dumped and loaded from json files, which can be useful when we want to save and
    # share it. The load() is another way to construct a RG other than the RG algorithms.

    def dump(self, filename: str) -> None:
        """Dump the region graph to the json file.

        The file will be opened with mode="w" and encoding="utf-8".

        Args:
            filename (str): The file name for dumping.
        """
        # NOTE: Below we don't assume the existence of RGNode.metadata["sort_key"], and try to
        #       preserve the ordering by file structure. However, the sort_key will be saved when
        #       available and with_meta enabled.

        # ANNOTATE: Specify content for empty container.
        rg_json: RegionGraphJson = {}
        region_idx: Dict[RegionNode, int] = {
            node: idx for idx, node in enumerate(self.region_nodes)
        }

        # Store the region nodes information
        rg_json["regions"] = {str(idx): list(node.scope) for node, idx in region_idx.items()}

        # Store the roots information
        rg_json["roots"] = [str(region_idx[rgn]) for rgn in self.outputs]

        # Store the partition nodes information
        rg_json["graph"] = []
        for partition in self.partition_nodes:
            input_idxs = [region_idx[cast(RegionNode, rgn)] for rgn in self.node_inputs(partition)]
            # partition.outputs is guaranteed to have len==1 by _validate().
            output_idx = region_idx[cast(RegionNode, self.node_outputs(partition)[0])]
            rg_json["graph"].append({"output": output_idx, "inputs": input_idxs})

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(rg_json, f, indent=4)

    @staticmethod
    def load(filename: str) -> "RegionGraph":
        """Load the region graph from the json file.

        The file will be opened with mode="r" and encoding="utf-8".

        Metadata is always loaded when available.

        Args:
            filename (str): The file name for loading.

        Returns:
            RegionGraph: The loaded region graph.
        """
        # NOTE: Below we don't assume the existence of RGNode.metadata["sort_key"], and try to
        #       recover the ordering from file structure. However, the sort_key will be used when
        #       available.

        with open(filename, "r", encoding="utf-8") as f:
            # ANNOTATE: json.load gives Any.
            rg_json: RegionGraphJson = json.load(f)

        nodes: List[RegionGraphNode] = []
        in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]] = defaultdict(list)
        outputs = []
        region_idx: Dict[int, RegionNode] = {}

        # Load the region nodes
        for idx, rgn_scope in rg_json["regions"].items():
            rgn = RegionNode(rgn_scope)
            nodes.append(rgn)
            region_idx[int(idx)] = rgn

        # Load the root region nodes
        for idx in rg_json["roots"]:
            outputs.append(region_idx[int(idx)])

        # Load the partition nodes
        for partitioning in rg_json["graph"]:
            in_rgns = [region_idx[int(idx)] for idx in partitioning["inputs"]]
            out_rgn = region_idx[partitioning["output"]]
            ptn = PartitionNode(out_rgn.scope)
            nodes.append(ptn)
            in_nodes[out_rgn].append(ptn)
            in_nodes[ptn] = in_rgns

        return RegionGraph(nodes, in_nodes, outputs=outputs)
