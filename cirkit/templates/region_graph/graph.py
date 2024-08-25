import itertools
import json
from abc import ABC
from functools import cached_property
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast, final

import numpy as np
from numpy.typing import NDArray

from cirkit.utils.algorithms import DiAcyclicGraph
from cirkit.utils.scope import Scope


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
class RegionGraph(DiAcyclicGraph):
    def __init__(
        self,
        nodes: List[RegionGraphNode],
        in_nodes: Dict[RegionGraphNode, List[RegionGraphNode]],
        outputs: List[RegionGraphNode],
    ) -> None:
        for node, node_children in in_nodes.items():
            if isinstance(node, RegionNode):
                for ptn in node_children:
                    if not isinstance(ptn, PartitionNode):
                        raise ValueError(
                            f"Expected partition node as children of '{node}', but found '{ptn}'"
                        )
                continue
            if not isinstance(node, PartitionNode):
                raise ValueError(
                    f"Region graph nodes must be either partition nodes or region nodes, found '{type(node)}'"
                )
            for rgn in node_children:
                if not isinstance(rgn, RegionNode):
                    raise ValueError(
                        f"Expected region node as children of '{node}', but found '{rgn}'"
                    )
        super().__init__(nodes, in_nodes, outputs)

    @cached_property
    def scope(self) -> Scope:
        return Scope.union(*(node.scope for node in self.outputs))

    @cached_property
    def num_variables(self) -> int:
        return len(self.scope)

    @cached_property
    def is_smooth(self) -> bool:
        return all(
            partition.scope == region.scope
            for region in self.inner_region_nodes
            for partition in region.inputs
        )

    @cached_property
    def is_decomposable(self) -> bool:
        return not any(
            region1.scope & region2.scope
            for partition in self.partition_nodes
            for region1, region2 in itertools.combinations(partition.inputs, 2)
        )

    @cached_property
    def is_structured_decomposable(self) -> bool:
        if not self.is_smooth:
            return False
        if not self.is_decomposable:
            return False
        is_structured_decomposable = True
        decompositions: Dict[Scope, Tuple[Scope, ...]] = {}
        for partition in self.partition_nodes:
            # The scopes are sorted by _sort_nodes(), so the tuple has a deterministic order.
            decomp = tuple(region.scope for region in partition.inputs)
            if partition.scope not in decompositions:
                decompositions[partition.scope] = decomp
            is_structured_decomposable &= decomp == decompositions[partition.scope]

    @cached_property
    def is_omni_compatible(self) -> bool:
        if not self.is_smooth:
            return False
        if not self.is_decomposable:
            return False
        return all(
            len(region.scope) == 1
            for partition in self.partition_nodes
            for region in partition.inputs
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
        if not (
            self.is_smooth and self.is_decomposable and other.is_smooth and other.is_decomposable
        ):  # Compatiblility first requires smoothness and decomposability.
            return False

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
        return (node for node in self.region_nodes if self.node_inputs(node))

    ####################################    (De)Serialization    ###################################
    # The RG can be dumped and loaded from json files, which can be useful when we want to save and
    # share it. The load() is another way to construct a RG other than the RG algorithms.

    # def dump(self, filename: str, with_meta: bool = True) -> None:
    #     """Dump the region graph to the json file.
    #
    #     The file will be opened with mode="w" and encoding="utf-8".
    #
    #     Args:
    #         filename (str): The file name for dumping.
    #         with_meta (bool, optional): Whether to include metadata of RGNode, set to False to \
    #             save some space while risking loss of information. Defaults to True.
    #     """
    #     # NOTE: Below we don't assume the existence of RGNode.metadata["sort_key"], and try to
    #     #       preserve the ordering by file structure. However, the sort_key will be saved when
    #     #       available and with_meta enabled.
    #
    #     # ANNOTATE: Specify content for empty container.
    #     rg_json: RegionGraphJson = {"regions": {}, "graph": []}
    #
    #     region_idx = {node: idx for idx, node in enumerate(self.region_nodes)}
    #
    #     # "regions" keeps ordered by the index, corresponding to the self.region_nodes order.
    #     rg_json["regions"] = (
    #         {
    #             str(idx): {"scope": list(node.scope), "metadata": node.metadata}
    #             for node, idx in region_idx.items()
    #         }
    #         if with_meta
    #         else {str(idx): list(node.scope) for node, idx in region_idx.items()}
    #     )
    #
    #     # "graph" keeps ordered by the list, corresponding to the self.partition_nodes order.
    #     for partition in self.partition_nodes:
    #         input_idxs = [region_idx[region_in] for region_in in partition.inputs]
    #         # partition.outputs is guaranteed to have len==1 by _validate().
    #         output_idx = region_idx[next(iter(partition.outputs))]
    #         rg_json["graph"].append({"inputs": input_idxs, "output": output_idx})
    #
    #     if with_meta:
    #         for partition, part_dict in zip(self.partition_nodes, rg_json["graph"]):
    #             part_dict["metadata"] = partition.metadata
    #
    #     # TODO: logging for dumping graph_json?
    #     with open(filename, "w", encoding="utf-8") as f:
    #         json.dump(rg_json, f)

    # @staticmethod
    # def load(filename: str) -> "RegionGraph":
    #     """Load the region graph from the json file.
    #
    #     The file will be opened with mode="r" and encoding="utf-8".
    #
    #     Metadata is always loaded when available.
    #
    #     Args:
    #         filename (str): The file name for loading.
    #
    #     Returns:
    #         RegionGraph: The loaded region graph.
    #     """
    #     # NOTE: Below we don't assume the existence of RGNode.metadata["sort_key"], and try to
    #     #       recover the ordering from file structure. However, the sort_key will be used when
    #     #       available.
    #
    #     with open(filename, "r", encoding="utf-8") as f:
    #         # ANNOTATE: json.load gives Any.
    #         rg_json: RegionGraphJson = json.load(f)
    #
    #     graph = RegionGraph()
    #     # ANNOTATE: Specify content for empty container.
    #     idx_region: Dict[int, RegionNode] = {}
    #
    #     # Iterate regions by the order of idx so that the order of graph.region_nodes is recovered.
    #     # NOTE: By json standard, rg_json["regions"] has no guaranteed order.
    #     for idx in range(len(rg_json["regions"])):
    #         dict_or_scope = rg_json["regions"][str(idx)]
    #         if isinstance(dict_or_scope, dict):
    #             region_node = RegionNode(dict_or_scope["scope"])
    #             graph.add_node(region_node, metadata=dict_or_scope.get("metadata", {}))
    #         else:
    #             region_node = RegionNode(dict_or_scope)
    #             graph.add_node(region_node)
    #         idx_region[idx] = region_node
    #
    #     # Iterate partitions by the order of list so that the order of graph.partition_nodes is
    #     # recovered.
    #     for partition in rg_json["graph"]:
    #         regions_in = [idx_region[idx_in] for idx_in in partition["inputs"]]
    #         region_out = idx_region[partition["output"]]
    #         graph.add_partitioning(region_out, regions_in, metadata=partition.get("metadata", {}))
    #
    #     return graph.freeze()
