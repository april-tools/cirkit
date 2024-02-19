import itertools
import json
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple, cast, final, overload
from typing_extensions import Self  # FUTURE: in typing from 3.11

import numpy as np
from numpy.typing import NDArray

from cirkit.new.region_graph.rg_node import PartitionNode, RegionNode, RGNode
from cirkit.new.utils import OrderedSet, Scope
from cirkit.new.utils.type_aliases import RegionGraphJson, RGNodeMetadata


# We mark RG as final to hint that RG algorithms should not be its subclasses but factories, so that
# constructed RGs and loaded RGs are all of type RegionGraph.
@final
class RegionGraph:
    """The region graph that holds the high-level abstraction of circuit structure.

    This class is initiated empty, and RG construction algorithms decides how to push nodes and \
    edges into the graph.

    After construction, the graph must be freezed before being used, so that some finalization \
    work for construction can be done properly.
    """

    def __init__(self) -> None:
        """Init class.

        The graph is empty upon creation.
        """
        # This node container will not be visible to the user. Instead, node views are provided for
        # read-only access to an iterable of nodes.
        # ANNOTATE: Specify content for empty container.
        self._nodes: OrderedSet[RGNode] = OrderedSet()

        # It's on purpose that some attributes are defined outside __init__ but in freeze().

    @property
    def _is_frozen(self) -> bool:
        """Whether freeze() has been called on this graph."""
        # self.scope is not set in __init__ and will be set in freeze().
        return hasattr(self, "scope")

    # TODO: __repr__?

    ######################################    Construction    ######################################
    # The RG is initiated empty, and the following routines are used to populate the RG after that.

    def add_node(self, node: RGNode, *, metadata: Optional[RGNodeMetadata] = None) -> None:
        """Add a node to the graph.

        If the node is already present, this is no-op.

        Args:
            node (RGNode): The node to add.
            metadata (Optional[RGNodeMetadata], optional): The metadata of the node, will be \
                updated, not replaced, on to the node. Defaults to None.
        """
        assert not self._is_frozen, "The RG should not be modified after calling freeze()."
        if metadata is not None:
            node.metadata.update(metadata)
        self._nodes.append(node)

    @overload
    def add_edge(self, tail: RegionNode, head: PartitionNode) -> None:
        ...

    @overload
    def add_edge(self, tail: PartitionNode, head: RegionNode) -> None:
        ...

    def add_edge(self, tail: RGNode, head: RGNode) -> None:
        """Add a directed edge to the graph.

        If the nodes are not present yet, they'll be automatically added (with no metadata). If \
        metadata needs to be specified, manually call add_node or directly assign instead.

        Args:
            tail (RGNode): The tail of the edge (from).
            head (RGNode): The head of the edge (to).
        """
        # add_node will check for _is_frozen.
        self.add_node(tail)
        self.add_node(head)
        assert tail.outputs.append(head), "The edges in RG should not be repeated."
        head.inputs.append(tail)  # Only need to check duplicate in one direction.

    def add_partitioning(
        self,
        region: RegionNode,
        sub_regions: Iterable[RegionNode],
        *,
        metadata: Optional[RGNodeMetadata] = None,
    ) -> None:
        """Add a partitioning structure to the graph, with a PartitionNode constructed internally.

        Args:
            region (RegionNode): The region to be partitioned.
            sub_regions (Iterable[RegionNode]): The partitioned sub-regions.
            metadata (Optional[RGNodeMetadata], optional): The metadata of the internally \
                constructed PartitionNode. Defaults to None.
        """
        partition = PartitionNode(region.scope)
        if metadata is not None:
            partition.metadata.update(metadata)  # Use update() even when empty, to avoid cross-ref.
        self.add_edge(partition, region)
        for sub_region in sub_regions:
            self.add_edge(sub_region, partition)

    ########################################    Freezing    ########################################
    # After construction, the RG should be validated and its properties will be calculated. The RG
    # should not be modified after being frozen.

    def freeze(self) -> Self:
        """Freeze the RG to mark the end of construction and return self.

        The work here includes:
            - Finalizing the maintenance on internal data structures;
            - Validating the RG structure;
            - Assigning public attributes/properties.

        Returns:
            Self: The self object.
        """
        self._sort_nodes()
        # TODO: include repr(self) in error msg?
        assert not (reason := self._validate()), f"Illegal RG structure: {reason}."
        self._set_properties()
        return self

    def _sort_nodes(self) -> None:
        """Sort the OrderedSet of RGNode for node list and edge tables."""
        # Now rg nodes have no sort_key. With a stable sort, equal nodes keep insertion order.
        self._nodes.sort()
        # Now the nodes are in an order determined solely by the construction algorithm.
        for i, node in enumerate(self._nodes):
            node.metadata["sort_key"] = i
        # Now the nodes have total ordering based on the original order.
        for node in self._nodes:
            node.inputs.sort()
            node.outputs.sort()
        # Now all containers are consistently sorted by the order decided by sort_key.

    # TODO: do we need these return? or just assert?
    # pylint: disable-next=too-many-return-statements
    def _validate(self) -> str:
        """Validate the RG structure to make sure it's a legal computational graph.

        Returns:
            str: The reason for error (NOTE: without period), empty for nothing wrong.
        """
        # These two conditions are also quick checks for DAG.
        if next(self.input_nodes, None) is None:
            return "RG must have at least one input node"
        if next(self.output_nodes, None) is None:
            return "RG must have at least one output node"

        # Also guarantees the input/output nodes are all regions.
        if not all(partition.inputs for partition in self.partition_nodes):
            return "PartitionNode must have at least one input"
        if any(len(partition.outputs) != 1 for partition in self.partition_nodes):
            return "PartitionNode can only have one output RegionNode"

        if any(
            Scope.union(*(node_input.scope for node_input in node.inputs)) != node.scope
            for node in self.inner_nodes
        ):
            return "The scope of an inner node should be the union of scopes of its inputs"

        if not self._check_dag():  # It's a bit complex, so extracted as a standalone method.
            return "RG must be a DAG"

        # TODO: Anything else needed?
        return ""

    def _check_dag(self) -> bool:
        """Check whether the graph is a DAG.

        Returns:
            bool: Whether a DAG.
        """
        # ANNOTATE: Specify content for empty container.
        visited: Set[RGNode] = set()  # Visited nodes during all DFS runs.
        path: Set[RGNode] = set()  # Path stack for the current DFS run.
        # Here we don't care about order and there's no duplicate, so set is used for fast in check.

        def _dfs(node: RGNode) -> bool:
            """Traverse and check for cycle from node.

            Args:
                node (RGNode): The node to start with.

            Returns:
                bool: Whether it's OK (not cyclic).
            """
            visited.add(node)
            path.add(node)  # If OK, we need to pop node out, otherwise just propagate failure.
            for next_node in node.outputs:
                if next_node in path:  # Loop to the current path, including next_node==node.
                    return False
                if next_node in visited:  # Already checked and is OK.
                    path.remove(node)
                    return True
                if not _dfs(next_node):  # Found problem in DFS.
                    return False
            path.remove(node)
            return True  # Nothing wrong in the current DFS run.

        # If visited, shortcut to True, otherwise run DFS from node.
        return all(node in visited or _dfs(node) for node in self.nodes)

    def _set_properties(self) -> None:
        """Set the attributes for RG properties in self.

        Names set here are not valid in self before calling this method.
        """
        # It's intended to assign these attributes outside __init__. Without calling into freeze(),
        # these attrs, especially self.scope and self.num_vars, will be undefined, and therefore
        # blocks downstream usage. Thus freeze() will be enforced to run before using the RG.

        # Guaranteed to be non-empty by _validate().
        self.scope = Scope.union(*(node.scope for node in self.output_nodes))
        self.num_vars = len(self.scope)

        self.is_smooth = all(
            partition.scope == region.scope
            for region in self.inner_region_nodes
            for partition in region.inputs
        )

        # Union of input scopes is guaranteed to be the node scope by _validate().
        self.is_decomposable = not any(
            region1.scope & region2.scope
            for partition in self.partition_nodes
            for region1, region2 in itertools.combinations(partition.inputs, 2)
        )

        # TODO: is this correct for more-than-2 partition?
        # Structured-decomposablity first requires smoothness and decomposability.
        self.is_structured_decomposable = self.is_smooth and self.is_decomposable
        # ANNOTATE: Specify content for empty container.
        decompositions: Dict[Scope, Tuple[Scope, ...]] = {}
        for partition in self.partition_nodes:
            # The scopes are sorted by _sort_nodes(), so the tuple has a deterministic order.
            decomp = tuple(region.scope for region in partition.inputs)
            if partition.scope not in decompositions:
                decompositions[partition.scope] = decomp
            self.is_structured_decomposable &= decomp == decompositions[partition.scope]

        # Omni-compatiblility first requires smoothness and decomposability, and then it's a
        # necessary and sufficient condition that all partitions decompose into univariate regions.
        self.is_omni_compatible = (
            self.is_smooth
            and self.is_decomposable
            and all(
                len(region.scope) == 1
                for partition in self.partition_nodes
                for region in partition.inputs
            )
        )

    #######################################    Properties    #######################################
    # Here are the basic properties and some structural properties of the RG. Some of them are
    # static and defined in the _set_properties() when the RG is freezed. Some requires further
    # information and is defined below to be calculated on the fly. We list everything here to add
    # "docstrings" to them, but note that they're not valid before freeze().

    scope: Scope
    """The scope of the RG, i.e., the union of scopes of all output units."""

    num_vars: int
    """The number of variables referenced in the RG, i.e., the size of scope."""

    is_smooth: bool
    """Whether the RG is smooth, i.e., all inputs to a region have the same scope."""

    is_decomposable: bool
    """Whether the RG is decomposable, i.e., inputs to a partition have disjoint scopes."""

    is_structured_decomposable: bool
    """Whether the RG is structured-decomposable, i.e., compatible to itself."""

    is_omni_compatible: bool
    """Whether the RG is omni-compatible, i.e., compatible to all circuits of the same scope."""

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

    #######################################    Node views    #######################################
    # These are iterable views of the nodes in the RG, and the topological order is guaranteed (by a
    # stronger ordering). For efficiency, all these views are iterators (a container iter or a
    # generator), so that they can be chained without instantiating intermediate containers.
    # NOTE: The views are even available when the graph is only partially constructed, but without
    #       freeze() there's no ordering graranteed for these views.

    @property
    def nodes(self) -> Iterator[RGNode]:
        """All nodes in the graph."""
        return iter(self._nodes)

    @property
    def region_nodes(self) -> Iterator[RegionNode]:
        """Region nodes in the graph."""
        return (node for node in self.nodes if isinstance(node, RegionNode))

    @property
    def partition_nodes(self) -> Iterator[PartitionNode]:
        """Partition nodes in the graph, which are always inner nodes."""
        return (node for node in self.nodes if isinstance(node, PartitionNode))

    @property
    def input_nodes(self) -> Iterator[RegionNode]:
        """Input nodes of the graph, which are always regions."""
        return (node for node in self.region_nodes if not node.inputs)

    @property
    def output_nodes(self) -> Iterator[RegionNode]:
        """Output nodes of the graph, which are always regions."""
        return (node for node in self.region_nodes if not node.outputs)

    @property
    def inner_nodes(self) -> Iterator[RGNode]:
        """Inner (non-input) nodes in the graph."""
        return (node for node in self.nodes if node.inputs)

    @property
    def inner_region_nodes(self) -> Iterator[RegionNode]:
        """Inner region nodes in the graph."""
        return (node for node in self.region_nodes if node.inputs)

    ####################################    (De)Serialization    ###################################
    # The RG can be dumped and loaded from json files, which can be useful when we want to save and
    # share it. The load() is another way to construct a RG other than the RG algorithms.

    def dump(self, filename: str, with_meta: bool = True) -> None:
        """Dump the region graph to the json file.

        The file will be opened with mode="w" and encoding="utf-8".

        Args:
            filename (str): The file name for dumping.
            with_meta (bool, optional): Whether to include metadata of RGNode, set to False to \
                save some space while risking loss of information. Defaults to True.
        """
        # NOTE: Below we don't assume the existence of RGNode.metadata["sort_key"], and try to
        #       preserve the ordering by file structure. However, the sort_key will be saved when
        #       available and with_meta enabled.

        # ANNOTATE: Specify content for empty container.
        rg_json: RegionGraphJson = {"regions": {}, "graph": []}

        region_idx = {node: idx for idx, node in enumerate(self.region_nodes)}

        # "regions" keeps ordered by the index, corresponding to the self.region_nodes order.
        rg_json["regions"] = (
            {
                str(idx): {"scope": list(node.scope), "metadata": node.metadata}
                for node, idx in region_idx.items()
            }
            if with_meta
            else {str(idx): list(node.scope) for node, idx in region_idx.items()}
        )

        # "graph" keeps ordered by the list, corresponding to the self.partition_nodes order.
        for partition in self.partition_nodes:
            input_idxs = [region_idx[region_in] for region_in in partition.inputs]
            # partition.outputs is guaranteed to have len==1 by _validate().
            output_idx = region_idx[next(iter(partition.outputs))]
            rg_json["graph"].append({"inputs": input_idxs, "output": output_idx})

        if with_meta:
            for partition, part_dict in zip(self.partition_nodes, rg_json["graph"]):
                part_dict["metadata"] = partition.metadata

        # TODO: logging for dumping graph_json?
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(rg_json, f)

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

        graph = RegionGraph()
        # ANNOTATE: Specify content for empty container.
        idx_region: Dict[int, RegionNode] = {}

        # Iterate regions by the order of idx so that the order of graph.region_nodes is recovered.
        # NOTE: By json standard, rg_json["regions"] has no guaranteed order.
        for idx in range(len(rg_json["regions"])):
            dict_or_scope = rg_json["regions"][str(idx)]
            if isinstance(dict_or_scope, dict):
                region_node = RegionNode(dict_or_scope["scope"])
                graph.add_node(region_node, metadata=dict_or_scope.get("metadata", {}))
            else:
                region_node = RegionNode(dict_or_scope)
                graph.add_node(region_node)
            idx_region[idx] = region_node

        # Iterate partitions by the order of list so that the order of graph.partition_nodes is
        # recovered.
        for partition in rg_json["graph"]:
            regions_in = [idx_region[idx_in] for idx_in in partition["inputs"]]
            region_out = idx_region[partition["output"]]
            graph.add_partitioning(region_out, regions_in, metadata=partition.get("metadata", {}))

        return graph.freeze()
