import itertools
import json
from typing import Dict, FrozenSet, Iterable, Iterator, Optional, Set, cast, final, overload
from typing_extensions import Self  # TODO: in typing from 3.11

import numpy as np
from numpy.typing import NDArray

from cirkit.new.region_graph.rg_node import PartitionNode, RegionNode, RGNode
from cirkit.new.utils.type_aliases import RegionGraphJson


# We mark RG as final to hint that RG algorithms should not be its subclasses but factories.
# Disable: It's designed to have these many attributes.
@final
class RegionGraph:  # pylint: disable=too-many-instance-attributes
    """The region graph that holds the high-level abstraction of circuit structure.
    
    This class is initiated empty and nodes can be pushed into the graph with edges. It can also \
    serve as a container of RGNode for use in the RG construction algorithms.
    """

    def __init__(self) -> None:
        """Init class."""
        super().__init__()
        # The nodes container will not be visible to the user. Instead, node views are provided for
        # read-only access to an iterable of nodes.
        self._nodes: Set[RGNode] = set()
        self._frozen = False

    # TODO: __repr__?

    ######################################    Construction    ######################################
    # The RG is initiated empty, and the following routines are used to populate the RG after that.

    def add_node(self, node: RGNode) -> None:
        """Add a node to the graph.

        If the node is already present, this is no-op.

        Args:
            node (RGNode): The node to add.
        """
        assert not self._frozen, "The RG should not be modified after calling freeze()."
        self._nodes.add(node)

    @overload
    def add_edge(self, tail: RegionNode, head: PartitionNode) -> None:
        ...

    @overload
    def add_edge(self, tail: PartitionNode, head: RegionNode) -> None:
        ...

    def add_edge(self, tail: RGNode, head: RGNode) -> None:
        """Add a directed edge to the graph.

        If the nodes are not present, they'll be automatically added.

        Args:
            tail (RGNode): The tail of the edge (from).
            head (RGNode): The head of the edge (to).
        """
        # add_node will check for _frozen.
        self.add_node(tail)
        self.add_node(head)
        tail.outputs.append(head)
        head.inputs.append(tail)

    def add_partitioning(self, region: RegionNode, sub_regions: Iterable[RegionNode]) -> None:
        """Add a partitioning structure to the graph, with a PartitionNode constructed internally.

        Args:
            region (RegionNode): The region to be partitioned.
            sub_regions (Iterable[RegionNode]): The partitioned regions.
        """
        partition = PartitionNode(region.scope)
        self.add_edge(partition, region)
        for sub_region in sub_regions:
            self.add_edge(sub_region, partition)

    #######################################    Validation    #######################################
    # After construction, the RG should be validated and its properties will be calculated. The RG
    # should not be modified after being validated and frozen.

    def freeze(self) -> Self:
        """Freeze the RG to prevent further modifications.

        With a frozen RG, we also validate the RG structure and calculate its properties.

        For convenience, self is returned after freezing.

        Returns:
            Self: The self object.
        """
        self._frozen = True
        # TODO: print repr of self?
        assert not (reason := self._validate()), f"The RG structure is not valid: {reason}."
        self._calc_properties()
        return self

    # NOTE: The reason returned should not include a period.
    def _validate(self) -> str:
        """Validate the RG structure to make sure it's a legal computational graph.

        Returns:
            str: Empty if the RG is valid, otherwise the reason.
        """
        # These two if conditions are also quick checks for DAG.
        if next(self.input_nodes, None) is None:
            return "RG must have at least one input node"
        if next(self.output_nodes, None) is None:
            return "RG must have at least one output node"

        if any(len(partition.outputs) != 1 for partition in self.partition_nodes):
            return "PartitionNode can only have one output RegionNode"

        if not self._check_dag():
            return "RG must be a DAG"

        # TODO: Anything else needed?
        return ""

    # Checking DAG is a bit complex, so it's extracted as a standalone method.
    def _check_dag(self) -> bool:
        """Check if the RG is a DAG.

        Returns:
            bool: Whether the RG is a DAG.
        """
        visited: Set[RGNode] = set()  # Visited nodes during all DFS runs.
        path: Set[RGNode] = set()  # Path stack for the current DFS run.

        def _dfs(node: RGNode) -> bool:
            """Try to traverse and check for cycle from node.

            Args:
                node (RGNode): The node to start with.

            Returns:
                bool: Whether it's OK (not cyclic).
            """
            visited.add(node)
            path.add(node)
            for next_node in node.outputs:
                if next_node in path:  # Loop to the current path, including next_node==node.
                    return False
                if next_node in visited:  # Already checked and is OK.
                    return True
                if not _dfs(next_node):  # Found problem in DFS.
                    return False
            path.remove(node)
            return True  # Nothing wrong in the current DFS run.

        # If visited, shortcut to True, otherwise run DFS from node.
        return all(node in visited or _dfs(node) for node in self._nodes)

    def _calc_properties(self) -> None:
        """Calculate the properties of the RG and save them to self.

        These properties are not valid before calling this method.
        """
        # It's intended to assign these attributes outside __init__: without calling into freeze(),
        # these attrs, especially self.num_vars, will be undefined, and therefore blocks downstream
        # usage. Thus freeze() will be enforced to run before using RG.

        self.scope = frozenset().union(*(node.scope for node in self.output_nodes))
        self.num_vars = len(self.scope)

        self.is_smooth = all(
            all(partition.scope == region.scope for partition in region.inputs)
            for region in self.inner_region_nodes
        )

        self.is_decomposable = all(
            not any(
                region1.scope & region2.scope
                for region1, region2 in itertools.combinations(partition.inputs, 2)
            )
            and set().union(*(region.scope for region in partition.inputs)) == partition.scope
            for partition in self.partition_nodes
        )

        # Structured-decomposablity first requires smoothness and decomposability.
        self.is_structured_decomposable = self.is_smooth and self.is_decomposable
        decompositions: Dict[FrozenSet[int], Set[FrozenSet[int]]] = {}
        for partition in self.partition_nodes:
            decomp = set(region.scope for region in partition.inputs)
            if partition.scope not in decompositions:
                decompositions[partition.scope] = decomp
            self.is_structured_decomposable &= decomp == decompositions[partition.scope]

        # Omni-compatiblility first requires smoothness and decomposability.
        self.is_omni_compatible = self.is_smooth and self.is_decomposable
        # TODO: currently we don't have a good way to represent omni-compatible circuits.
        self.is_omni_compatible &= False

    #######################################    Properties    #######################################
    # Here are the basic properties and some structural properties of the RG. Some of them are
    # static and defined in the _calc_properties after the RG is freezed. Some requires further
    # information and is define below to be calculated on the fly.
    # We list everything here to add "docstrings" to them.

    scope: FrozenSet[int]
    """The scope of the RG, i.e., the union of scopes of all output units."""

    num_vars: int
    """The number of variables referenced in the RG, i.e., the size of scope."""

    is_smooth: bool
    """Whether the RG is smooth, i.e., all inputs to a region have the same scope."""

    is_decomposable: bool
    """Whether the RG is decomposable, i.e., inputs to a partition have disjoint scopes and their \
    union is the scope of the partition."""

    is_structured_decomposable: bool
    """Whether the RG is structured-decomposable, i.e., compatible to itself."""

    is_omni_compatible: bool
    """Whether the RG is omni-compatible, i.e., compatible to all circuits of the same scope."""

    def is_compatible(self, other: "RegionGraph", scope: Optional[Iterable[int]] = None) -> bool:
        """Test compatibility with another region graph over the given scope.

        Args:
            other (RegionGraph): The other region graph to compare with.
            scope (Optional[Iterable[int]], optional): The scope over which to check. If None, \
                will use the intersection of the scopes of two RG. Defaults to None.

        Returns:
            bool: Whether the RG is compatible to the other.
        """
        if not (
            self.is_smooth and self.is_decomposable and other.is_smooth and other.is_decomposable
        ):  # Compatiblility first requires smoothness and decomposability.
            return False

        scope = frozenset(scope) if scope is not None else self.scope & other.scope

        for partition1, partition2 in itertools.product(
            self.partition_nodes, other.partition_nodes
        ):
            if partition1.scope & scope != partition2.scope & scope:
                continue  # Only check partitions with the same scope.

            adj_mat = np.zeros((len(partition1.inputs), len(partition2.inputs)), dtype=np.bool_)
            for (i, region1), (j, region2) in itertools.product(
                enumerate(partition1.inputs), enumerate(partition2.inputs)
            ):
                adj_mat[i, j] = bool(region1.scope & region2.scope)  # I.e., scopes intersect.
            # Disable: It's wrong to do @= as the shape does not match.
            adj_mat = adj_mat @ adj_mat.T  # pylint: disable=consider-using-augmented-assign
            # Now we have adjencency from inputs1 (of self) to inputs1. An edge means the two
            # regions must be partitioned together.

            # The number of zero eigen values for the laplacian matrix is the number of connected
            # components in a graph.
            # Cast: Numpy has typing issues.
            deg_mat = np.diag(cast(NDArray[np.int64], adj_mat.sum(axis=1)))
            laplacian: NDArray[np.int64] = deg_mat - adj_mat
            eigen_values = np.linalg.eigvals(laplacian)
            num_connected: int = cast(NDArray[np.int64], np.isclose(eigen_values, 0).sum()).item()
            if num_connected == 1:  # All connected, meaning there's no valid partitioning.
                return False

        return True

    #######################################    Node views    #######################################
    # These are iterable views of the nodes in the RG, available even when the graph is only
    # partially constructed. For efficiency, all these views are iterators (implemented as a
    # container iter or a generator), so that they can be chained for iteration without
    # instantiating intermediate containers.
    # NOTE: There's no ordering graranteed for these views. However RGNode can be sorted if needed.

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
        """Input nodes of the graph, which are guaranteed to be regions."""
        return (node for node in self.region_nodes if not node.inputs)

    @property
    def output_nodes(self) -> Iterator[RegionNode]:
        """Output nodes of the graph, which are guaranteed to be regions."""
        return (node for node in self.region_nodes if not node.outputs)

    @property
    def inner_region_nodes(self) -> Iterator[RegionNode]:
        """Inner (non-input) region nodes in the graph."""
        return (node for node in self.region_nodes if node.inputs)

    ####################################    (De)Serialization    ###################################
    # The RG can be dumped and loaded from json files, which can be useful when we want to save and
    # share it. The load() is another way to construct a RG.

    # TODO: we can only deal with 2-partition here

    def dump(self, filename: str) -> None:
        """Dump the region graph to the json file.

        The file will be opened with mode="w" and encoding="utf-8".

        Args:
            filename (str): The file name for dumping.
        """
        graph_json: RegionGraphJson = {"regions": {}, "graph": []}

        region_id = {node: idx for idx, node in enumerate(self.region_nodes)}
        graph_json["regions"] = {str(idx): list(node.scope) for node, idx in region_id.items()}

        for partition in self.partition_nodes:
            part_inputs = partition.inputs
            assert len(part_inputs) == 2, "We can only dump RG with 2-partitions."
            (part_output,) = partition.outputs
            # partition.outputs is guaranteed to have len==1 by _validate().

            graph_json["graph"].append(
                {
                    "l": region_id[part_inputs[0]],
                    "r": region_id[part_inputs[1]],
                    "p": region_id[part_output],
                }
            )

        # TODO: logging for graph_json?
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_json, f)

    @staticmethod
    def load(filename: str) -> "RegionGraph":
        """Load the region graph from the json file.

        The file will be opened with mode="r" and encoding="utf-8".

        Args:
            filename (str): The file name for loading.

        Returns:
            RegionGraph: The loaded region graph.
        """
        with open(filename, "r", encoding="utf-8") as f:
            graph_json: RegionGraphJson = json.load(f)

        id_region = {int(idx): RegionNode(scope) for idx, scope in graph_json["regions"].items()}

        graph = RegionGraph()

        if not graph_json["graph"]:
            # A corner case: no edge in RG, meaning ther's only one region node, and the following
            # for-loop does not work, so we need to handle it here.
            assert len(id_region) == 1
            graph.add_node(id_region[0])

        for partition in graph_json["graph"]:
            part_inputs = id_region[partition["l"]], id_region[partition["r"]]
            part_output = id_region[partition["p"]]

            partition_node = PartitionNode(part_output.scope)

            for part_input in part_inputs:
                graph.add_edge(part_input, partition_node)
            graph.add_edge(partition_node, part_output)

        return graph.freeze()
