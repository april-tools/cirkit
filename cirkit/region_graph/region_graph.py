import itertools
import json
from functools import cached_property
from typing import Dict, FrozenSet, Iterable, List, Set, TypedDict, Union, final, overload

import networkx as nx

from .rg_node import PartitionNode, RegionNode, RGNode

# TODO: unify what names to use: sum/region, product/partition, leaf/input
# TODO: directly subclass the DiGraph?
# TODO: rework docstrings??


class _PartitionJson(TypedDict):
    """The struction of a partitioning in the json file."""

    p: int
    l: int
    r: int


class _RGJson(TypedDict):
    """The structure of region graph json file."""

    regions: Dict[str, List[int]]
    graph: List[_PartitionJson]


@final
class RegionGraph:
    """The base class for region graphs."""

    def __init__(self) -> None:
        """Init graph empty."""
        super().__init__()
        self._graph = nx.DiGraph()

    def add_node(self, node: RGNode) -> None:
        """Add a node to the graph.

        Args:
            node (RGNode): Node to add.
        """
        self._graph.add_node(node)

    @overload
    def add_edge(self, tail: RegionNode, head: PartitionNode) -> None:
        ...

    @overload
    def add_edge(self, tail: PartitionNode, head: RegionNode) -> None:
        ...

    def add_edge(self, tail: RGNode, head: RGNode) -> None:
        """Add an edge to the graph. Nodes are automatically added.

        Args:
            tail (RGNode): The tail of the edge (from).
            head (RGNode): The head of the edge (to).
        """
        self._graph.add_edge(tail, head)
        tail.outputs.append(head)  # type: ignore[misc]
        head.inputs.append(tail)  # type: ignore[misc]

    ###############################    Node views    ###############################

    # For efficiency, all these node views return an iterable (implemented as a generator).
    # Downstream code can wrap them in containers based on the needs. Keep in mind there's no
    # guarantee on the iteration order.

    @property
    def nodes(self) -> Iterable[RGNode]:
        """Get all the nodes in the graph."""
        return iter(self._graph.nodes)  # type: ignore[no-any-return,misc]

    @property
    def region_nodes(self) -> Iterable[RegionNode]:
        """Get region nodes in the graph."""
        return (node for node in self.nodes if isinstance(node, RegionNode))

    @property
    def partition_nodes(self) -> Iterable[PartitionNode]:
        """Get partition nodes in the graph."""
        return (node for node in self.nodes if isinstance(node, PartitionNode))

    @property
    def input_nodes(self) -> Iterable[RegionNode]:
        """Get input nodes of the graph, which are guaranteed to be regions."""
        return (node for node in self.region_nodes if not node.inputs)

    @property
    def output_nodes(self) -> Iterable[RegionNode]:
        """Get output nodes of the graph, which are guaranteed to be regions."""
        return (node for node in self.region_nodes if not node.outputs)

    @property
    def inner_region_nodes(self) -> Iterable[RegionNode]:
        """Get inner (non-input) region nodes in the graph."""
        return (node for node in self.region_nodes if node.inputs)

    ##########################    Structural properties    #########################

    # The RG is expected to be immutable after construction. Also, each of these properties is
    # simply a bool, which is cheap to save. Therefore, we use cached_property to save computation.

    @cached_property
    def is_smooth(self) -> bool:
        """Test smoothness."""
        return all(
            all(partition.scope == region.scope for partition in region.inputs)
            for region in self.inner_region_nodes
        )

    @cached_property
    def is_decomposable(self) -> bool:
        """Test decomposability."""
        return all(
            not any(
                reg1.scope & reg2.scope
                for reg1, reg2 in itertools.combinations(partition.inputs, 2)
            )
            and set().union(*(region.scope for region in partition.inputs)) == partition.scope
            for partition in self.partition_nodes
        )

    @cached_property
    def is_structured_decomposable(self) -> bool:
        """Test structured-decomposability."""
        if not (self.is_smooth and self.is_decomposable):
            return False
        decompositions: Dict[FrozenSet[int], Set[FrozenSet[int]]] = {}
        for partition in self.partition_nodes:
            decomp = set(region.scope for region in partition.inputs)
            if partition.scope not in decompositions:
                decompositions[partition.scope] = decomp
            if decomp != decompositions[partition.scope]:
                return False
        return True

    ##############################    Serialization    #############################

    # TODO: we can only deal with two children here

    def save(self, filename: str) -> None:
        """Save the region graph to json file.

        Args:
            filename (str): The file name to save.
        """
        # TODO: doc the format?
        graph_json: _RGJson = {"regions": {}, "graph": []}

        # TODO: give each node an id as attr? they do have one defined. but what about load?
        region_ids = {node: idx for idx, node in enumerate(self.region_nodes)}
        graph_json["regions"] = {str(idx): list(node.scope) for node, idx in region_ids.items()}

        for partition in self.partition_nodes:
            part_input = partition.inputs
            assert len(part_input) == 2
            part_output = partition.outputs
            assert len(part_output) == 1

            graph_json["graph"].append(
                {
                    "p": region_ids[part_output[0]],
                    "l": region_ids[part_input[0]],
                    "r": region_ids[part_input[1]],
                }
            )

        # TODO: log graph_json
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_json, f)

    @staticmethod
    def load(filename: str) -> "RegionGraph":
        """Load a region graph from json file.

        Args:
            filename (str): The file name to load.

        Returns:
            RegionGraph: The loaded region graph.
        """
        with open(filename, "r", encoding="utf-8") as f:
            graph_json: _RGJson = json.load(f)

        ids_region = {int(idx): RegionNode(scope) for idx, scope in graph_json["regions"].items()}

        graph = RegionGraph()

        if not graph_json["graph"]:  # No edges in graph, meaning only one region node
            assert len(ids_region) == 1
            graph.add_node(ids_region[0])

        for partition in graph_json["graph"]:
            part_output = ids_region[partition["p"]]
            part_input_left = ids_region[partition["l"]]
            part_input_right = ids_region[partition["r"]]

            partition_node = PartitionNode(part_output.scope)

            graph.add_edge(partition_node, part_output)
            graph.add_edge(part_input_left, partition_node)
            graph.add_edge(part_input_right, partition_node)

        # TODO: need to set? but this is wrong for random bin tree
        # for node in get_leaves(graph):
        #     node.einet_address.replica_idx = 0

        return graph

    ##############################    Layerization    ##############################

    # TODO: do we have it here or decouple from RG? also how to properly name "layer"?
    def topological_layers(
        self, bottom_up: bool
    ) -> List[Union[List[RegionNode], List[PartitionNode]]]:
        """Get the layerized computational graph.

        Arranging the PC graph in topological layers -- see Algorithm 1 in the paper.

        Args:
            bottom_up (bool): Whether to build bottom-up or top-down.

        Returns:
            List[Union[List[RegionNode], List[PartitionNode]]]: list of layers, \
                alternating between  DistributionVector and Product layers (list of lists of nodes).
        """
        return (
            self._topological_layers_bottom_up()
            if bottom_up
            else self._topological_layers_top_down()
        )

    def _topological_layers_bottom_up(self) -> List[Union[List[RegionNode], List[PartitionNode]]]:
        """Layerize in the bottom-up manner.

        Returns:
            List[Union[List[RegionNode], List[PartitionNode]]]: \
                Nodes in each layer from input to output.
        """
        inner_region_nodes = sorted(self.inner_region_nodes)  # TODO: why sort?
        partition_nodes = sorted(self.partition_nodes)
        input_nodes = sorted(self.input_nodes)

        visited_nodes: Set[RGNode] = set(input_nodes)
        # TODO: list variance issues
        layers: List[Union[List[RegionNode], List[PartitionNode]]] = [input_nodes]

        num_nodes = len(input_nodes) + len(inner_region_nodes) + len(partition_nodes)

        while len(visited_nodes) != num_nodes:  # pylint: disable=while-used
            partition_layer = [
                partition
                for partition in partition_nodes
                if partition not in visited_nodes
                and all(region in visited_nodes for region in partition.inputs)
            ]
            partition_layer = sorted(partition_layer)
            layers.append(partition_layer)
            visited_nodes.update(partition_layer)

            region_layer = [
                region
                for region in inner_region_nodes
                if region not in visited_nodes
                and all(partition in visited_nodes for partition in region.inputs)
            ]
            region_layer = sorted(region_layer)
            layers.append(region_layer)
            visited_nodes.update(region_layer)

        return layers

    def _topological_layers_top_down(self) -> List[Union[List[RegionNode], List[PartitionNode]]]:
        """Layerize in the top-down manner.

        Returns:
            List[Union[List[RegionNode], List[PartitionNode]]]: \
                Nodes in each layer from input to output.
        """
        inner_region_nodes = sorted(self.inner_region_nodes)  # TODO: why sort?
        partition_nodes = sorted(self.partition_nodes)
        input_nodes = sorted(self.input_nodes)

        visited_nodes: Set[RGNode] = set()
        # TODO: list variance issues
        layers_inv: List[Union[List[RegionNode], List[PartitionNode]]] = []

        num_inner_nodes = len(inner_region_nodes) + len(partition_nodes)

        while len(visited_nodes) != num_inner_nodes:  # pylint: disable=while-used
            # TODO: repeated conditions. can we fuse the layer for reg and part?
            region_layer = [
                region
                for region in inner_region_nodes
                if region not in visited_nodes
                and all(partition in visited_nodes for partition in region.outputs)
            ]
            region_layer = sorted(region_layer)
            layers_inv.append(region_layer)
            visited_nodes.update(region_layer)

            partition_layer = [
                partition
                for partition in partition_nodes
                if partition not in visited_nodes
                and all(region in visited_nodes for region in partition.outputs)
            ]
            partition_layer = sorted(partition_layer)
            layers_inv.append(partition_layer)
            visited_nodes.update(partition_layer)

        layers_inv.append(input_nodes)
        return list(reversed(layers_inv))
