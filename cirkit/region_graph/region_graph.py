import json
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
    overload,
)
from typing_extensions import Self  # TODO: in typing from 3.11

import networkx as nx

from .rg_node import PartitionNode, RegionNode, RGNode

# TODO: unify what names to use: sum/region, product/partition, leaf/input
# TODO: confirm the direction of edges
# TODO: directly subclass the DiGraph?
# TODO: rework docstrings??


class _RGJson(TypedDict):
    """The structure of region graph json file."""

    regions: Dict[str, List[int]]
    graph: List[Dict[str, int]]


class RegionGraph:
    """The base class for region graphs."""

    # TODO: is it a good practice to allow any args? what about for inherit?
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        """Init graph with any args/kwargs given."""
        # TODO: careful with this way to pass args when multi-inherit then it's a must
        super().__init__()
        self._graph = nx.DiGraph(*args, **kwargs)  # type: ignore[misc]

    # TODO: do we need a class for node view?
    # TODO: do we return a generic container or concrete class?
    @property
    def nodes(self) -> Collection[RGNode]:
        """Get all the nodes in the graph."""
        nodes: Collection[RGNode] = self._graph.nodes  # DiGraph.nodes is both set and dict
        return nodes

    @property
    def region_nodes(self) -> Collection[RegionNode]:
        """Get region nodes in the graph."""
        return [node for node in self.nodes if isinstance(node, RegionNode)]

    @property
    def partition_nodes(self) -> Collection[PartitionNode]:
        """Get partition nodes in the graph."""
        return [node for node in self.nodes if isinstance(node, PartitionNode)]

    @property
    def input_nodes(self) -> Collection[RegionNode]:
        """Get input nodes of the graph, which are regions."""
        node_indegs: Iterable[Tuple[RGNode, int]] = self._graph.in_degree
        # enforce type because we know they're regions
        return [cast(RegionNode, node) for node, deg in node_indegs if not deg]

    @property
    def output_nodes(self) -> Collection[RegionNode]:
        """Get output nodes of the graph, which are regions."""
        node_outdegs: Iterable[Tuple[RGNode, int]] = self._graph.out_degree
        # enforce type because we know they're regions
        return [cast(RegionNode, node) for node, deg in node_outdegs if not deg]

    @property
    def inner_region_nodes(self) -> Collection[RegionNode]:
        """Get inner (non-input) region nodes in the graph."""
        node_indegs: Iterable[Tuple[RGNode, int]] = self._graph.in_degree
        return [node for node, deg in node_indegs if isinstance(node, RegionNode) and deg]

    @overload
    def get_node_input(self, node: PartitionNode) -> Iterable[RegionNode]:
        ...

    @overload
    def get_node_input(self, node: RegionNode) -> Iterable[PartitionNode]:
        ...

    # TODO: return list or iter? predecessors itself returns an iterator
    def get_node_input(self, node: RGNode) -> Iterable[RGNode]:
        """Get input nodes of a node.

        Args:
            node (RGNode): The node queried.

        Returns:
            Iterable[RGNode]: The inputs to the node.
        """
        inputs: Iterable[RGNode] = self._graph.predecessors(node)
        return inputs

    @overload
    def get_node_output(self, node: PartitionNode) -> Iterable[RegionNode]:
        ...

    @overload
    def get_node_output(self, node: RegionNode) -> Iterable[PartitionNode]:
        ...

    # TODO: return list or iter?
    def get_node_output(self, node: RGNode) -> Iterable[RGNode]:
        """Get output nodes of a node.

        Args:
            node (RGNode): The node queried.

        Returns:
            Iterable[RGNode]: The outputs to the node.
        """
        outputs: Iterable[RGNode] = self._graph.successors(node)
        return outputs

    def save(self, filename: str) -> None:
        """Save the region graph to json file.

        Args:
            filename (str): The file name to save.
        """
        # TODO: doc the format?
        graph_json: _RGJson = {"regions": {}, "graph": []}

        # TODO: give each node an id as attr? they do have one defined. but what about load?
        region_ids = {node: n for n, node in enumerate(self.region_nodes)}
        graph_json["regions"] = {str(n): list(node.scope) for node, n in region_ids.items()}

        for partition_node in self.partition_nodes:
            part_input = list(self.get_node_input(partition_node))
            assert len(part_input) == 2
            part_output = list(self.get_node_output(partition_node))
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

    @classmethod
    def load(cls, filename: str) -> Self:
        """Load a region graph from json file.

        Args:
            filename (str): The file name to load.

        Raises:
            NotImplementedError: It is not implemented for children classes.

        Returns:
            RegionGraph: The loaded region graph.
        """
        # TODO: is this a bug of mypy? caused by __init__ include Any
        if cls is not RegionGraph:  # type: ignore[misc]
            raise NotImplementedError(
                "Must be called as `RegionGraph.load()` instead of from child class."
            )

        with open(filename, "r", encoding="utf-8") as f:
            graph_json: _RGJson = json.load(f)

        ids_region = {int(idx): RegionNode(scope) for idx, scope in graph_json["regions"].items()}

        graph = nx.DiGraph()

        if not graph_json["graph"]:  # Only the root region is present
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

        return cls(graph)

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

    # TODO: the 4 `pylint disable not-an-iterable` is due to a bug in astroid in 2021
    # check the progress here https://github.com/pylint-dev/astroid/issues/1015

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
                # pylint: disable-next=not-an-iterable
                and all(region in visited_nodes for region in self.get_node_input(partition))
            ]
            partition_layer = sorted(partition_layer)
            layers.append(partition_layer)
            visited_nodes.update(partition_layer)

            region_layer = [
                region
                for region in inner_region_nodes
                if region not in visited_nodes
                # pylint: disable-next=not-an-iterable
                and all(partition in visited_nodes for partition in self.get_node_input(region))
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
                # pylint: disable-next=not-an-iterable
                and all(partition in visited_nodes for partition in self.get_node_output(region))
            ]
            region_layer = sorted(region_layer)
            layers_inv.append(region_layer)
            visited_nodes.update(region_layer)

            partition_layer = [
                partition
                for partition in partition_nodes
                if partition not in visited_nodes
                # pylint: disable-next=not-an-iterable
                and all(region in visited_nodes for region in self.get_node_output(partition))
            ]
            partition_layer = sorted(partition_layer)
            layers_inv.append(partition_layer)
            visited_nodes.update(partition_layer)

        layers_inv.append(input_nodes)
        return list(reversed(layers_inv))
