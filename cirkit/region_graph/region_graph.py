import json
from typing import Any, Dict, List, Set, TypedDict, Union

import networkx as nx

from .rg_node import PartitionNode, RegionNode, RGNode

# TODO: unify what names to use: sum/region, product/partition, leaf/input
# TODO: confirm the direction of edges
# TODO: directly subclass the DiGraph?


class _RGJson(TypedDict):
    """The structure of region graph json file."""

    regions: Dict[str, List[int]]
    graph: List[Dict[str, int]]


def _get_sums(graph: nx.DiGraph) -> List[RegionNode]:
    return [  # type: ignore[misc]
        n
        for n, d in graph.out_degree()  # type: ignore[misc]
        if d > 0 and isinstance(n, RegionNode)  # type: ignore[misc]
    ]


def _get_products(graph: nx.DiGraph) -> List[PartitionNode]:
    return [n for n in graph.nodes() if isinstance(n, PartitionNode)]  # type: ignore[misc]


def _get_region_nodes(graph: nx.DiGraph) -> List[RegionNode]:
    return [n for n in graph.nodes() if isinstance(n, RegionNode)]  # type: ignore[misc]


# def _get_roots(graph: nx.DiGraph) -> List[RegionNode]:
#     return [n for n, d in graph.in_degree() if not d]  # type: ignore[misc]


def _get_leaves(graph: nx.DiGraph) -> List[RegionNode]:
    return [n for n, d in graph.out_degree() if not d]  # type: ignore[misc]


class RegionGraph:
    """The base class for region graphs."""

    # TODO: is it a good practice to allow any args? what about for inherit?
    def __init__(self, *_: Any, **__: Any) -> None:  # type: ignore[misc]
        """Init shared attrs."""
        super().__init__()
        self._graph = nx.DiGraph()

    def save(self, filename: str) -> None:
        """Save the region graph to json file.

        Args:
            filename (str): The file name to save.
        """
        graph_json: _RGJson = {"regions": {}, "graph": []}

        regions = _get_region_nodes(self._graph)
        regions_dict = {node: n for n, node in enumerate(regions)}
        graph_json["regions"] = {str(n): list(node.scope) for node, n in regions_dict.items()}

        products = _get_products(self._graph)

        for product in products:
            children: List[RegionNode] = list(self._graph.successors(product))  # type: ignore[misc]
            parent: List[RegionNode] = list(self._graph.predecessors(product))  # type: ignore[misc]
            assert len(children) == 2
            assert len(parent) == 1

            graph_json["graph"].append(
                {
                    "p": regions_dict[parent[0]],
                    "l": regions_dict[children[0]],
                    "r": regions_dict[children[1]],
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
        regions = graph_json["regions"]
        region_nodes: Dict[str, RegionNode] = {
            idx: RegionNode(scope) for idx, scope in regions.items()
        }

        graph = nx.DiGraph()

        for partition in graph_json["graph"]:
            parent_node = region_nodes[str(partition["p"])]
            left_child_node = region_nodes[str(partition["l"])]
            right_child_node = region_nodes[str(partition["r"])]

            product_node = PartitionNode(parent_node.scope)

            graph.add_edge(product_node, parent_node)
            graph.add_edge(left_child_node, product_node)
            graph.add_edge(right_child_node, product_node)

        # TODO: need to set?
        # for node in get_leaves(graph):
        #     node.einet_address.replica_idx = 0

        rg = RegionGraph()
        rg._graph = graph  # pylint: disable=protected-access
        # TODO: how to fix this warning?
        return rg

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
        sums = list(sorted(_get_sums(self._graph)))  # TODO: why sort?
        products = list(sorted(_get_products(self._graph)))
        leaves = list(sorted(_get_leaves(self._graph)))

        visited_nodes: Set[RGNode] = set(leaves)
        # TODO: list variance issues
        layers: List[Union[List[RegionNode], List[PartitionNode]]] = [leaves]

        num_nodes = len(leaves) + len(sums) + len(products)

        while len(visited_nodes) != num_nodes:  # pylint: disable=while-used
            product_layer = [
                p
                for p in products
                if p not in visited_nodes
                and all(s in visited_nodes for s in self._graph.successors(p))  # type: ignore[misc]
            ]
            product_layer = sorted(product_layer)
            layers.append(product_layer)
            visited_nodes.update(product_layer)

            sum_layer = [
                s
                for s in sums
                if s not in visited_nodes
                and all(p in visited_nodes for p in self._graph.successors(s))  # type: ignore[misc]
            ]
            sum_layer = sorted(sum_layer)
            layers.append(sum_layer)
            visited_nodes.update(sum_layer)

        return layers

    def _topological_layers_top_down(self) -> List[Union[List[RegionNode], List[PartitionNode]]]:
        """Layerize in the top-down manner.

        Returns:
            List[Union[List[RegionNode], List[PartitionNode]]]: \
                Nodes in each layer from input to output.
        """
        sums = list(sorted(_get_sums(self._graph)))  # TODO: why sort?
        products = list(sorted(_get_products(self._graph)))
        leaves = list(sorted(_get_leaves(self._graph)))

        visited_nodes: Set[RGNode] = set()
        # TODO: list variance issues
        layers: List[Union[List[RegionNode], List[PartitionNode]]] = []

        num_internal_nodes = len(sums) + len(products)

        while len(visited_nodes) != num_internal_nodes:  # pylint: disable=while-used
            # TODO: repeated conditions
            sum_layer = [
                s
                for s in sums
                if s not in visited_nodes
                and all(  # type: ignore[misc]
                    p in visited_nodes for p in self._graph.predecessors(s)  # type: ignore[misc]
                )
            ]
            sum_layer = sorted(sum_layer)
            layers.insert(0, sum_layer)
            visited_nodes.update(sum_layer)

            product_layer = [
                p
                for p in products
                if p not in visited_nodes
                and all(  # type: ignore[misc]
                    s in visited_nodes for s in self._graph.predecessors(p)  # type: ignore[misc]
                )
            ]
            product_layer = sorted(product_layer)
            layers.insert(0, product_layer)
            visited_nodes.update(product_layer)

        layers.insert(0, leaves)
        return layers
