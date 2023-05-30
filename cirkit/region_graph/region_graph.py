from abc import ABC, abstractmethod
from typing import Any, List, Set, Union

import networkx as nx

from .rg_node import PartitionNode, RegionNode, RGNode

# TODO: unify what names to use: sum/region, product/partition, leaf/input
# TODO: confirm the direction of edges
# TODO: directly subclass the DiGraph?


def _get_sums(graph: nx.DiGraph) -> List[RegionNode]:
    return [  # type: ignore[misc]
        n
        for n, d in graph.out_degree()  # type: ignore[misc]
        if d > 0 and isinstance(n, RegionNode)  # type: ignore[misc]
    ]


def _get_products(graph: nx.DiGraph) -> List[PartitionNode]:
    return [n for n in graph.nodes() if isinstance(n, PartitionNode)]  # type: ignore[misc]


def _get_leaves(graph: nx.DiGraph) -> List[RegionNode]:
    return [n for n, d in graph.out_degree() if not d]  # type: ignore[misc]


class RegionGraph(ABC):
    """The base class for region graphs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        """Init shared attrs."""
        super().__init__()
        self._graph = self._construct_graph(*args, **kwargs)  # type: ignore[misc]

    @staticmethod
    @abstractmethod
    def _construct_graph(*args: Any, **kwargs: Any) -> nx.DiGraph:  # type: ignore[misc]
        pass

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
