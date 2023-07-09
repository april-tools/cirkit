import itertools
import random
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .region_graph import RegionGraph
from .rg_node import PartitionNode, RegionNode

# TODO: rework docstrings


def _partition_node_randomly(
    graph: RegionGraph,
    node: RegionNode,
    num_parts: Optional[int] = None,
    proportions: Optional[Sequence[float]] = None,
    repetition: int = 0,
) -> List[RegionNode]:
    """Call partition_on_node with a random partition -- used for random binary trees (RAT-SPNs).

    :param graph: PC graph (DiGraph)
    :param node: node in the graph (DistributionVector)
    :param num_parts: number of parts in the partition. If None, use len proportions (int)
    :param proportions: split proportions of each part, len must be num_parts, can be sum not 1
    :param repetition: the repetition index
    :return: a list of partitioned region nodes
    """  # TODO: rework docstring
    scope = list(node.scope)
    random.shuffle(scope)

    split: NDArray[np.float64]
    if proportions is None:
        assert num_parts, "Must provide at least one of num_parts and proportions."
        split = np.ones(num_parts) / num_parts
    else:
        num_parts = num_parts if num_parts is not None else len(proportions)
        split = np.array(proportions, dtype=np.float64)
        split /= split.sum()  # type: ignore[misc]

    # TODO: arg-type is a bug?
    split_point: List[int] = (
        np.round(split.cumsum() * len(scope))  # type: ignore[arg-type,misc]
        .astype(np.int64)
        .tolist()
    )

    split_point.insert(0, 0)  # add a 0 at the beginning for a complete cumsum

    partition_node = PartitionNode(node.scope)
    graph.add_edge(partition_node, node)  # automatically add_node

    region_nodes: List[RegionNode] = []
    for l, r in zip(split_point[:-1], split_point[1:]):
        assert l < r, f"Over-partitioned with proportions {proportions} on {node}."
        region_node = RegionNode(scope[l:r], replica_idx=repetition)
        graph.add_edge(region_node, partition_node)
        region_nodes.append(region_node)

    return region_nodes


# TODO: do we need to warn invalid name here?
# pylint: disable-next=invalid-name
def RandomBinaryTree(num_vars: int, depth: int, num_repetitions: int) -> RegionGraph:
    """Generate a PC graph via several random binary trees -- RAT-SPNs.

    See
        Random sum-product networks: A simple but effective approach to probabilistic deep learning
        Robert Peharz, Antonio Vergari, Karl Stelzner, Alejandro Molina, Xiaoting Shao,
        Martin Trapp, Kristian Kersting, Zoubin Ghahramani
        UAI 2019


    :param num_vars: number of random variables (int)
    :param depth: splitting depth (int)
    :param num_repetitions: number of repetitions (int)
    :return: generated graph (DiGraph)
    """
    root = RegionNode(range(num_vars))
    graph = RegionGraph()
    graph.add_node(root)

    for repetition in range(num_repetitions):
        nodes = [root]
        for _ in range(depth):
            nodes = list(
                itertools.chain.from_iterable(
                    _partition_node_randomly(graph, n, num_parts=2, repetition=repetition)
                    for n in nodes
                )
            )

    return graph
