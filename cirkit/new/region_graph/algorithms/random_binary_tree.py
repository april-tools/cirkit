import random
from typing import List, Optional, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from cirkit.new.region_graph.region_graph import RegionGraph
from cirkit.new.region_graph.rg_node import RegionNode


def _partition_node_randomly(
    graph: RegionGraph,
    node: RegionNode,
    num_parts: Optional[int] = None,
    proportions: Optional[Sequence[float]] = None,
) -> List[RegionNode]:
    """Partition a region node randomly and add to RG.

    Args:
        graph (RegionGraph): The region graph to hold the partitioning.
        node (RegionNode): The node to partition.
        num_parts (Optional[int], optional): The number of parts to partition. If not provided, \
            will be inferred from proportions. Defaults to None.
        proportions (Optional[Sequence[float]], optional): The proportions of each part, can be \
            unnormalized. If not provided, will equally divide to num_parts. Defaults to None.

    Returns:
        List[RegionNode]: The region nodes forming the partitioning.
    """
    scope_list = list(node.scope)
    random.shuffle(scope_list)

    # ANNOTATE: Numpy has typing issues.
    split: NDArray[np.float64]  # Unnormalized split points including 0 and 1.
    if proportions is None:
        assert num_parts, "Must provide at least one of num_parts and proportions."
        split = np.arange(num_parts + 1, dtype=np.float64)
    else:
        split = np.array([0.0] + list(proportions), dtype=np.float64).cumsum()

    # ANNOTATE: ndarray.tolist gives Any.
    # CAST: Numpy has typing issues.
    # IGNORE: Numpy has typing issues.
    split_point: List[int] = (
        cast(NDArray[np.float64], split / split[-1] * len(scope_list))  # type: ignore[misc]
        .round()
        .astype(np.int64)
        .tolist()
    )

    # ANNOTATE: Specify content for empty container.
    region_nodes: List[RegionNode] = []
    for l, r in zip(split_point[:-1], split_point[1:]):
        # FUTURE: for l, r in itertools.pairwise(split_point) in 3.10
        if l < r:  # A region must have as least one var, otherwise we skip it.
            region_node = RegionNode(scope_list[l:r])
            region_nodes.append(region_node)

    if len(region_nodes) == 1:
        # Only one region, meaning cannot partition anymore, and we just keep the original node as
        # the leaf.
        return [node]

    graph.add_partitioning(node, region_nodes)
    return region_nodes


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def RandomBinaryTree(*, num_vars: int, depth: int, num_repetitions: int) -> RegionGraph:
    """Construct a RG with random binary trees.

    See:
        Random sum-product networks: A simple but effective approach to probabilistic deep learning.
        Robert Peharz, Antonio Vergari, Karl Stelzner, Alejandro Molina, Xiaoting Shao,
        Martin Trapp, Kristian Kersting, Zoubin Ghahramani.
        UAI 2019.

    Args:
        num_vars (int): The number of variables in the RG.
        depth (int): The depth of the binary tree.
        num_repetitions (int): The number of repetitions of binary trees (degree of root).

    Returns:
        RegionGraph: The RBT RG.
    """
    graph = RegionGraph()
    root = RegionNode(range(num_vars))
    graph.add_node(root)

    for _ in range(num_repetitions):
        layer = [root]
        for _ in range(depth):
            layer = sum((_partition_node_randomly(graph, node, num_parts=2) for node in layer), [])

    return graph.freeze()
