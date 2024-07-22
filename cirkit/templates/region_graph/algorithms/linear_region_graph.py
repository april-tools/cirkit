from typing import List

import numpy as np

from cirkit.templates.region_graph.region_graph import RegionGraph
from cirkit.templates.region_graph.rg_node import RegionNode


def _partition_node(graph: RegionGraph, node: RegionNode, rm_item: int) -> RegionNode:
    """Partition a region node with one specific item out and add to RG.

    Args:
        graph (RegionGraph): The region graph to hold the partitioning.
        node (RegionNode): The node to partition.
        rm_item (int): The identifier of node to split out.

    Returns:
        RegionNode: The region node need more partitioning.
    """
    scope_list = list(node.scope)

    rm_id = scope_list.index(rm_item)
    scope_list.pop(rm_id)
    region_nodes: List[RegionNode] = [RegionNode(scope_list), RegionNode([rm_item])]

    graph.add_partitioning(node, region_nodes)
    return region_nodes[0]


# DISABLE: We use function name with upper case to mimic a class constructor.
# pylint: disable-next=invalid-name
def LinearRegionGraph(
    num_variables: int, random: bool = True, seed: int = 42, order: List[int] = None
) -> RegionGraph:
    """Construct a Linear RG. (random)

    Args:
                num_variables (int): The number of variables in the RG.
        random (bool): Whether to split out the nodes in a random order.
        order (List[int]): The spcified order to split out nodes.

    Returns:
        RegionGraph: The Linear RG.
    """
    graph = RegionGraph()
    root = RegionNode(range(num_variables))
    graph.add_node(root)

    if random:
        random_state = np.random.RandomState(seed)
        order = list(range(num_variables))
        random_state.shuffle(order)
    else:
        assert order is not None and len(order) == num_variables

    rgn = root
    for i in range(num_variables - 1):
        rgn = _partition_node(graph, rgn, order[i])

    return graph.freeze()
