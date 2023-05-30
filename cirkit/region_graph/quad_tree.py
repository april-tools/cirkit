from typing import List

import networkx as nx

from .poon_domingos_structure import _HypercubeToScopeCache  # TODO: make util?
from .region_graph import RegionGraph
from .rg_node import PartitionNode, RegionNode

# TODO: add routine for add regions->part->reg structure


def merge_2_regions(regions: List[RegionNode], graph: nx.DiGraph) -> RegionNode:
    """Make the structure to connect 2 children.

    Args:
        regions (List[RegionNode]): The children regions.
        graph (nx.DiGraph): The region graph.

    Returns:
        RegionNode: The merged region node.
    """
    assert len(regions) == 2

    scope = regions[0].scope.union(regions[1].scope)
    partition_node = PartitionNode(scope)
    region_node = RegionNode(scope)

    graph.add_edge(regions[0], partition_node)
    graph.add_edge(regions[1], partition_node)
    graph.add_edge(partition_node, region_node)

    return region_node


# pylint: disable-next=too-many-locals
def merge_4_regions_mixed(regions: List[RegionNode], graph: nx.DiGraph) -> RegionNode:
    """Make the structure to connect 4 children with mixed partitions.

    Args:
        regions (List[RegionNode]): The children regions.
        graph (nx.DiGraph): The region graph.

    Returns:
        RegionNode: The merged region node.
    """
    assert len(regions) == 4

    # regions have to have TL, TR, BL, BR
    # MERGE TL & TR, BL & BR
    top_scope = regions[0].scope.union(regions[1].scope)
    bottom_scope = regions[2].scope.union(regions[3].scope)
    top_partition = PartitionNode(top_scope)
    top_region = RegionNode(top_scope)
    bottom_partition = PartitionNode(bottom_scope)
    bottom_region = RegionNode(bottom_scope)

    graph.add_edge(regions[0], top_partition)
    graph.add_edge(regions[1], top_partition)
    graph.add_edge(top_partition, top_region)
    graph.add_edge(regions[2], bottom_partition)
    graph.add_edge(regions[3], bottom_partition)
    graph.add_edge(bottom_partition, bottom_region)

    # MERGE T & B
    whole_scope = top_scope.union(bottom_scope)
    horizontal_patition = PartitionNode(whole_scope)
    graph.add_edge(top_region, horizontal_patition)
    graph.add_edge(bottom_region, horizontal_patition)

    # MERGE TL & BL, TR & BR
    left_scope = regions[0].scope.union(regions[2].scope)
    right_scope = regions[1].scope.union(regions[3].scope)
    left_partition = PartitionNode(left_scope)
    left_region = RegionNode(left_scope)
    right_partition = PartitionNode(right_scope)
    right_region = RegionNode(right_scope)

    graph.add_edge(regions[0], left_partition)
    graph.add_edge(regions[2], left_partition)
    graph.add_edge(left_partition, left_region)
    graph.add_edge(regions[1], right_partition)
    graph.add_edge(regions[3], right_partition)
    graph.add_edge(right_partition, right_region)

    # MERGE L & R
    vertical_partition = PartitionNode(whole_scope)
    graph.add_edge(left_region, vertical_partition)
    graph.add_edge(right_region, vertical_partition)

    # Mix
    whole_region = RegionNode(whole_scope)
    graph.add_edge(horizontal_patition, whole_region)
    graph.add_edge(vertical_partition, whole_region)

    return whole_region


def merge_4_regions_struct_decomp(regions: List[RegionNode], graph: nx.DiGraph) -> RegionNode:
    """Make the structure to connect 4 children with structured-decomposability \
        (horizontal then vertical).

    Args:
        regions (List[RegionNode]): The children regions.
        graph (nx.DiGraph): The region graph.

    Returns:
        RegionNode: The merged region node.
    """
    assert len(regions) == 4
    # Horizontal and then vertical
    top_scope = regions[0].scope.union(regions[1].scope)
    bottom_scope = regions[2].scope.union(regions[3].scope)
    top_partition = PartitionNode(top_scope)
    top_region = RegionNode(top_scope)
    bottom_partition = PartitionNode(bottom_scope)
    bottom_region = RegionNode(bottom_scope)

    graph.add_edge(regions[0], top_partition)
    graph.add_edge(regions[1], top_partition)
    graph.add_edge(top_partition, top_region)
    graph.add_edge(regions[2], bottom_partition)
    graph.add_edge(regions[3], bottom_partition)
    graph.add_edge(bottom_partition, bottom_region)

    # MERGE T & B
    whole_scope = top_scope.union(bottom_scope)
    horizontal_patition = PartitionNode(whole_scope)
    graph.add_edge(top_region, horizontal_patition)
    graph.add_edge(bottom_region, horizontal_patition)

    whole_region = RegionNode(whole_scope)
    graph.add_edge(horizontal_patition, whole_region)

    return whole_region


def square_from_buffer(buffer: List[List[RegionNode]], i: int, j: int) -> List[RegionNode]:
    """Get the children of the current position from the buffer.

    Args:
        buffer (List[List[RegionNode]]): The buffer of all children.
        i (int): The i coordinate currently.
        j (int): The j coordinate currently.

    Returns:
        List[RegionNode]: The children nodes.
    """
    children = [buffer[i][j]]
    # TODO: rewrite: len only useful at 2n-1 boundary
    if len(buffer) > i + 1:
        children.append(buffer[i + 1][j])
    if len(buffer[i]) > j + 1:
        children.append(buffer[i][j + 1])
    if len(buffer) > i + 1 and len(buffer[i]) > j + 1:
        children.append(buffer[i + 1][j + 1])
    return children


class QuadTree(RegionGraph):
    """Some docstring."""

    def __init__(self, width: int, height: int, struct_decomp: bool = False) -> None:
        """Init class.

        Args:
            width (int): Width of scope.
            height (int): Height of scope.
            struct_decomp (bool, optional): Whether structured-decomposability \
                is required. Defaults to False.
        """
        assert width == height and width > 0  # TODO: then we don't need two
        super().__init__(width, height, struct_decomp)

    @staticmethod
    # pylint: disable-next=arguments-differ,too-many-locals
    def _construct_graph(width: int, height: int, struct_decomp: bool = False) -> nx.DiGraph:
        graph = nx.DiGraph()
        shape = (width, height)

        hypercube_to_scope = _HypercubeToScopeCache()

        buffer: List[List[RegionNode]] = [[] for _ in range(width)]

        # Add Leaves
        for i in range(width):
            for j in range(height):
                hypercube = ((i, j), (i + 1, j + 1))

                c_scope = hypercube_to_scope(hypercube, shape)
                c_node = RegionNode(c_scope)
                graph.add_node(c_node)
                buffer[i].append(c_node)

        lr_choice = 0  # left right # TODO: or choose from 0 and 1?
        td_choice = 0  # top down

        old_buffer_width = width
        old_buffer_height = height
        old_buffer = buffer

        # TODO: also no need to have two for h/w
        while old_buffer_width != 1 and old_buffer_height != 1:  # pylint: disable=while-used
            buffer_width = (old_buffer_width + 1) // 2
            buffer_height = (old_buffer_height + 1) // 2

            buffer = [[] for _ in range(buffer_width)]

            for i in range(buffer_width):
                for j in range(buffer_height):
                    regions = square_from_buffer(old_buffer, 2 * i + lr_choice, 2 * j + td_choice)
                    buffer[i].append(
                        regions[0]
                        if len(regions) == 1
                        else merge_2_regions(regions, graph)
                        if len(regions) == 2
                        else merge_4_regions_struct_decomp(regions, graph)
                        if struct_decomp
                        else merge_4_regions_mixed(regions, graph)
                    )

            old_buffer = buffer
            old_buffer_width = buffer_width
            old_buffer_height = buffer_height

        # TODO: do we need this? already defaults to 0
        # for node in get_leaves(graph):
        #     node.einet_address.replica_idx = 0

        return graph
